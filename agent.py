"""Customer support agent.

Polls a Chatwoot inbox, works out what each new message wants, and either
answers it from a markdown knowledge file or issues a Stripe refund.

The whole system is this file. Two things are deliberately kept apart:

  * the model decides *what the customer is asking for*
  * plain code decides *whether we give them money*

Everything the agent talks to is passed in, so the logic can be tested
without a Chatwoot instance, a Groq key or a Stripe account.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol, Sequence

import httpx

log = logging.getLogger("support-agent")


# --------------------------------------------------------------------------
# Configuration. Reading .env here keeps the whole system in one file with no
# extra dependency. Real environment variables always win over the file.
# --------------------------------------------------------------------------


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def env_value(*names: str, default: str = "") -> str:
    """The first of these variables that is set.

    Lets the model settings be named for what they are rather than for one
    provider, while still honouring the older OPENROUTER_* names.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def require_env(*names: str, hint: str = "") -> str:
    value = env_value(*names)
    if not value:
        message = f"{names[0]} is not set. Add it to .env or export it."
        raise SystemExit(f"{message} {hint}".strip())
    return value


# --------------------------------------------------------------------------
# Refund policy. These are the numbers that decide the 90/10 split, and they
# live here in the open rather than inside a prompt.
# --------------------------------------------------------------------------

MAX_AUTO_REFUND_CENTS = 5_000  # $50.00
MAX_CHARGE_AGE_DAYS = 30
MIN_CONFIDENCE = 0.8


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    id: int
    content: str
    incoming: bool
    private: bool = False
    # Chatwoot files things other than conversation turns as messages: status
    # changes and label edits as `activity`, and its own prompts to the visitor
    # (such as asking for an email address) as `template`. Neither is somebody
    # speaking.
    activity: bool = False
    template: bool = False


@dataclass(frozen=True)
class Conversation:
    id: int
    contact_email: str | None
    messages: tuple[Message, ...]

    @property
    def latest(self) -> Message | None:
        """The newest message that represents somebody speaking.

        Only the customer and us count. Chatwoot's own entries are skipped, and
        both kinds matter in practice: a label added after a customer writes in
        would otherwise look like a reply, and the widget appends its
        "give the team a way to reach you" prompt as a template message
        immediately after a visitor's first message, which silenced every
        widget conversation until this excluded it.
        """
        spoken = [
            message
            for message in self.messages
            if not message.activity and not message.template
        ]
        return spoken[-1] if spoken else None


@dataclass(frozen=True)
class Article:
    """One help centre article: a fact the agent is allowed to state."""

    title: str
    content: str


@dataclass(frozen=True)
class Charge:
    id: str
    amount_cents: int
    created: datetime
    refunded: bool  # any refund at all, full or partial
    prior_refund_count: int
    # Other charges on the same customer that could equally be the one they
    # mean. Customers rarely say which payment they are talking about, so more
    # than one candidate is treated as a question for a human, not a guess.
    sibling_unrefunded_count: int = 0


@dataclass(frozen=True)
class Classification:
    intent: str  # "refund" | "support"
    confidence: float


@dataclass(frozen=True)
class Decision:
    auto_approve: bool
    reason: str


# --------------------------------------------------------------------------
# Ports. Real implementations are further down; tests substitute their own.
# --------------------------------------------------------------------------


class Inbox(Protocol):
    def open_conversations(self) -> list[Conversation]: ...
    def send_reply(self, conversation_id: int, content: str) -> None: ...
    def add_private_note(self, conversation_id: int, content: str) -> None: ...
    def resolve(self, conversation_id: int) -> None: ...


class Payments(Protocol):
    def latest_charge(self, email: str) -> Charge | None: ...
    def refund(self, charge_id: str, idempotency_key: str) -> str: ...


class Classifier(Protocol):
    def __call__(self, message: str) -> Classification: ...


class Answerer(Protocol):
    def __call__(
        self, message: str, knowledge: str, articles: Sequence[Article]
    ) -> str: ...


class HelpCentre(Protocol):
    def relevant(self, question: str, limit: int = 3) -> list[Article]: ...


# --------------------------------------------------------------------------
# The refund rules. Pure function, no I/O, no model.
# --------------------------------------------------------------------------


def refund_decision(
    charge: Charge | None,
    confidence: float,
    now: datetime | None = None,
) -> Decision:
    """Decide whether a refund can go through without a human.

    Every condition has to hold. The first one that fails is the reason,
    which is what gets shown to whoever picks the ticket up.
    """
    now = now or datetime.now(timezone.utc)

    if charge is None:
        return Decision(False, "no charge found for this customer")

    if confidence < MIN_CONFIDENCE:
        return Decision(
            False,
            f"classifier confidence {confidence:.2f} below {MIN_CONFIDENCE}",
        )

    if charge.refunded:
        return Decision(False, f"charge {charge.id} has already been refunded")

    if charge.sibling_unrefunded_count > 0:
        total = charge.sibling_unrefunded_count + 1
        return Decision(
            False,
            f"customer has {total} unrefunded charges, cannot tell which one "
            "they mean",
        )

    if charge.amount_cents > MAX_AUTO_REFUND_CENTS:
        return Decision(
            False,
            f"amount {charge.amount_cents / 100:.2f} over the "
            f"{MAX_AUTO_REFUND_CENTS / 100:.2f} auto-approval limit",
        )

    age = now - charge.created
    if age > timedelta(days=MAX_CHARGE_AGE_DAYS):
        return Decision(
            False, f"charge is {age.days} days old, limit is {MAX_CHARGE_AGE_DAYS}"
        )

    if charge.prior_refund_count > 0:
        return Decision(
            False, f"customer has {charge.prior_refund_count} previous refund(s)"
        )

    return Decision(True, "within auto-approval policy")


# --------------------------------------------------------------------------
# Deciding whether a conversation is ours to touch
# --------------------------------------------------------------------------


class HandledMessages:
    """The messages this process has already acted on.

    Deciding by "who spoke last" is correct but not instantaneous: between
    classifying a message and posting the reply there is a window where the
    customer's message is still the newest thing in the conversation. A second
    overlapping pass, or a operator who starts the agent twice, would pick it
    up again. Remembering the ids closes that window.

    In memory only, which is deliberate for a demo but worth being clear about:
    it is gone on restart. The Stripe idempotency key is what makes a restart
    mid-refund safe, not this.
    """

    def __init__(self, capacity: int = 5_000):
        self._capacity = capacity
        self._ids: dict[int, None] = {}  # insertion-ordered, used as a bounded set

    def __contains__(self, message_id: int) -> bool:
        return message_id in self._ids

    def add(self, message_id: int) -> None:
        self._ids[message_id] = None
        while len(self._ids) > self._capacity:
            self._ids.pop(next(iter(self._ids)))

    def __len__(self) -> int:
        return len(self._ids)


def refund_idempotency_key(conversation_id: int, message_id: int) -> str:
    """A stable key for one refund request.

    Derived from the conversation and the message that asked for it, so every
    retry of the same request carries the same key and Stripe returns the
    original refund instead of issuing a second one. No personal data in it,
    and well inside Stripe's 255 character limit.
    """
    return f"refund-conv{conversation_id}-msg{message_id}"


def needs_reply(conversation: Conversation) -> bool:
    """True when the customer spoke last.

    Polling means we see the same conversation repeatedly. Rather than track
    what we have already handled, we look at who spoke last: if the newest
    message is ours, there is nothing to do.
    """
    latest = conversation.latest
    return latest is not None and latest.incoming


# --------------------------------------------------------------------------
# Handling one conversation
# --------------------------------------------------------------------------


def handle_conversation(
    conversation: Conversation,
    *,
    inbox: Inbox,
    payments: Payments,
    classify: Classifier,
    answer: Answerer,
    knowledge: str,
    help_centre: HelpCentre | None = None,
    handled: HandledMessages | None = None,
    now: datetime | None = None,
) -> str:
    """Process a single conversation. Returns the action taken, for logging."""
    latest = conversation.latest
    if latest is None or not latest.incoming:
        return "skipped"

    if handled is not None and latest.id in handled:
        return "already handled"

    result = classify(latest.content)
    log.info(
        "conversation %s classified as %s (%.2f)",
        conversation.id,
        result.intent,
        result.confidence,
    )

    if result.intent == "refund":
        action = _handle_refund(
            conversation,
            result,
            message_id=latest.id,
            inbox=inbox,
            payments=payments,
            now=now,
        )
    else:
        articles = help_centre.relevant(latest.content) if help_centre else []
        if articles:
            log.info(
                "conversation %s answered from %s",
                conversation.id,
                ", ".join(article.title for article in articles),
            )
        reply = answer(latest.content, knowledge, articles)
        inbox.send_reply(conversation.id, reply)
        # Resolved rather than left open: Chatwoot reopens a conversation as
        # soon as the customer writes again, so nothing is lost and the inbox
        # does not fill up with conversations we have already dealt with.
        inbox.resolve(conversation.id)
        action = "answered"

    # Recorded only once the work succeeded. Marking it earlier would mean a
    # transient model or network failure silently swallowed the message.
    if handled is not None:
        handled.add(latest.id)
    return action


def _handle_refund(
    conversation: Conversation,
    result: Classification,
    *,
    message_id: int,
    inbox: Inbox,
    payments: Payments,
    now: datetime | None,
) -> str:
    charge = (
        payments.latest_charge(conversation.contact_email)
        if conversation.contact_email
        else None
    )
    decision = refund_decision(charge, result.confidence, now=now)

    if not decision.auto_approve:
        # Tell the customer something, then tell the team why. Posting only the
        # note leaves the customer staring at silence, which is indistinguishable
        # from a broken agent: they asked for a refund and nothing came back. The
        # acknowledgement deliberately says nothing about the outcome, since
        # whether to refund is the human's decision, not ours.
        inbox.send_reply(
            conversation.id,
            "Thanks for getting in touch. I've passed this to a colleague who "
            "will look at your refund and come back to you shortly.",
        )
        # Deliberately left open, and not resolved: a person still has to act.
        inbox.add_private_note(
            conversation.id,
            "Refund request held for review.\n"
            f"Reason: {decision.reason}\n\n"
            "The customer has been told a colleague will follow up.",
        )
        return "flagged"

    assert charge is not None  # guaranteed by refund_decision
    # If this same request is ever sent twice, whether by an overlapping pass
    # or by a restart after a crash between the refund and the reply, Stripe
    # returns the original refund rather than issuing another one.
    refund_id = payments.refund(
        charge.id, refund_idempotency_key(conversation.id, message_id)
    )
    log.info("refunded charge %s (%s)", charge.id, refund_id)
    inbox.send_reply(
        conversation.id,
        f"That's refunded, {charge.amount_cents / 100:.2f} is on its way back to "
        "your original payment method. It usually lands within a few days.",
    )
    inbox.resolve(conversation.id)
    return "refunded"


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_once(
    *,
    inbox: Inbox,
    payments: Payments,
    classify: Classifier,
    answer: Answerer,
    knowledge: str,
    help_centre: HelpCentre | None = None,
    handled: HandledMessages | None = None,
) -> list[str]:
    actions = []
    for conversation in inbox.open_conversations():
        if not needs_reply(conversation):
            continue
        actions.append(
            handle_conversation(
                conversation,
                inbox=inbox,
                payments=payments,
                classify=classify,
                answer=answer,
                knowledge=knowledge,
                help_centre=help_centre,
                handled=handled,
            )
        )
    return actions


def run_forever(
    *,
    knowledge_path: Path,
    interval_seconds: int = 5,
    sleep=time.sleep,
    **kwargs,
) -> None:
    """Poll until interrupted.

    The knowledge file is read on every pass rather than once at startup, so
    editing it changes the agent's answers within one interval. That is the
    whole point of keeping the guidance in a text file.
    """
    handled = kwargs.pop("handled", None) or HandledMessages()
    while True:
        try:
            run_once(knowledge=knowledge_path.read_text(), handled=handled, **kwargs)
        except Exception:  # a bad poll should not kill the agent
            log.exception("poll failed")
        sleep(interval_seconds)


# --------------------------------------------------------------------------
# Chatwoot
# --------------------------------------------------------------------------


class ChatwootInbox:
    def __init__(
        self,
        base_url: str,
        account_id: str,
        token: str,
        transport: httpx.BaseTransport | None = None,
    ):
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api/v1/accounts/{account_id}",
            headers={"api_access_token": token},
            timeout=20,
            transport=transport,
        )

    def open_conversations(self) -> list[Conversation]:
        response = self._client.get("/conversations", params={"status": "open"})
        response.raise_for_status()
        payload = response.json()["data"]["payload"]
        return [self._conversation(entry["id"], entry) for entry in payload]

    def _conversation(self, conversation_id: int, entry: dict) -> Conversation:
        response = self._client.get(f"/conversations/{conversation_id}/messages")
        response.raise_for_status()
        # message_type: 0 incoming, 1 outgoing, 2 activity, 3 template.
        # Private notes are kept. They are how we mark that a refund has
        # already been handed to a human, so dropping them here would make
        # the agent flag the same conversation on every single poll.
        messages = tuple(
            Message(
                id=item["id"],
                content=item.get("content") or "",
                incoming=item.get("message_type") == 0,
                private=bool(item.get("private")),
                activity=item.get("message_type") == 2,
                template=item.get("message_type") == 3,
            )
            for item in response.json()["payload"]
        )
        contact = (entry.get("meta") or {}).get("sender") or {}
        return Conversation(
            id=conversation_id,
            contact_email=contact.get("email"),
            messages=messages,
        )

    def send_reply(self, conversation_id: int, content: str) -> None:
        self._post(conversation_id, content, private=False)

    def add_private_note(self, conversation_id: int, content: str) -> None:
        self._post(conversation_id, content, private=True)

    def resolve(self, conversation_id: int) -> None:
        response = self._client.post(
            f"/conversations/{conversation_id}/toggle_status",
            json={"status": "resolved"},
        )
        response.raise_for_status()

    def _post(self, conversation_id: int, content: str, *, private: bool) -> None:
        response = self._client.post(
            f"/conversations/{conversation_id}/messages",
            json={
                "content": content,
                "message_type": "outgoing",
                "private": private,
            },
        )
        response.raise_for_status()


# --------------------------------------------------------------------------
# The help centre, as the source of facts
# --------------------------------------------------------------------------

# Words too common to say anything about which article is relevant.
STOPWORDS = frozenset(
    """a about after all also am an and any are as at be been but by can cannot
    could did do does doing for from get got had has have how i if in into is it
    its just me my no not of on or our out over please so some than that the
    their them then there these they this to too up us was we were what when
    where which who why will with would you your""".split()
)


def keywords(text: str) -> set[str]:
    """The words in a question worth matching on."""
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {word for word in cleaned.split() if len(word) > 2 and word not in STOPWORDS}


def overlap(wanted: set[str], text: str) -> int:
    """How many of the wanted words appear in this text.

    Words count as the same when one is a prefix of the other, from four
    characters. Exact matching would miss the obvious: someone asking to
    "cancel" would not find "Cancelling your subscription" or the word
    "cancellation" in its body. This is a blunt instrument compared with proper
    stemming, and it is the right amount of machinery for a few dozen articles.
    """
    present = keywords(text)
    hits = 0
    for want in wanted:
        if want in present:
            hits += 1
            continue
        if len(want) >= 4 and any(
            word.startswith(want) or (len(word) >= 4 and want.startswith(word))
            for word in present
        ):
            hits += 1
    return hits


class ChatwootHelpCentre:
    """Finds the articles relevant to a question.

    Chatwoot's own `?query=` is a phrase match rather than a search: asking it
    for "How do I cancel my subscription?" returns nothing even though an
    article called "Cancelling your subscription" exists. So the articles are
    fetched once, cached, and scored here on word overlap, with a title match
    counting for more than a body match.

    That is the right shape at this size. A corpus too large to fetch is where
    embeddings and a vector store start to earn their keep.
    """

    TITLE_WEIGHT = 3

    def __init__(
        self,
        base_url: str,
        account_id: str,
        token: str,
        portal_slug: str,
        ttl_seconds: int = 300,
        transport: httpx.BaseTransport | None = None,
        now=time.monotonic,
    ):
        self._portal = portal_slug
        self._ttl = ttl_seconds
        self._now = now
        self._cached: list[Article] = []
        self._fetched_at: float | None = None
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api/v1/accounts/{account_id}",
            headers={"api_access_token": token},
            timeout=20,
            transport=transport,
        )

    def articles(self) -> list[Article]:
        fresh = (
            self._fetched_at is not None
            and self._now() - self._fetched_at < self._ttl
        )
        if fresh:
            return self._cached

        response = self._client.get(f"/portals/{self._portal}/articles")
        response.raise_for_status()
        body = response.json()
        rows = body.get("payload", body) if isinstance(body, dict) else body
        self._cached = [
            Article(title=row.get("title") or "", content=row.get("content") or "")
            for row in rows
            if isinstance(row, dict) and (row.get("title") or row.get("content"))
        ]
        self._fetched_at = self._now()
        log.info("help centre: %s articles loaded", len(self._cached))
        return self._cached

    def relevant(self, question: str, limit: int = 3) -> list[Article]:
        wanted = keywords(question)
        if not wanted:
            return []

        scored = []
        for article in self.articles():
            title_hits = overlap(wanted, article.title)
            body_hits = overlap(wanted, article.content)
            score = title_hits * self.TITLE_WEIGHT + body_hits
            if score:
                scored.append((score, article))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [article for _, article in scored[:limit]]


# --------------------------------------------------------------------------
# Stripe
# --------------------------------------------------------------------------


class StripePayments:
    """Stripe, over plain HTTP. The key should be a restricted key that can
    only read charges and write refunds."""

    def __init__(self, secret_key: str, transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(
            base_url="https://api.stripe.com/v1",
            headers={"Authorization": f"Bearer {secret_key}"},
            timeout=20,
            transport=transport,
        )

    @staticmethod
    def _has_any_refund(entry: dict) -> bool:
        """Stripe sets `refunded` only when a charge is refunded in full, so a
        partially refunded charge looks untouched. Both count as refunded here,
        otherwise the policy would hand back the remaining balance of a charge
        somebody has already been partly refunded for."""
        return bool(entry.get("refunded")) or entry.get("amount_refunded", 0) > 0

    def latest_charge(self, email: str) -> Charge | None:
        customers = self._client.get("/customers", params={"email": email, "limit": 1})
        customers.raise_for_status()
        found = customers.json()["data"]
        if not found:
            return None

        charges = self._client.get(
            "/charges", params={"customer": found[0]["id"], "limit": 100}
        )
        charges.raise_for_status()
        entries = charges.json()["data"]
        if not entries:
            return None

        prior_refund_count = sum(1 for item in entries if self._has_any_refund(item))
        untouched = [item for item in entries if not self._has_any_refund(item)]

        if untouched:
            # Newest charge that could still be refunded, plus how many other
            # candidates there are, so the policy can refuse to guess.
            target = untouched[0]
            siblings = len(untouched) - 1
        else:
            # Everything is already refunded. Return the newest anyway so the
            # policy can say so rather than reporting no charge at all.
            target = entries[0]
            siblings = 0

        return Charge(
            id=target["id"],
            amount_cents=target["amount"],
            created=datetime.fromtimestamp(target["created"], tz=timezone.utc),
            refunded=self._has_any_refund(target),
            prior_refund_count=prior_refund_count - (0 if untouched else 1),
            sibling_unrefunded_count=siblings,
        )

    def refund(self, charge_id: str, idempotency_key: str) -> str:
        response = self._client.post(
            "/refunds",
            data={"charge": charge_id},
            headers={"Idempotency-Key": idempotency_key},
        )
        response.raise_for_status()
        return response.json()["id"]


# --------------------------------------------------------------------------
# Groq
# --------------------------------------------------------------------------

CLASSIFY_SCHEMA = {
    "name": "intent",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["refund", "support"]},
            "confidence": {"type": "number"},
        },
        "required": ["intent", "confidence"],
        "additionalProperties": False,
    },
}

# The prompt states the output contract as well as sending the schema. Relying
# on the schema alone is a mistake: only some provider endpoints enforce it
# with constrained decoding, and the rest treat `strict` as a suggestion. An
# earlier version of this prompt said "reply with refund", which a
# non-enforcing provider followed to the letter and returned prose.
CLASSIFY_PROMPT = """You sort incoming customer support messages.

Classify the message as "refund" when the customer is asking for their money
back. Everything else, including complaints, questions about billing and
requests to cancel, is "support".

Respond with a single JSON object and nothing else, of exactly this shape:
{"intent": "refund" or "support", "confidence": number between 0 and 1}

confidence is how sure you are of the classification."""


class Brain:
    """The model, over OpenRouter.

    The base URL is configurable because every provider worth using speaks the
    same OpenAI-shaped API, so switching back to Groq or anywhere else is a
    change of two environment variables rather than a change of code.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    # Free, and fast enough to feel immediate: about a second to classify and
    # two to write an answer, served by Nvidia directly. The obvious pick,
    # openai/gpt-oss-20b:free, has a single provider that took seconds to
    # minutes and generated JSON by instruction rather than enforcing it.
    DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
    # Both jobs here want the most likely answer, not a creative one. Left at
    # the provider default, which is usually 1.0, a small model occasionally
    # drops a stray token into an otherwise fine sentence, and the classifier's
    # confidence wanders between runs on identical input.
    DEFAULT_TEMPERATURE = 0.0

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        provider_order: tuple[str, ...] = (),
        temperature: float = DEFAULT_TEMPERATURE,
        require_schema: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ):
        self._model = model
        self._temperature = temperature
        # Asking OpenRouter to route only to endpoints advertising structured
        # output sounds prudent, but it narrows the free catalogue from fourteen
        # models to four, and the endpoints it does allow may still not enforce
        # the schema. What actually guarantees a usable answer here is the
        # prompt stating the contract, plus validation and a retry. Off by
        # default so any model can be used; turn it on to insist.
        self._require_schema = require_schema
        self._sleep = sleep
        # `provider` is an OpenRouter extension. Sending it to a plain
        # OpenAI-shaped endpoint like Groq risks a rejected request, so it only
        # goes out when we are actually talking to OpenRouter.
        self._openrouter = "openrouter.ai" in base_url
        self._provider: dict[str, object] = {}
        if provider_order:
            # Pinning matters: a model can be served by several providers and
            # some advertise strict schema support without honouring it. Naming
            # the provider and refusing fallbacks is the only way to be sure.
            self._provider["order"] = list(provider_order)
            self._provider["allow_fallbacks"] = False
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
            transport=transport,
        )

    INTENTS = ("refund", "support")

    def classify(self, message: str) -> Classification:
        """Classify one message, retrying if the reply is not usable.

        Only some provider endpoints enforce the schema with constrained
        decoding. The rest generate JSON by following instructions, which is
        probabilistic: it is usually right and occasionally malformed. Since
        the model is stochastic, asking again is the appropriate response to an
        unusable answer rather than crashing the poll.
        """
        last_payload = ""
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            last_payload = self._chat(
                [
                    {"role": "system", "content": CLASSIFY_PROMPT},
                    {"role": "user", "content": message},
                ],
                response_format={"type": "json_schema", "json_schema": CLASSIFY_SCHEMA},
                **self._provider_preferences(require_parameters=self._require_schema),
            )
            classification = self._parse(last_payload)
            if classification is not None:
                return classification
            log.warning(
                "classification %s of %s was not usable: %r",
                attempt,
                self.MAX_ATTEMPTS,
                last_payload[:120],
            )

        raise RuntimeError(
            f"model did not return a usable classification after "
            f"{self.MAX_ATTEMPTS} attempts. Last reply: {last_payload[:300]}"
        )

    def _parse(self, payload: str) -> Classification | None:
        """A classification, or None if the reply cannot be trusted.

        An out-of-range confidence or an unknown intent is treated the same as
        malformed JSON. Letting either through would quietly corrupt the refund
        decision, which reads confidence as a number and compares it to a
        threshold.
        """
        try:
            parsed = json.loads(payload)
            intent = parsed["intent"]
            confidence = float(parsed["confidence"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

        if intent not in self.INTENTS or not 0.0 <= confidence <= 1.0:
            return None
        return Classification(intent=intent, confidence=confidence)

    def answer(
        self, message: str, knowledge: str, articles: Sequence[Article] = ()
    ) -> str:
        """Write a reply.

        Two separate inputs, deliberately. `knowledge` is how to behave: tone,
        when to ask a follow-up, and the standing rule never to invent. The
        articles are what may be stated as fact. Keeping them apart is what lets
        the documentation change without touching the instructions.
        """
        instructions = [
            "You are a customer support agent.",
            "",
            "How to behave:",
            knowledge,
        ]
        if articles:
            instructions += [
                "",
                "Documentation you may state as fact. Use only what is here for "
                "anything specific about the product, its prices or its "
                "policies. Do not add details that are not written below.",
                "",
            ]
            instructions += [
                f"## {article.title}\n{article.content}" for article in articles
            ]
        else:
            instructions += [
                "",
                "No documentation matched this question, so do not state "
                "anything specific about the product, its prices or its "
                "policies. Answer generally if you can, otherwise say a "
                "colleague will follow up.",
            ]

        return self._chat(
            [
                {"role": "system", "content": "\n".join(instructions)},
                {"role": "user", "content": message},
            ],
            **self._provider_preferences(),
        )

    def _provider_preferences(self, *, require_parameters: bool = False) -> dict:
        """OpenRouter's provider routing block, when it applies.

        `require_parameters` asks OpenRouter not to route to endpoints that do
        not advertise the parameters we sent. It is necessary but not
        sufficient: an endpoint can advertise strict schema support and still
        return prose, which is what `provider_order` is for.
        """
        if not self._openrouter:
            return {}
        preferences = dict(self._provider)
        if require_parameters:
            preferences["require_parameters"] = True
        return {"provider": preferences} if preferences else {}

    # A one-off 400 has been observed for a request that then succeeds
    # unchanged, so 400 is retried rather than treated as permanent. A
    # genuinely malformed request just fails three times quickly and reports
    # the response body, which raise_for_status() would have hidden.
    RETRY_ON = {400, 408, 409, 425, 429, 500, 502, 503, 504}
    MAX_ATTEMPTS = 3

    def _chat(self, messages: list[dict], **extra) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            **extra,
        }
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            response = self._client.post("/chat/completions", json=payload)
            if response.is_success:
                return response.json()["choices"][0]["message"]["content"]

            if response.status_code not in self.RETRY_ON or attempt == self.MAX_ATTEMPTS:
                raise RuntimeError(
                    f"model API returned {response.status_code} after {attempt} "
                    f"attempt(s): {response.text[:300]}"
                )
            delay = 2 ** (attempt - 1)
            log.warning("model API returned %s, retrying in %ss", response.status_code, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env_file()

    knowledge_path = Path(os.environ.get("KNOWLEDGE_FILE", "knowledge.md"))
    if not knowledge_path.exists():
        raise SystemExit(f"knowledge file not found: {knowledge_path}")

    stripe_key = require_env(
        "STRIPE_API_KEY",
        hint="A restricted key from a Stripe sandbox (rk_test_...).",
    )
    # This is a demo and stays in a test environment, so refuse anything that
    # could move real money rather than trusting the operator to notice.
    if not stripe_key.startswith(("rk_test_", "sk_test_")):
        raise SystemExit(
            f"STRIPE_API_KEY is not a test key (starts {stripe_key[:8]!r}). "
            "This agent only runs against Stripe test mode."
        )

    chatwoot_url = require_env("CHATWOOT_URL", hint="e.g. http://localhost:3000")
    inbox = ChatwootInbox(
        base_url=chatwoot_url,
        account_id=require_env("CHATWOOT_ACCOUNT_ID", hint="e.g. 1"),
        token=require_env(
            "CHATWOOT_TOKEN", hint="Chatwoot Profile Settings -> Access Token."
        ),
    )
    payments = StripePayments(stripe_key)
    # MODEL_* is the honest name, since any OpenAI-shaped endpoint works here.
    # The OPENROUTER_* names still work so existing .env files keep running.
    brain = Brain(
        require_env(
            "MODEL_API_KEY",
            "OPENROUTER_API_KEY",
            hint="Free from openrouter.ai, no card needed.",
        ),
        model=env_value("MODEL_NAME", "OPENROUTER_MODEL", default=Brain.DEFAULT_MODEL),
        base_url=env_value(
            "MODEL_BASE_URL", "OPENROUTER_BASE_URL", default=Brain.DEFAULT_BASE_URL
        ),
        provider_order=tuple(
            name.strip()
            for name in env_value("MODEL_PROVIDER", "OPENROUTER_PROVIDER").split(",")
            if name.strip()
        ),
        temperature=float(
            os.environ.get("MODEL_TEMPERATURE", Brain.DEFAULT_TEMPERATURE)
        ),
        require_schema=os.environ.get("MODEL_REQUIRE_SCHEMA", "").lower()
        in ("1", "true", "yes"),
    )

    portal = env_value("HELP_CENTRE_PORTAL")
    help_centre = (
        ChatwootHelpCentre(
            base_url=chatwoot_url,
            account_id=env_value("CHATWOOT_ACCOUNT_ID", default="1"),
            token=env_value("CHATWOOT_TOKEN"),
            portal_slug=portal,
        )
        if portal
        else None
    )
    if help_centre is None:
        log.info("no HELP_CENTRE_PORTAL set, answering from %s alone", knowledge_path)

    interval = int(os.environ.get("POLL_INTERVAL_SECONDS", "5"))
    log.info("polling %s every %ss, knowledge from %s", chatwoot_url, interval, knowledge_path)
    run_forever(
        knowledge_path=knowledge_path,
        interval_seconds=interval,
        inbox=inbox,
        payments=payments,
        classify=brain.classify,
        answer=brain.answer,
        help_centre=help_centre,
    )


if __name__ == "__main__":
    main()
