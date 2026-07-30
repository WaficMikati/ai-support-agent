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
from typing import Protocol

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


def require_env(name: str, hint: str = "") -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        message = f"{name} is not set. Add it to .env or export it."
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
    activity: bool = False


@dataclass(frozen=True)
class Conversation:
    id: int
    contact_email: str | None
    messages: tuple[Message, ...]

    @property
    def latest(self) -> Message | None:
        """The newest message that represents somebody speaking.

        Chatwoot also files status changes, assignments and labels as
        messages. Those are nobody's turn, so they are skipped: otherwise a
        label being added after a customer writes in would look like a reply
        and the conversation would never be answered.
        """
        spoken = [message for message in self.messages if not message.activity]
        return spoken[-1] if spoken else None


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
    def refund(self, charge_id: str) -> str: ...


class Classifier(Protocol):
    def __call__(self, message: str) -> Classification: ...


class Answerer(Protocol):
    def __call__(self, message: str, knowledge: str) -> str: ...


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
    now: datetime | None = None,
) -> str:
    """Process a single conversation. Returns the action taken, for logging."""
    latest = conversation.latest
    if latest is None or not latest.incoming:
        return "skipped"

    result = classify(latest.content)
    log.info(
        "conversation %s classified as %s (%.2f)",
        conversation.id,
        result.intent,
        result.confidence,
    )

    if result.intent == "refund":
        return _handle_refund(
            conversation,
            result,
            inbox=inbox,
            payments=payments,
            now=now,
        )

    reply = answer(latest.content, knowledge)
    inbox.send_reply(conversation.id, reply)
    # Resolved rather than left open: Chatwoot reopens a conversation as soon
    # as the customer writes again, so nothing is lost and the inbox does not
    # fill up with conversations we have already dealt with.
    inbox.resolve(conversation.id)
    return "answered"


def _handle_refund(
    conversation: Conversation,
    result: Classification,
    *,
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
        # Deliberately left open: a person still has to act on it.
        inbox.add_private_note(
            conversation.id,
            "Refund request held for review.\n"
            f"Reason: {decision.reason}\n\n"
            "Suggested reply: Thanks for getting in touch, we're looking at "
            "your refund now and will come back to you shortly.",
        )
        return "flagged"

    assert charge is not None  # guaranteed by refund_decision
    refund_id = payments.refund(charge.id)
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
    while True:
        try:
            run_once(knowledge=knowledge_path.read_text(), **kwargs)
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

    def refund(self, charge_id: str) -> str:
        response = self._client.post("/refunds", data={"charge": charge_id})
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

CLASSIFY_PROMPT = """You sort incoming customer support messages.

Reply with "refund" only when the customer is asking for their money back.
Everything else, including complaints, questions about billing and requests
to cancel, is "support".

confidence is how sure you are, from 0 to 1."""


class GroqBrain:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ):
        self._model = model
        self._sleep = sleep
        self._client = httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
            transport=transport,
        )

    def classify(self, message: str) -> Classification:
        payload = self._chat(
            [
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": message},
            ],
            response_format={"type": "json_schema", "json_schema": CLASSIFY_SCHEMA},
        )
        parsed = json.loads(payload)
        return Classification(
            intent=parsed["intent"],
            confidence=float(parsed["confidence"]),
        )

    def answer(self, message: str, knowledge: str) -> str:
        return self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a customer support agent. Answer using only the "
                        "guidance below. If it does not cover the question, say "
                        "you are passing it to a colleague.\n\n" + knowledge
                    ),
                },
                {"role": "user", "content": message},
            ]
        )

    # Groq has been observed returning a one-off 400 for a request that then
    # succeeds unchanged, so 400 is retried rather than treated as permanent.
    # A genuinely malformed request just fails three times quickly and reports
    # the response body, which raise_for_status() would have hidden.
    RETRY_ON = {400, 408, 409, 425, 429, 500, 502, 503, 504}
    MAX_ATTEMPTS = 3

    def _chat(self, messages: list[dict], **extra) -> str:
        payload = {"model": self._model, "messages": messages, **extra}
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            response = self._client.post("/chat/completions", json=payload)
            if response.is_success:
                return response.json()["choices"][0]["message"]["content"]

            if response.status_code not in self.RETRY_ON or attempt == self.MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Groq returned {response.status_code} after {attempt} "
                    f"attempt(s): {response.text[:300]}"
                )
            delay = 2 ** (attempt - 1)
            log.warning("groq returned %s, retrying in %ss", response.status_code, delay)
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
        "STRIPE_API_KEY", "A restricted key from a Stripe sandbox (rk_test_...)."
    )
    # This is a demo and stays in a test environment, so refuse anything that
    # could move real money rather than trusting the operator to notice.
    if not stripe_key.startswith(("rk_test_", "sk_test_")):
        raise SystemExit(
            f"STRIPE_API_KEY is not a test key (starts {stripe_key[:8]!r}). "
            "This agent only runs against Stripe test mode."
        )

    chatwoot_url = require_env("CHATWOOT_URL", "e.g. http://localhost:3000")
    inbox = ChatwootInbox(
        base_url=chatwoot_url,
        account_id=require_env("CHATWOOT_ACCOUNT_ID", "e.g. 1"),
        token=require_env("CHATWOOT_TOKEN", "Chatwoot Profile Settings -> Access Token."),
    )
    payments = StripePayments(stripe_key)
    brain = GroqBrain(
        require_env("GROQ_API_KEY", "Free from console.groq.com, no card needed."),
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
    )

    interval = int(os.environ.get("POLL_INTERVAL_SECONDS", "5"))
    log.info("polling %s every %ss, knowledge from %s", chatwoot_url, interval, knowledge_path)
    run_forever(
        knowledge_path=knowledge_path,
        interval_seconds=interval,
        inbox=inbox,
        payments=payments,
        classify=brain.classify,
        answer=brain.answer,
    )


if __name__ == "__main__":
    main()
