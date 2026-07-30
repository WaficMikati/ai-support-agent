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
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

import httpx

log = logging.getLogger("support-agent")

# Where the last handled message id is kept, on the Chatwoot conversation.
HANDLED_ATTRIBUTE = "bot_last_message_id"


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
# Two of the three rubric points. Worth spelling out, because the arithmetic
# decides real behaviour: a plain "I want my money back" scores clear request and
# no hedging but does not name a charge, so it lands on 0.67 and is approved,
# while "maybe I could get a refund?" loses the hedging point as well and is held
# for a human. Requiring all three would hold almost every genuine request,
# because customers hardly ever name the payment.
MIN_CONFIDENCE = 0.6


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
class Turn:
    """One thing somebody said, as the model sees it."""

    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Conversation:
    id: int
    contact_email: str | None
    messages: tuple[Message, ...]
    # The id of the last message this agent acted on, as recorded in Chatwoot's
    # own custom_attributes. Remembering it there rather than in this process is
    # what makes the memory survive a restart, and what stops a second copy of
    # the agent answering the same message.
    handled_message_id: int | None = None
    # Whether `messages` is the whole thread or only the newest entry the poll
    # happened to carry. Deciding whether to reply needs one message; writing
    # the reply needs all of them, and the two are fetched at different moments.
    history_complete: bool = False

    def turns(self, limit: int = 10) -> tuple[Turn, ...]:
        """The conversation as the model should read it, oldest last-limit first.

        Private notes are left out: they are written for colleagues, not for the
        customer, and showing them to the model invites it to repeat internal
        reasoning back to whoever is asking. Chatwoot's own activity and
        template entries are left out because nobody said them.
        """
        spoken = [
            Turn(role="user" if message.incoming else "assistant", content=message.content)
            for message in self.messages
            if not message.activity and not message.template and not message.private
            and message.content.strip()
        ]
        return tuple(spoken[-limit:])

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
class Proposal:
    """What the model thinks should happen next.

    A proposal rather than a decision. The reply is used as written unless money
    moves, and whether money moves is settled afterwards by `refund_decision`,
    in code, against thresholds the model never sees.

    Confidence is not asked for, it is worked out here from three plain questions
    about the message. A model asked "how confident are you" is introspecting,
    which it is bad at, and it returned a number that looked authoritative while
    meaning little. Asked whether the request was clear, or whether the customer
    hedged, it is reporting something observable. It also makes the note to the
    human specific: "they hedged" rather than "confidence 0.67".
    """

    reply: str
    refund_requested: bool
    clear_request: bool  # they plainly asked, rather than mused about it
    charge_identified: bool  # they said which payment they mean
    hedging: bool  # "maybe", "I think", "would I be able to"

    @property
    def confidence(self) -> float:
        return (
            int(self.clear_request) + int(self.charge_identified) + int(not self.hedging)
        ) / 3

    @property
    def signals(self) -> str:
        """The rubric in words, for the note left to a colleague."""
        return (
            f"clear request: {'yes' if self.clear_request else 'no'}, "
            f"charge identified: {'yes' if self.charge_identified else 'no'}, "
            f"hedging: {'yes' if self.hedging else 'no'}"
        )


@dataclass(frozen=True)
class Decision:
    auto_approve: bool
    reason: str


# --------------------------------------------------------------------------
# Ports. Real implementations are further down; tests substitute their own.
# --------------------------------------------------------------------------


class Inbox(Protocol):
    def open_conversations(self) -> list[Conversation]: ...
    def with_history(self, conversation: Conversation) -> Conversation: ...
    def send_reply(self, conversation_id: int, content: str) -> None: ...
    def add_private_note(self, conversation_id: int, content: str) -> None: ...
    def resolve(self, conversation_id: int) -> None: ...
    def record_handled(self, conversation_id: int, message_id: int) -> None: ...


class Payments(Protocol):
    def latest_charge(self, email: str) -> Charge | None: ...
    def refund(self, charge_id: str, idempotency_key: str) -> str: ...


class Understander(Protocol):
    def __call__(
        self,
        turns: Sequence[Turn],
        knowledge: str,
        articles: Sequence[Article],
    ) -> Proposal: ...


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
            f"rubric score {confidence:.2f} below {MIN_CONFIDENCE}",
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


def refund_idempotency_key(conversation_id: int, charge_id: str) -> str:
    """A stable key for refunding this charge in this conversation.

    Keyed on the charge rather than on the message that asked. Those differ in a
    case that matters: if somebody asks twice in one conversation, two messages
    mean two keys, and two keys mean Stripe treats the second attempt as a new
    refund. Keyed on the charge, the second attempt returns the first refund,
    which is the point of having a key at all. The policy would usually catch it
    first, since the charge now reads as refunded, but that is a check racing a
    write rather than a guarantee.

    Readable on purpose. A hash would work equally well for Stripe and tell you
    nothing when you are looking at it in their dashboard at four in the morning.
    No personal data, and well inside Stripe's 255 character limit.
    """
    return f"refund-conv{conversation_id}-{charge_id}"


HOLDING_REPLY = (
    "Thanks for getting in touch. I've passed this to a colleague who will look "
    "at your refund and come back to you shortly."
)

# Phrases that would tell a customer their money is on the way. Deliberately
# blunt: this only has to catch a model that ignored the instruction not to
# promise an outcome, and a false positive costs nothing but a plainer reply.
PROMISES_A_REFUND = (
    "refunded",
    "refund has been",
    "on its way back",
    "back to your",
    "processed your refund",
    "issued your refund",
    "you will receive",
    "you'll receive",
)


def safe_holding_reply(reply: str) -> str:
    """The model's reply, unless it promised a refund nobody has approved."""
    lowered = reply.lower()
    if any(phrase in lowered for phrase in PROMISES_A_REFUND):
        log.warning("proposal promised a refund before approval, using safe wording")
        return HOLDING_REPLY
    return reply


def retrieval_query(turns: Sequence[Turn]) -> str:
    """What to search the help centre for.

    The last two things the customer said rather than only the newest, because
    a follow-up is often too short to search on: "and the decaf one?" means
    nothing by itself but is answerable next to the question before it.
    """
    said = [turn.content for turn in turns if turn.role == "user"]
    return " ".join(said[-2:])


def needs_reply(conversation: Conversation) -> bool:
    """True when the customer spoke last.

    Polling means we see the same conversation repeatedly. Rather than track
    what we have already handled, we look at who spoke last: if the newest
    message is ours, there is nothing to do.
    """
    latest = conversation.latest
    return latest is not None and latest.incoming


# --------------------------------------------------------------------------
# The graph: what carries state, the nodes, the edges, and the walk
# --------------------------------------------------------------------------


@dataclass
class State:
    """What one conversation carries as it moves through the graph.

    Mutable and passed from node to node, which is the whole trick: each node is
    `state -> state`, so the routing can live in data rather than in nested ifs.
    Everything the nodes talk to is in here too, so a node never reaches for a
    global and a test can hand it fakes.
    """

    conversation: Conversation
    message: Message
    knowledge: str
    inbox: Inbox
    payments: Payments
    understand: Understander
    help_centre: HelpCentre | None = None
    now: datetime | None = None

    articles: tuple[Article, ...] = ()
    proposal: Proposal | None = None
    charge: Charge | None = None
    decision: Decision | None = None
    action: str = ""


# --------------------------------------------------------------------------
# The graph: nodes do the work, edges choose what happens next.
#
# There is no framework here and there does not need to be one. A graph is a
# dict of functions and a dict of routing rules, and writing it out is worth
# more than importing it, because the one decision that matters is visible as an
# edge: `understand` proposes, and the edge out of `refund` is chosen by
# refund_decision in code. The model can ask for money to move. It cannot make
# it move.
# --------------------------------------------------------------------------


def understand_node(state: State) -> State:
    """Read the conversation, fetch anything relevant, propose a reply."""
    turns = state.conversation.turns()
    articles = (
        state.help_centre.relevant(retrieval_query(turns)) if state.help_centre else []
    )
    state.articles = tuple(articles)
    if articles:
        log.info(
            "conversation %s retrieved %s",
            state.conversation.id,
            ", ".join(article.title for article in articles),
        )

    state.proposal = state.understand(turns, state.knowledge, state.articles)
    log.info(
        "conversation %s: refund_requested=%s (%.2f) over %s turns",
        state.conversation.id,
        state.proposal.refund_requested,
        state.proposal.confidence,
        len(turns),
    )
    return state


def answer_node(state: State) -> State:
    """Send the reply the model wrote, and close the turn."""
    assert state.proposal is not None
    state.inbox.send_reply(state.conversation.id, state.proposal.reply)
    # Resolved rather than left open: Chatwoot reopens a conversation as soon as
    # the customer writes again, so nothing is lost and the inbox does not fill
    # up with conversations we have already dealt with.
    state.inbox.resolve(state.conversation.id)
    state.action = "answered"
    return state


def refund_node(state: State) -> State:
    """Find the charge and apply the policy. No model involved.

    Deliberately decides nothing about the conversation: it works out the facts
    and lets the edge do the routing, so the approval is a line you can point at.
    """
    assert state.proposal is not None
    state.charge = (
        state.payments.latest_charge(state.conversation.contact_email)
        if state.conversation.contact_email
        else None
    )
    state.decision = refund_decision(
        state.charge, state.proposal.confidence, now=state.now
    )
    return state


def execute_refund_node(state: State) -> State:
    """Move the money, then say so with the real amount."""
    assert state.charge is not None  # guaranteed by refund_decision
    assert state.proposal is not None
    # If this same request is ever sent twice, whether by an overlapping pass or
    # by a restart after a crash between the refund and the reply, Stripe returns
    # the original refund rather than issuing another one.
    refund_id = state.payments.refund(
        state.charge.id,
        refund_idempotency_key(state.conversation.id, state.charge.id),
    )
    log.info("refunded charge %s (%s)", state.charge.id, refund_id)
    # Written here rather than by the model, because it states an amount.
    state.inbox.send_reply(
        state.conversation.id,
        f"That's refunded, {state.charge.amount_cents / 100:.2f} is on its way "
        "back to your original payment method. It usually lands within a few days.",
    )
    state.inbox.resolve(state.conversation.id)
    state.action = "refunded"
    return state


def hold_node(state: State) -> State:
    """Hand it to a person, and tell the customer that is what happened."""
    assert state.proposal is not None and state.decision is not None
    # Tell the customer something, then tell the team why. Posting only the note
    # leaves the customer staring at silence, which is indistinguishable from a
    # broken agent: they asked for a refund and nothing came back.
    #
    # The model's own reply is used, so a message that asked two things gets both
    # addressed. It was told not to promise an outcome, and if it did anyway the
    # safe wording replaces it: the customer must not be told their money is
    # coming back when a human has not agreed to it.
    state.inbox.send_reply(
        state.conversation.id, safe_holding_reply(state.proposal.reply)
    )
    # Deliberately left open, and not resolved: a person still has to act.
    state.inbox.add_private_note(
        state.conversation.id,
        "Refund request held for review.\n"
        f"Reason: {state.decision.reason}\n"
        f"Signals: {state.proposal.signals}\n\n"
        "The customer has been told a colleague will follow up.",
    )
    state.action = "flagged"
    return state


NODES: dict[str, Callable[[State], State]] = {
    "understand": understand_node,
    "answer": answer_node,
    "refund": refund_node,
    "execute_refund": execute_refund_node,
    "hold": hold_node,
}

EDGES: dict[str, Callable[[State], str | None]] = {
    # What the model asked for.
    "understand": lambda s: "refund" if s.proposal.refund_requested else "answer",
    # What the policy allows. This edge is the money gate, and nothing the model
    # returned is consulted here beyond the rubric score the policy scores itself.
    "refund": lambda s: "execute_refund" if s.decision.auto_approve else "hold",
    "answer": lambda s: None,
    "execute_refund": lambda s: None,
    "hold": lambda s: None,
}


def run_graph(state: State, start: str = "understand") -> State:
    """Walk the graph until an edge says to stop."""
    node: str | None = start
    while node:
        state = NODES[node](state)
        node = EDGES[node](state)
    return state


# --------------------------------------------------------------------------
# Handling one conversation
# --------------------------------------------------------------------------


def handle_conversation(
    conversation: Conversation,
    *,
    inbox: Inbox,
    payments: Payments,
    understand: Understander,
    knowledge: str,
    help_centre: HelpCentre | None = None,
    handled: HandledMessages | None = None,
    now: datetime | None = None,
) -> str:
    """Decide whether this conversation is ours to touch, then run the graph."""
    latest = conversation.latest
    if latest is None or not latest.incoming:
        return "skipped"

    # Two layers, cheapest first. The in-process set saves a round trip within a
    # run; Chatwoot's marker is what survives a restart and is shared with any
    # other copy of the agent.
    if handled is not None and latest.id in handled:
        return "already handled"
    if conversation.handled_message_id == latest.id:
        return "already handled"

    state = run_graph(
        State(
            conversation=conversation,
            message=latest,
            knowledge=knowledge,
            inbox=inbox,
            payments=payments,
            understand=understand,
            help_centre=help_centre,
            now=now,
        )
    )

    # Recorded only once the work succeeded. Marking it earlier would mean a
    # transient model or network failure silently swallowed the message.
    if handled is not None:
        handled.add(latest.id)
    inbox.record_handled(conversation.id, latest.id)
    return state.action


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_once(
    *,
    inbox: Inbox,
    payments: Payments,
    understand: Understander,
    knowledge: str,
    help_centre: HelpCentre | None = None,
    handled: HandledMessages | None = None,
) -> list[str]:
    actions = []
    for conversation in inbox.open_conversations():
        if not needs_reply(conversation):
            continue
        # Only now is the rest of the thread worth fetching. Deciding to reply
        # takes one message, writing the reply takes all of them, and this is
        # the line between the two.
        conversation = inbox.with_history(conversation)
        actions.append(
            handle_conversation(
                conversation,
                inbox=inbox,
                payments=payments,
                understand=understand,
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
        return [self._conversation(entry) for entry in payload]

    @staticmethod
    def _message(item: dict) -> Message:
        # message_type: 0 incoming, 1 outgoing, 2 activity, 3 template.
        # Private notes are kept. They are how we mark that a refund has
        # already been handed to a human, so dropping them here would make
        # the agent flag the same conversation on every single poll.
        return Message(
            id=item["id"],
            content=item.get("content") or "",
            incoming=item.get("message_type") == 0,
            private=bool(item.get("private")),
            activity=item.get("message_type") == 2,
            template=item.get("message_type") == 3,
        )

    def _conversation(self, entry: dict) -> Conversation:
        """One conversation, fetching its messages only when it has to.

        The list response already carries the newest message, and that is all
        *this* step needs: whether it is our turn. Fetching the full thread per
        conversation made the poll cost one request per open conversation, so
        the agent got measurably slower as the queue of refunds awaiting a human
        grew, and those conversations stay open by design.

        The fallback matters though. Chatwoot files its own entries as messages,
        and if the newest one is an activity or a template we cannot tell who
        spoke last without looking further back.

        What comes back is therefore usually a conversation with one message in
        it, which is why `history_complete` is on it. Anything that reads the
        thread must call `with_history` first: replying from the newest message
        alone is exactly the bug that made the agent greet a customer halfway
        through a conversation.
        """
        conversation_id = entry["id"]
        contact = (entry.get("meta") or {}).get("sender") or {}
        attributes = entry.get("custom_attributes") or {}
        handled = attributes.get(HANDLED_ATTRIBUTE)
        newest = [
            self._message(item)
            for item in (entry.get("messages") or [])
            if isinstance(item, dict) and "id" in item
        ]

        usable = bool(newest) and not (newest[-1].activity or newest[-1].template)
        messages = tuple(newest) if usable else self._all_messages(conversation_id)

        return Conversation(
            id=conversation_id,
            contact_email=contact.get("email"),
            messages=messages,
            handled_message_id=int(handled) if str(handled).isdigit() else None,
            # The fallback fetched everything, so in that case it is already whole.
            history_complete=not usable,
        )

    def with_history(self, conversation: Conversation) -> Conversation:
        """The same conversation with the whole thread attached.

        Called for the conversations about to be answered and no others, which
        is what keeps a poll over a long queue at one request. Deciding whether
        to reply needs the newest message; writing the reply needs everything
        said so far, and only a small fraction of open conversations are being
        replied to on any given pass.
        """
        if conversation.history_complete:
            return conversation
        return replace(
            conversation,
            messages=self._all_messages(conversation.id),
            history_complete=True,
        )

    def _all_messages(self, conversation_id: int) -> tuple[Message, ...]:
        response = self._client.get(f"/conversations/{conversation_id}/messages")
        response.raise_for_status()
        return tuple(self._message(item) for item in response.json()["payload"])

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

    def record_handled(self, conversation_id: int, message_id: int) -> None:
        """Remember, in Chatwoot, which message this agent last acted on.

        Chatwoot's own database becomes the memory, so it survives a restart and
        is shared by anything polling the same account. An in-process set cannot
        manage either: a crash forgets it, and a second copy of the agent has its
        own, which is exactly how the same message got answered twice.
        """
        response = self._client.post(
            f"/conversations/{conversation_id}/custom_attributes",
            json={"custom_attributes": {HANDLED_ATTRIBUTE: message_id}},
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
# The model. Any OpenAI-shaped endpoint; the name is set in .env.
# --------------------------------------------------------------------------

PROPOSAL_SCHEMA = {
    "name": "proposal",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "refund_requested": {"type": "boolean"},
            "clear_request": {"type": "boolean"},
            "charge_identified": {"type": "boolean"},
            "hedging": {"type": "boolean"},
        },
        "required": [
            "reply",
            "refund_requested",
            "clear_request",
            "charge_identified",
            "hedging",
        ],
        "additionalProperties": False,
    },
}

# Appended after the behaviour guidance and the documentation, so the output
# contract is the last thing read. Only one judgement is asked for beyond the
# reply, and it is deliberately narrow: is this person asking us to send their
# money back. Everything else the agent needs to do is just answering well.
UNDERSTAND_RULES = """Write the next reply in this conversation.

Separately, decide one thing: is the customer asking us to refund a payment they
have already made? That is a request about their own money. Questions about how
refunds work, whether they are offered, or what would happen in some situation
are not requests, and should be answered from the documentation.

If it is a request, say you are looking into it. Do not say a refund has been
made and do not promise one: whether it can be approved is decided after you
reply, and you are not the one deciding.

Reply as yourself, in the conversation, using what was said earlier. Do not
repeat a greeting you have already given.

Then answer three plain questions about what they wrote. Answer them about the
message in front of you, not about how sure you feel:

  clear_request     did they plainly ask for a refund, rather than wondering
                    aloud about one?
  charge_identified did they say which payment they mean, by date, amount or
                    description?
  hedging           did they hedge, with "maybe", "I think", "would I be able
                    to", or similar?

Respond with a single JSON object and nothing else, of exactly this shape:
{"reply": "...", "refund_requested": true, "clear_request": true,
 "charge_identified": false, "hedging": false}"""


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

    def understand(
        self,
        turns: Sequence[Turn],
        knowledge: str,
        articles: Sequence[Article] = (),
    ) -> Proposal:
        """Read the conversation and propose the next reply.

        One call, given the thread rather than a single message, returning both
        the reply and whether a refund is being requested. It is a proposal: the
        thresholds that decide whether money actually moves are applied
        afterwards, in code, and are never shown to the model.

        This replaced a classify-then-branch design, where each message was
        reduced to one of two labels before anything was understood. That shape
        could not hold a conversation, because a label carries no context, and it
        pushed every distinction into the rubric: "how do I get a refund?" was
        read as a request and refunded twenty dollars until an example was added
        forbidding it. The list of such examples has no end.
        """
        system = [
            "You are a customer support agent for a subscription business.",
            "",
            "How to behave:",
            knowledge,
        ]
        if articles:
            system += [
                "",
                "Documentation you may state as fact. Anything specific about the "
                "product, its prices or its policies must come from here. Do not "
                "add details that are not written below.",
                "",
            ]
            system += [f"## {a.title}\n{a.content}" for a in articles]
        else:
            system += [
                "",
                "No documentation matched this conversation, so state nothing "
                "specific about the product, its prices or its policies. Answer "
                "generally if the question does not need them, otherwise say a "
                "colleague will follow up.",
            ]
        system += ["", UNDERSTAND_RULES]

        messages = [{"role": "system", "content": "\n".join(system)}]
        messages += [{"role": t.role, "content": t.content} for t in turns]

        last_payload = ""
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            last_payload = self._chat(
                messages,
                response_format={"type": "json_schema", "json_schema": PROPOSAL_SCHEMA},
                **self._provider_preferences(require_parameters=self._require_schema),
            )
            proposal = self._parse(last_payload)
            if proposal is not None:
                return proposal
            log.warning(
                "proposal %s of %s was not usable: %r",
                attempt,
                self.MAX_ATTEMPTS,
                last_payload[:120],
            )

        raise RuntimeError(
            f"model did not return a usable proposal after {self.MAX_ATTEMPTS} "
            f"attempts. Last reply: {last_payload[:300]}"
        )

    @staticmethod
    def _parse(payload: str) -> Proposal | None:
        """A proposal, or None if the reply cannot be trusted.

        An empty reply or a flag that is not a real boolean is treated like
        malformed JSON. Letting either through would corrupt the refund decision,
        which scores those flags and compares the result to a threshold.
        """
        try:
            parsed = json.loads(payload)
            reply = parsed["reply"]
            flags = {
                name: parsed[name]
                for name in ("refund_requested", "clear_request", "charge_identified", "hedging")
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        if not isinstance(reply, str) or not reply.strip():
            return None
        # Every flag must be a real boolean. "yes" is truthy in Python, and
        # letting a string through here would move money on a malformed reply.
        if not all(isinstance(value, bool) for value in flags.values()):
            return None
        return Proposal(reply=reply.strip(), **flags)

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
    # Five rather than three. Groq's constrained decoder intermittently answers
    # json_validate_failed with an empty generation, and a request that failed
    # three times in a row succeeded unchanged straight afterwards. Each failure
    # is fast, so the extra attempts cost little and buy a demo that does not
    # die on a bad patch.
    MAX_ATTEMPTS = 5

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
            delay = self._retry_after(response) or 2 ** (attempt - 1)
            log.warning("model API returned %s, retrying in %ss", response.status_code, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        """How long the provider asked us to wait, if it said.

        Worth honouring rather than guessing. Groq's free tier limits tokens per
        minute, not just requests, and this agent sends a whole conversation plus
        documentation in one call, so that ceiling arrives sooner than a request
        count suggests. Backing off for one second when it asked for five just
        burns the remaining attempts.
        """
        header = response.headers.get("retry-after", "")
        try:
            return min(float(header), 30.0)
        except ValueError:
            pass
        # Groq puts the figure in the message rather than a header.
        match = re.search(r"try again in ([0-9.]+)s", response.text)
        if match:
            return min(float(match.group(1)) + 0.5, 30.0)
        return None


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
        understand=brain.understand,
        help_centre=help_centre,
    )


if __name__ == "__main__":
    main()
