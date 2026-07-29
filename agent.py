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


@dataclass(frozen=True)
class Conversation:
    id: int
    contact_email: str | None
    messages: tuple[Message, ...]

    @property
    def latest(self) -> Message | None:
        return self.messages[-1] if self.messages else None


@dataclass(frozen=True)
class Charge:
    id: str
    amount_cents: int
    created: datetime
    refunded: bool
    prior_refund_count: int


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
        return Decision(False, f"charge {charge.id} is already refunded")

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


def run_forever(interval_seconds: int = 5, **kwargs) -> None:
    while True:
        try:
            run_once(**kwargs)
        except Exception:  # a bad poll should not kill the agent
            log.exception("poll failed")
        time.sleep(interval_seconds)


# --------------------------------------------------------------------------
# Chatwoot
# --------------------------------------------------------------------------


class ChatwootInbox:
    def __init__(self, base_url: str, account_id: str, token: str):
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api/v1/accounts/{account_id}",
            headers={"api_access_token": token},
            timeout=20,
        )

    def open_conversations(self) -> list[Conversation]:
        response = self._client.get("/conversations", params={"status": "open"})
        response.raise_for_status()
        payload = response.json()["data"]["payload"]
        return [self._conversation(entry["id"], entry) for entry in payload]

    def _conversation(self, conversation_id: int, entry: dict) -> Conversation:
        response = self._client.get(f"/conversations/{conversation_id}/messages")
        response.raise_for_status()
        messages = tuple(
            Message(
                id=item["id"],
                content=item.get("content") or "",
                incoming=item.get("message_type") == 0,  # 0 incoming, 1 outgoing
            )
            for item in response.json()["payload"]
            if not item.get("private")
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

    def __init__(self, secret_key: str):
        self._client = httpx.Client(
            base_url="https://api.stripe.com/v1",
            headers={"Authorization": f"Bearer {secret_key}"},
            timeout=20,
        )

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

        newest = entries[0]
        return Charge(
            id=newest["id"],
            amount_cents=newest["amount"],
            created=datetime.fromtimestamp(newest["created"], tz=timezone.utc),
            refunded=newest["refunded"],
            prior_refund_count=sum(1 for item in entries[1:] if item["refunded"]),
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
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        self._model = model
        self._client = httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
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

    def _chat(self, messages: list[dict], **extra) -> str:
        response = self._client.post(
            "/chat/completions",
            json={"model": self._model, "messages": messages, **extra},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    knowledge = Path(
        os.environ.get("KNOWLEDGE_FILE", "knowledge.md")
    ).read_text()

    inbox = ChatwootInbox(
        base_url=os.environ["CHATWOOT_URL"],
        account_id=os.environ["CHATWOOT_ACCOUNT_ID"],
        token=os.environ["CHATWOOT_TOKEN"],
    )
    payments = StripePayments(os.environ["STRIPE_API_KEY"])
    brain = GroqBrain(os.environ["GROQ_API_KEY"])

    log.info("polling %s", os.environ["CHATWOOT_URL"])
    run_forever(
        inbox=inbox,
        payments=payments,
        classify=brain.classify,
        answer=brain.answer,
        knowledge=knowledge,
    )


if __name__ == "__main__":
    main()
