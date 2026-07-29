"""Tests for the support agent.

Nothing here touches the network. Chatwoot, Stripe and the model are all
replaced with fakes, so the refund policy and the routing can be checked
without any credentials.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agent import (
    MAX_AUTO_REFUND_CENTS,
    MAX_CHARGE_AGE_DAYS,
    Charge,
    Classification,
    Conversation,
    Message,
    handle_conversation,
    needs_reply,
    refund_decision,
    run_once,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- fakes


class FakeInbox:
    def __init__(self, conversations=()):
        self.conversations = list(conversations)
        self.replies: list[tuple[int, str]] = []
        self.notes: list[tuple[int, str]] = []

    def open_conversations(self):
        return self.conversations

    def send_reply(self, conversation_id, content):
        self.replies.append((conversation_id, content))

    def add_private_note(self, conversation_id, content):
        self.notes.append((conversation_id, content))


class FakePayments:
    def __init__(self, charge=None):
        self.charge = charge
        self.refunded: list[str] = []

    def latest_charge(self, email):
        return self.charge

    def refund(self, charge_id):
        self.refunded.append(charge_id)
        return "re_fake_1"


def charge(**overrides) -> Charge:
    defaults = dict(
        id="ch_1",
        amount_cents=2_000,
        created=NOW - timedelta(days=3),
        refunded=False,
        prior_refund_count=0,
    )
    return Charge(**{**defaults, **overrides})


def conversation(
    text="hello", incoming=True, email: str | None = "a@b.com"
) -> Conversation:
    return Conversation(
        id=1,
        contact_email=email,
        messages=(Message(id=1, content=text, incoming=incoming),),
    )


def classifier(intent, confidence=0.95):
    return lambda message: Classification(intent=intent, confidence=confidence)


def answerer(text="here is your answer"):
    return lambda message, knowledge: text


def run(conv, *, inbox=None, payments=None, intent="support", confidence=0.95):
    inbox = inbox or FakeInbox()
    payments = payments or FakePayments()
    action = handle_conversation(
        conv,
        inbox=inbox,
        payments=payments,
        classify=classifier(intent, confidence),
        answer=answerer(),
        knowledge="guidance",
        now=NOW,
    )
    return action, inbox, payments


# ---------------------------------------------------------------- policy


def test_ordinary_charge_is_auto_approved():
    assert refund_decision(charge(), 0.95, now=NOW).auto_approve


def test_amount_exactly_at_the_limit_is_allowed():
    at_limit = charge(amount_cents=MAX_AUTO_REFUND_CENTS)
    assert refund_decision(at_limit, 0.95, now=NOW).auto_approve


def test_amount_one_cent_over_the_limit_is_flagged():
    over = charge(amount_cents=MAX_AUTO_REFUND_CENTS + 1)
    decision = refund_decision(over, 0.95, now=NOW)
    assert not decision.auto_approve
    assert "limit" in decision.reason


def test_charge_exactly_at_the_age_limit_is_allowed():
    old = charge(created=NOW - timedelta(days=MAX_CHARGE_AGE_DAYS))
    assert refund_decision(old, 0.95, now=NOW).auto_approve


def test_charge_past_the_age_limit_is_flagged():
    stale = charge(created=NOW - timedelta(days=MAX_CHARGE_AGE_DAYS, seconds=1))
    decision = refund_decision(stale, 0.95, now=NOW)
    assert not decision.auto_approve
    assert "days old" in decision.reason


def test_previous_refunds_flag_the_customer():
    repeat = charge(prior_refund_count=1)
    decision = refund_decision(repeat, 0.95, now=NOW)
    assert not decision.auto_approve
    assert "previous refund" in decision.reason


def test_already_refunded_charge_is_flagged():
    decision = refund_decision(charge(refunded=True), 0.95, now=NOW)
    assert not decision.auto_approve
    assert "already refunded" in decision.reason


def test_low_confidence_is_flagged():
    decision = refund_decision(charge(), 0.5, now=NOW)
    assert not decision.auto_approve
    assert "confidence" in decision.reason


def test_missing_charge_is_flagged():
    decision = refund_decision(None, 0.95, now=NOW)
    assert not decision.auto_approve
    assert "no charge" in decision.reason


# ------------------------------------------------------------ idempotency


def test_conversation_awaiting_us_needs_a_reply():
    assert needs_reply(conversation(incoming=True))


def test_conversation_we_answered_last_is_left_alone():
    assert not needs_reply(conversation(incoming=False))


def test_empty_conversation_is_left_alone():
    assert not needs_reply(Conversation(id=1, contact_email=None, messages=()))


def test_loop_skips_conversations_we_already_answered():
    inbox = FakeInbox([conversation(incoming=False)])
    actions = run_once(
        inbox=inbox,
        payments=FakePayments(),
        classify=classifier("support"),
        answer=answerer(),
        knowledge="guidance",
    )
    assert actions == []
    assert inbox.replies == []


# ---------------------------------------------------------------- routing


def test_support_question_is_answered_from_the_knowledge_file():
    action, inbox, payments = run(conversation("how do I reset my password"))
    assert action == "answered"
    assert inbox.replies == [(1, "here is your answer")]
    assert payments.refunded == []


def test_qualifying_refund_goes_through_and_tells_the_customer():
    payments = FakePayments(charge())
    action, inbox, payments = run(
        conversation("I want my money back"), payments=payments, intent="refund"
    )
    assert action == "refunded"
    assert payments.refunded == ["ch_1"]
    assert len(inbox.replies) == 1
    assert "20.00" in inbox.replies[0][1]
    assert inbox.notes == []


def test_oversized_refund_is_held_and_never_charges_stripe():
    payments = FakePayments(charge(amount_cents=90_000))
    action, inbox, payments = run(
        conversation("refund please"), payments=payments, intent="refund"
    )
    assert action == "flagged"
    assert payments.refunded == []
    assert inbox.replies == []
    assert len(inbox.notes) == 1
    assert "limit" in inbox.notes[0][1]


def test_refund_with_no_matching_customer_is_held():
    payments = FakePayments(None)
    action, inbox, payments = run(
        conversation("refund"), payments=payments, intent="refund"
    )
    assert action == "flagged"
    assert payments.refunded == []
    assert "no charge found" in inbox.notes[0][1]


def test_refund_from_an_anonymous_conversation_is_held():
    payments = FakePayments(charge())
    action, inbox, payments = run(
        conversation("refund", email=None), payments=payments, intent="refund"
    )
    assert action == "flagged"
    assert payments.refunded == []


def test_uncertain_refund_never_moves_money():
    payments = FakePayments(charge())
    action, inbox, payments = run(
        conversation("maybe I want a refund?"),
        payments=payments,
        intent="refund",
        confidence=0.4,
    )
    assert action == "flagged"
    assert payments.refunded == []
    assert "confidence" in inbox.notes[0][1]


def test_held_refund_note_carries_a_suggested_reply():
    payments = FakePayments(charge(amount_cents=90_000))
    _, inbox, _ = run(conversation("refund"), payments=payments, intent="refund")
    assert "Suggested reply" in inbox.notes[0][1]


@pytest.mark.parametrize("intent", ["support", "refund"])
def test_nothing_happens_when_we_spoke_last(intent):
    payments = FakePayments(charge())
    action, inbox, payments = run(
        conversation("...", incoming=False), payments=payments, intent=intent
    )
    assert action == "skipped"
    assert inbox.replies == []
    assert inbox.notes == []
    assert payments.refunded == []
