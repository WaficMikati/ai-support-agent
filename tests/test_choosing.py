"""Tests for having more than one payment.

Being told "there is more than one payment, so a colleague will look at it" is a
dead end for something the customer can settle in a word. They are shown what is
on the account and asked which they mean, and the next thing they say picks one.

Which one they picked is read out of their words in code, the same way their
address is. The model can say that a refund is wanted; it never says whose money
or which payment.
"""

from datetime import datetime, timedelta, timezone

from agent import (
    Charge,
    Conversation,
    Message,
    Proposal,
    describe_charges,
    handle_conversation,
    stated_choice,
    why_not,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


def charge(amount, days=0, **overrides) -> Charge:
    return Charge(
        **{
            "id": f"ch_{amount}",
            "amount_cents": amount,
            "created": NOW - timedelta(days=days),
            "refunded": False,
            "prior_refund_count": 0,
            "sibling_unrefunded_count": 1,
            **overrides,
        }
    )


# What the demo page offers for two payments: different amounts and different
# dates, so they can be told apart when they are read back as a choice. The
# second is inside the amount limit and outside the window, so the reason it
# cannot be refunded is the window rather than the amount.
TWO = [charge(2_000), charge(3_500, days=60)]


class Inbox:
    def __init__(self):
        self.replies, self.notes, self.resolved = [], [], []

    def open_conversations(self):
        return []

    def with_history(self, conversation):
        return conversation

    def send_reply(self, conversation_id, content):
        self.replies.append(content)

    def add_private_note(self, conversation_id, content):
        self.notes.append(content)

    def resolve(self, conversation_id):
        self.resolved.append(conversation_id)

    def record_handled(self, conversation_id, message_id):
        pass


class Payments:
    def __init__(self, charges):
        self.charges, self.refunded = charges, []

    def latest_charge(self, email):
        return self.charges[0] if self.charges else None

    def charges_for(self, email):
        return list(self.charges)

    def refund(self, charge_id, idempotency_key):
        self.refunded.append(charge_id)
        return "re_1"


def said(*texts) -> Conversation:
    return Conversation(
        id=1,
        contact_email="a@b.com",
        messages=tuple(
            Message(id=n, content=t, incoming=True) for n, t in enumerate(texts, 1)
        ),
    )


def act(conversation, charges):
    inbox, payments = Inbox(), Payments(charges)
    action = handle_conversation(
        conversation,
        inbox=inbox,
        payments=payments,
        understand=lambda turns, knowledge, articles=(), tools=None: Proposal(
            reply="looking into it",
            refund_requested=True,
            clear_request=True,
            charge_identified=False,
            hedging=False,
        ),
        knowledge="guidance",
        now=NOW,
    )
    return action, inbox, payments


# ------------------------------------------------------------ being asked


def test_two_payments_are_offered_as_a_choice_rather_than_escalated():
    action, inbox, payments = act(said("I want a refund"), TWO)
    assert action == "asked which"
    assert payments.refunded == [], "nothing moves while they are deciding"
    assert inbox.notes == [], "a question is not a hand-off"
    assert inbox.resolved == [], "left open for their answer"


def test_the_choice_names_both_amounts_and_dates():
    _, inbox, _ = act(said("I want a refund"), TWO)
    reply = inbox.replies[0]
    assert "$20.00" in reply and "$35.00" in reply
    assert "30 July 2026" in reply and "31 May 2026" in reply


def test_a_payment_it_cannot_refund_is_still_offered_and_says_why():
    """Leaving it out would answer with a shorter list than the truth, and drop
    the very one they were about to ask about."""
    _, inbox, _ = act(said("I want a refund"), TWO)
    assert "outside the refund window" in inbox.replies[0]


# ------------------------------------------------------------ answering it


def test_naming_an_amount_refunds_that_payment():
    action, inbox, payments = act(
        said("I want a refund", "the $20 one please"), TWO
    )
    assert action == "refunded"
    assert payments.refunded == ["ch_2000"]


def test_naming_the_one_that_cannot_be_refunded_is_held_for_its_own_reason():
    """Not "there are two", which they have already settled, but the thing
    actually wrong with the payment they chose."""
    action, inbox, payments = act(said("I want a refund", "the $35 one"), TWO)
    assert action == "flagged"
    assert payments.refunded == []
    assert "older than I am able to refund" in inbox.replies[0]


def test_the_newest_and_oldest_can_be_named_in_words():
    assert stated_choice(said("the most recent one"), TWO).id == "ch_2000"
    assert stated_choice(said("the oldest"), TWO).id == "ch_3500"


def test_an_answer_naming_neither_asks_again():
    action, _, payments = act(said("I want a refund", "yes please"), TWO)
    assert action == "asked which"
    assert payments.refunded == []


def test_only_the_newest_thing_they_said_counts():
    """So changing their mind works."""
    chosen = stated_choice(said("the $20 one", "actually the $35 one"), TWO)
    assert chosen.id == "ch_3500"


def test_one_payment_needs_no_choosing():
    action, _, payments = act(said("I want a refund"), [charge(2_000, sibling_unrefunded_count=0)])
    assert action == "refunded"
    assert payments.refunded == ["ch_2000"]


# ------------------------------------------------------ what stands in the way


def test_why_not_reports_the_same_order_the_policy_uses():
    assert why_not(charge(2_000), NOW) == ""
    assert "outside the refund window" in why_not(charge(2_000, days=60), NOW)
    assert "over the amount" in why_not(charge(90_000), NOW)
    assert "already refunded" in why_not(charge(2_000, refunded=True), NOW)
    assert "disputed" in why_not(charge(2_000, disputed=True), NOW)


def test_a_disputed_payment_is_reported_as_disputed_not_as_refunded():
    both = charge(2_000, refunded=True, disputed=True)
    assert "disputed" in why_not(both, NOW)
