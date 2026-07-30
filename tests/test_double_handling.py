"""Tests for the two protections against acting on the same message twice.

Deciding by "who spoke last" is correct but not instant: between classifying a
message and posting the reply, the customer's message is still the newest
thing in the conversation. These are the two things that close that window.
"""

from datetime import datetime, timezone

from agent import (
    Charge,
    Conversation,
    HandledMessages,
    Message,
    Proposal,
    handle_conversation,
    refund_idempotency_key,
    run_once,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


class FakeInbox:
    def __init__(self, conversations=()):
        self.conversations = list(conversations)
        self.replies = []
        self.notes = []
        self.resolved = []
        self.recorded = []

    def open_conversations(self):
        return self.conversations

    def with_history(self, conversation):
        # These fakes are built with their whole thread already in them.
        return conversation

    def send_reply(self, conversation_id, content):
        self.replies.append((conversation_id, content))

    def send_choice(self, conversation_id, content, options):
        self.replies.append((conversation_id, content))
        self.offered = list(options)


    def add_private_note(self, conversation_id, content):
        self.notes.append((conversation_id, content))

    def resolve(self, conversation_id):
        self.resolved.append(conversation_id)

    def record_handled(self, conversation_id, message_id):
        self.recorded.append((conversation_id, message_id))


class FakePayments:
    def __init__(self, charge=None):
        self.charge = charge
        self.refunded = []
        self.keys = []

    def latest_charge(self, email):
        return self.charge

    def charges_for(self, email):
        return [self.charge] if self.charge else []

    def refund(self, charge_id, idempotency_key):
        self.refunded.append(charge_id)
        self.keys.append(idempotency_key)
        return "re_1"


class ExplodingBrain:
    """Fails the first time it is asked, then works."""

    def __init__(self):
        self.calls = 0

    def __call__(self, turns, knowledge, articles=(), tools=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("model API returned 503")
        return Proposal(reply="answer", refund_requested=False, clear_request=True, charge_identified=True, hedging=False)


def charge(**overrides) -> Charge:
    return Charge(
        **{
            "id": "ch_1",
            "amount_cents": 2_000,
            "created": NOW,
            "refunded": False,
            "prior_refund_count": 0,
            **overrides,
        }
    )


def conversation(message_id=1, text="hello", incoming=True) -> Conversation:
    return Conversation(
        id=7,
        contact_email="a@b.com",
        messages=(Message(id=message_id, content=text, incoming=incoming),),
    )


def act(conv, handled, *, inbox=None, payments=None, intent="support", brain=None):
    inbox = inbox or FakeInbox()
    payments = payments or FakePayments(charge())
    action = handle_conversation(
        conv,
        inbox=inbox,
        payments=payments,
        understand=brain
        or (
            lambda turns, knowledge, articles=(), tools=None: Proposal(
                reply="answer",
                refund_requested=intent == "refund",
                clear_request=True,
                charge_identified=True,
                hedging=False,
            )
        ),
        knowledge="guidance",
        handled=handled,
        now=NOW,
    )
    return action, inbox, payments


# ------------------------------------------------------ the remembered set


def test_the_same_message_is_not_answered_twice():
    handled = HandledMessages()
    conv = conversation()

    first, inbox, _ = act(conv, handled)
    assert first == "answered"

    # The same conversation comes back on the next poll, still showing the
    # customer's message as newest because the reply has not landed yet.
    second, inbox_again, _ = act(conv, handled)
    assert second == "already handled"
    assert inbox_again.replies == [], "must not reply a second time"


def test_the_same_message_is_not_refunded_twice():
    handled = HandledMessages()
    payments = FakePayments(charge())
    conv = conversation(text="refund me")

    first, _, payments = act(conv, handled, payments=payments, intent="refund")
    assert first == "refunded"
    assert payments.refunded == ["ch_1"]

    second, _, payments_again = act(
        conv, handled, payments=payments, intent="refund"
    )
    assert second == "already handled"
    assert payments.refunded == ["ch_1"], "one refund, not two"


def test_a_new_message_in_the_same_conversation_is_still_handled():
    handled = HandledMessages()
    act(conversation(message_id=1), handled)
    action, inbox, _ = act(conversation(message_id=2, text="another question"), handled)
    assert action == "answered"
    assert len(inbox.replies) == 1


def test_a_failed_attempt_is_not_marked_as_handled():
    """A transient model failure must leave the message to be retried, not
    swallow it."""
    handled = HandledMessages()
    brain = ExplodingBrain()
    conv = conversation()

    try:
        act(conv, handled, brain=brain)
    except RuntimeError:
        pass
    assert len(handled) == 0, "nothing should be recorded after a failure"

    action, inbox, _ = act(conv, handled, brain=brain)
    assert action == "answered", "the retry succeeds"
    assert len(inbox.replies) == 1


def test_held_refunds_are_remembered_too():
    handled = HandledMessages()
    payments = FakePayments(charge(amount_cents=90_000))
    action, inbox, _ = act(
        conversation(text="refund"), handled, payments=payments, intent="refund"
    )
    assert action == "flagged"
    assert len(handled) == 1, "otherwise the note is added on every poll"


def test_the_set_is_bounded():
    handled = HandledMessages(capacity=3)
    for message_id in range(1, 6):
        handled.add(message_id)
    assert len(handled) == 3
    assert 5 in handled and 4 in handled
    assert 1 not in handled, "oldest ids are dropped rather than growing forever"


def test_the_loop_shares_one_set_across_passes():
    conv = conversation()
    inbox = FakeInbox([conv])
    handled = HandledMessages()
    common = dict(
        inbox=inbox,
        payments=FakePayments(charge()),
        understand=lambda turns, knowledge, articles=(), tools=None: Proposal(
            reply="answer",
            refund_requested=False,
            clear_request=True,
            charge_identified=True,
            hedging=False,
        ),
        knowledge="guidance",
        handled=handled,
    )
    assert run_once(**common) == ["answered"]
    assert run_once(**common) == ["already handled"]
    assert len(inbox.replies) == 1


# -------------------------------------------------------- idempotency key


def test_the_key_is_derived_from_the_conversation_and_charge():
    assert refund_idempotency_key(482, "ch_abc") == "refund-conv482-ch_abc"


def test_the_key_is_stable_for_the_same_request():
    assert refund_idempotency_key(1, "ch_1") == refund_idempotency_key(1, "ch_1")


def test_asking_twice_in_one_conversation_reuses_the_key():
    """Two messages, one charge. Keyed on the message these would differ and
    Stripe would treat the second attempt as a fresh refund."""
    assert refund_idempotency_key(7, "ch_1") == refund_idempotency_key(7, "ch_1")


def test_different_charges_and_conversations_get_different_keys():
    keys = {
        refund_idempotency_key(1, "ch_1"),
        refund_idempotency_key(1, "ch_2"),
        refund_idempotency_key(2, "ch_1"),
    }
    assert len(keys) == 3


def test_the_key_carries_no_personal_data_and_fits_stripes_limit():
    key = refund_idempotency_key(999_999, "ch_3TymYHKS7Y6zQSXz0FzMdd7e")
    assert "@" not in key
    assert len(key) <= 255


def test_the_refund_is_sent_with_the_key_for_that_message():
    handled = HandledMessages()
    payments = FakePayments(charge())
    act(conversation(message_id=91, text="refund"), handled, payments=payments, intent="refund")
    assert payments.keys == ["refund-conv7-ch_1"]
