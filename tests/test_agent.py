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
    Conversation,
    Message,
    Proposal,
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
        self.resolved: list[int] = []
        self.recorded: list[tuple[int, int]] = []

    def open_conversations(self):
        return self.conversations

    def with_history(self, conversation):
        # These fakes are built with their whole thread already in them.
        return conversation

    def send_reply(self, conversation_id, content):
        self.replies.append((conversation_id, content))

    def add_private_note(self, conversation_id, content):
        self.notes.append((conversation_id, content))

    def resolve(self, conversation_id):
        self.resolved.append(conversation_id)

    def record_handled(self, conversation_id, message_id):
        self.recorded.append((conversation_id, message_id))


class FakePayments:
    def __init__(self, charge=None):
        self.charge = charge
        self.refunded: list[str] = []
        self.keys: list[str] = []

    def latest_charge(self, email):
        return self.charge

    def refund(self, charge_id, idempotency_key):
        self.refunded.append(charge_id)
        self.keys.append(idempotency_key)
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


def understander(
    intent="support",
    confidence=None,
    text="here is your answer",
    clear=True,
    charge_named=True,
    hedging=False,
):
    """Stands in for the model: proposes a reply and whether a refund is asked
    for."""
    return lambda turns, knowledge, articles=(), tools=None: Proposal(
        reply=text,
        refund_requested=intent == "refund",
        clear_request=clear,
        charge_identified=charge_named,
        hedging=hedging,
    )


def run(conv, *, inbox=None, payments=None, intent="support", confidence=0.95):
    inbox = inbox or FakeInbox()
    payments = payments or FakePayments()
    action = handle_conversation(
        conv,
        inbox=inbox,
        payments=payments,
        understand=understander(intent, confidence),
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
    assert "already been refunded" in decision.reason


def test_several_refundable_charges_are_flagged_rather_than_guessed():
    ambiguous = charge(sibling_unrefunded_count=2)
    decision = refund_decision(ambiguous, 0.95, now=NOW)
    assert not decision.auto_approve
    assert "3 unrefunded charges" in decision.reason


def test_a_single_refundable_charge_is_not_ambiguous():
    assert refund_decision(charge(sibling_unrefunded_count=0), 0.95, now=NOW).auto_approve


def test_a_low_rubric_score_is_flagged():
    decision = refund_decision(charge(), 1 / 3, now=NOW)
    assert not decision.auto_approve
    assert "rubric score" in decision.reason


def test_two_of_three_rubric_points_is_enough():
    """A plain "I want my money back" names no charge, so it scores two thirds.
    Requiring all three would hold nearly every genuine request."""
    assert refund_decision(charge(), 2 / 3, now=NOW).auto_approve


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
        understand=understander("support"),
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
    assert len(inbox.notes) == 1
    assert "limit" in inbox.notes[0][1]


def test_a_held_refund_still_tells_the_customer_something():
    """Silence is indistinguishable from a broken agent: they asked for a refund
    and nothing came back."""
    payments = FakePayments(charge(amount_cents=90_000))
    action, inbox, _ = run(
        conversation("refund please"), payments=payments, intent="refund"
    )
    assert action == "flagged"
    assert len(inbox.replies) == 1, "the customer must not be left in silence"


def test_a_hedged_request_keeps_the_reply_the_model_wrote():
    """When the doubt is about what the customer meant rather than about the
    payment, the model's words are the better answer: it may have asked
    something useful, and a message that asked two things still gets both."""
    inbox = FakeInbox()
    handle_conversation(
        conversation("cancel me, and maybe a refund?"),
        inbox=inbox,
        payments=FakePayments(charge()),
        understand=understander(
            "refund",
            text="I have cancelled your subscription.",
            clear=False,
            charge_named=False,
            hedging=True,
        ),
        knowledge="guidance",
        now=NOW,
    )
    assert inbox.replies[0][1].startswith("I have cancelled your subscription.")


def test_each_reason_for_holding_gets_its_own_explanation():
    """The model writes before the policy runs, so it cannot know a colleague is
    taking over. Where code knows the payment is the problem, it says so instead
    of leaving a question that is moot by the time it arrives."""
    cases = {
        "already_refunded": (charge(refunded=True), "already been refunded"),
        "ambiguous": (charge(sibling_unrefunded_count=2), "more than one payment"),
        "too_large": (charge(amount_cents=90_000), "needs a colleague to approve"),
        "too_old": (
            charge(created=NOW - timedelta(days=MAX_CHARGE_AGE_DAYS + 1)),
            "older than I am able to refund",
        ),
        "prior_refunds": (charge(prior_refund_count=1), "earlier refunds"),
        "no_charge": (None, "couldn't find any payments"),
    }
    for name, (on_file, expected) in cases.items():
        inbox = FakeInbox()
        handle_conversation(
            conversation("refund me"),
            inbox=inbox,
            payments=FakePayments(on_file),
            understand=understander("refund", text="Which payment do you mean?"),
            knowledge="guidance",
            now=NOW,
        )
        reply = inbox.replies[0][1]
        assert expected in reply, f"{name}: got {reply!r}"
        assert "Which payment do you mean?" not in reply, f"{name} kept a moot question"
        assert "within the next 24 hours" in reply, f"{name} dropped the commitment"


def test_no_explanation_quotes_a_threshold():
    """The colleague's note gives the number. The customer gets the shape of the
    problem without the policy being published back at them."""
    from agent import HELD_EXPLANATIONS

    for name, sentence in HELD_EXPLANATIONS.items():
        assert "50" not in sentence and "30" not in sentence, name
        assert "limit" not in sentence.lower(), name


def test_a_refund_on_an_empty_account_says_there_is_nothing_there():
    """Code knows this and the model does not: asked for a refund it often never
    looks, and asking for a date it cannot use reads badly next to a message
    saying the request has already been passed on."""
    inbox = FakeInbox()
    handle_conversation(
        conversation("refund me"),
        inbox=inbox,
        payments=FakePayments(None),
        understand=understander(
            "refund", text="Could you tell me the date of the payment?"
        ),
        knowledge="guidance",
        now=NOW,
    )
    reply = inbox.replies[0][1]
    assert "couldn't find any payments" in reply
    assert "date of the payment" not in reply, "the useless question is dropped"
    assert "within the next 24 hours" in reply


def test_a_held_refund_always_says_what_happens_next():
    """The customer is owed the same commitment every time, so it is written
    here rather than left to whatever the model happened to say."""
    payments = FakePayments(charge(amount_cents=90_000))
    inbox = FakeInbox()
    handle_conversation(
        conversation("refund please"),
        inbox=inbox,
        payments=payments,
        understand=understander("refund", text="Sorry about that."),
        knowledge="guidance",
        now=NOW,
    )
    reply = inbox.replies[0][1]
    assert "sent your refund request to my colleague for review" in reply
    assert "within the next 24 hours" in reply


def test_a_held_refund_never_tells_the_customer_money_is_coming():
    """Whether to refund is the human's decision. If the model promises anyway,
    the safe wording replaces it."""
    payments = FakePayments(charge(amount_cents=90_000))
    inbox = FakeInbox()
    handle_conversation(
        conversation("refund please"),
        inbox=inbox,
        payments=payments,
        understand=understander(
            "refund", text="Good news, that's refunded and on its way back!"
        ),
        knowledge="guidance",
        now=NOW,
    )
    said = inbox.replies[0][1].lower()
    assert "on its way back" not in said
    assert "colleague" in said


def test_a_held_refund_is_not_resolved():
    payments = FakePayments(charge(amount_cents=90_000))
    _, inbox, _ = run(
        conversation("refund please"), payments=payments, intent="refund"
    )
    assert inbox.resolved == [], "a person still has to act on it"


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


def test_a_hedged_refund_never_moves_money():
    """"maybe I want a refund?" loses the hedging point and the charge point, so
    it lands under the threshold and goes to a human."""
    payments = FakePayments(charge())
    inbox = FakeInbox()
    action = handle_conversation(
        conversation("maybe I want a refund?"),
        inbox=inbox,
        payments=payments,
        understand=understander("refund", charge_named=False, hedging=True),
        knowledge="guidance",
        now=NOW,
    )
    assert action == "flagged"
    assert payments.refunded == []
    assert "rubric score" in inbox.notes[0][1]
    assert "hedging: yes" in inbox.notes[0][1], "the note should say which signal failed"


def test_answered_conversation_is_resolved():
    action, inbox, _ = run(conversation("how do I cancel"))
    assert action == "answered"
    assert inbox.resolved == [1]


def test_refunded_conversation_is_resolved():
    payments = FakePayments(charge())
    action, inbox, _ = run(
        conversation("refund please"), payments=payments, intent="refund"
    )
    assert action == "refunded"
    assert inbox.resolved == [1]


def test_held_refund_stays_open_for_a_human():
    payments = FakePayments(charge(amount_cents=90_000))
    action, inbox, _ = run(
        conversation("refund please"), payments=payments, intent="refund"
    )
    assert action == "flagged"
    assert inbox.resolved == [], "a human still has to act on it"


def test_the_note_records_why_and_that_the_customer_was_told():
    payments = FakePayments(charge(amount_cents=90_000))
    _, inbox, _ = run(conversation("refund"), payments=payments, intent="refund")
    note = inbox.notes[0][1]
    assert "held for review" in note
    assert "limit" in note, "the reason must be in the note"
    assert "colleague will follow up" in note


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


# ------------------------------------------------- the promise safety net


@pytest.mark.parametrize(
    "reply",
    [
        "That's refunded, 20.00 is on its way back to you.",
        "Your refund has been processed.",
        "I have issued your refund today.",
        "You will receive the money within a few days.",
        "The money is back to your card already.",
    ],
)
def test_a_reply_promising_a_refund_is_replaced(reply):
    from agent import HOLDING_REPLY, safe_holding_reply

    assert safe_holding_reply(reply) == HOLDING_REPLY


@pytest.mark.parametrize(
    "reply",
    [
        "Thanks, I'm looking into that now.",
        "I've asked a colleague to check your payment.",
        "Sorry about the stale bag. Let me get someone to look at this.",
        "I have cancelled your subscription, and I'm checking the payment.",
    ],
)
def test_an_honest_reply_is_left_alone(reply):
    from agent import safe_holding_reply

    assert safe_holding_reply(reply) == reply
