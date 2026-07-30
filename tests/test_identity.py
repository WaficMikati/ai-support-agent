"""Tests for who a conversation is allowed to be about.

The agent used to take the customer straight off the Chatwoot contact, which
meant it knew who you were before you said anything. That is convenient and it
is also the thing that made somebody watching a demo ask, reasonably, how it
knew about them.

An anonymous conversation now starts as a stranger. What ends that is an address
said out loud, and the two lookup models differ only in which addresses count:

  gated  it must be the one this browser registered with
  open   any address at all, which is what a first implementation does

Both read the address out of the conversation in code. Neither lets the model
supply one, so the tools stay argument-free whichever model is in use.
"""

from agent import (
    ANONYMOUS,
    GATED,
    IDENTIFIED,
    LOOKUP_ATTRIBUTE,
    OPEN,
    START_ATTRIBUTE,
    Conversation,
    Message,
    identified_email,
    stated_email,
)

MINE = "alejandro@example.com"
SOMEBODY_ELSE = "victim@example.com"


def talking(*said: str, contact: str | None = MINE, **settings) -> Conversation:
    """A conversation where the customer said these things, oldest first."""
    return Conversation(
        id=1,
        contact_email=contact,
        messages=tuple(
            Message(id=n, content=text, incoming=True)
            for n, text in enumerate(said, start=1)
        ),
        contact_attributes=settings,
    )


# ------------------------------------------------------------- identified


def test_an_identified_conversation_knows_from_the_start():
    """A signed-in session, where saying nothing is still enough."""
    conversation = talking("hello", **{START_ATTRIBUTE: IDENTIFIED})
    assert identified_email(conversation) == MINE


def test_identified_is_the_default_when_nothing_was_set():
    """Existing conversations, and anything the page did not label."""
    assert identified_email(talking("hello")) == MINE


# ------------------------------------------------------------- anonymous


def test_an_anonymous_conversation_starts_as_a_stranger():
    conversation = talking("hello", **{START_ATTRIBUTE: ANONYMOUS})
    assert identified_email(conversation) is None


def test_saying_your_address_identifies_you():
    conversation = talking(
        "hello", f"it is {MINE}", **{START_ATTRIBUTE: ANONYMOUS}
    )
    assert identified_email(conversation) == MINE


def test_an_anonymous_conversation_stays_identified_afterwards():
    """So it asks once rather than on every message."""
    conversation = talking(
        f"my email is {MINE}",
        "so can I get a refund?",
        **{START_ATTRIBUTE: ANONYMOUS},
    )
    assert identified_email(conversation) == MINE


# --------------------------------------------------------- the two models


def test_gated_refuses_somebody_elses_address():
    """The demonstration worth doing: naming a stranger gets you nothing."""
    conversation = talking(
        f"look up {SOMEBODY_ELSE}",
        **{START_ATTRIBUTE: ANONYMOUS, LOOKUP_ATTRIBUTE: GATED},
    )
    assert identified_email(conversation) is None


def test_gated_accepts_your_own_address():
    conversation = talking(
        f"look up {MINE}",
        **{START_ATTRIBUTE: ANONYMOUS, LOOKUP_ATTRIBUTE: GATED},
    )
    assert identified_email(conversation) == MINE


def test_open_accepts_somebody_elses_address():
    """Not a bug. It is what a first implementation does, and the reason to
    show it beside the other one."""
    conversation = talking(
        f"look up {SOMEBODY_ELSE}",
        **{START_ATTRIBUTE: ANONYMOUS, LOOKUP_ATTRIBUTE: OPEN},
    )
    assert identified_email(conversation) == SOMEBODY_ELSE


def test_gated_is_the_default_of_the_two():
    """Leaving it out must not be the permissive one."""
    conversation = talking(
        f"look up {SOMEBODY_ELSE}", **{START_ATTRIBUTE: ANONYMOUS}
    )
    assert identified_email(conversation) is None


def test_gated_ignores_the_case_of_what_they_typed():
    conversation = talking(
        f"it is {MINE.upper()}",
        **{START_ATTRIBUTE: ANONYMOUS, LOOKUP_ATTRIBUTE: GATED},
    )
    assert identified_email(conversation) is not None


# ------------------------------------------------------- reading the thread


def test_only_what_the_customer_said_counts():
    """Our own replies quote addresses back. Reading those would let the agent
    identify somebody using its own words."""
    conversation = Conversation(
        id=1,
        contact_email=MINE,
        messages=(
            Message(id=1, content="hello", incoming=True),
            Message(id=2, content=f"is it {MINE}?", incoming=False),
        ),
        contact_attributes={START_ATTRIBUTE: ANONYMOUS},
    )
    assert identified_email(conversation) is None


def test_the_most_recent_address_wins():
    conversation = talking("first@example.com", "sorry, second@example.com")
    assert stated_email(conversation) == "second@example.com"


def test_a_message_with_no_address_finds_nothing():
    assert stated_email(talking("I want my money back")) is None


def test_an_address_in_a_sentence_is_found():
    assert stated_email(talking("Sure, it's bob.smith+tag@sub.example.co.uk!")) == (
        "bob.smith+tag@sub.example.co.uk"
    )


def test_an_anonymous_contact_with_no_email_cannot_be_gated_into():
    """Nothing to compare against means nothing is let through, rather than
    everything."""
    conversation = talking(
        f"it is {SOMEBODY_ELSE}",
        contact=None,
        **{START_ATTRIBUTE: ANONYMOUS, LOOKUP_ATTRIBUTE: GATED},
    )
    assert identified_email(conversation) is None


# --------------------------------------------- saying what we already knew


def replying(*said, contact=MINE, **settings):
    """A conversation where the last thing said was ours, plus one before."""
    return Conversation(
        id=1,
        contact_email=contact,
        messages=tuple(
            Message(id=n, content=text, incoming=incoming)
            for n, (text, incoming) in enumerate(said, start=1)
        ),
        contact_attributes=settings,
    )


def test_an_identified_chat_says_where_it_got_the_address():
    """The objection this answers is "how did it know about me?", asked by
    somebody who had not typed anything."""
    from agent import announced

    conversation = talking("hello", **{START_ATTRIBUTE: IDENTIFIED})
    said = announced("Your last purchase was $20.00.", conversation)
    assert MINE in said
    assert said.endswith("Your last purchase was $20.00.")


def test_it_only_says_so_once():
    """On the second reply they have already been told."""
    from agent import announced

    conversation = replying(
        ("hello", True), ("hi there", False), ("and my refund?", True),
        **{START_ATTRIBUTE: IDENTIFIED},
    )
    assert announced("here you go", conversation) == "here you go"


def test_an_anonymous_chat_does_not_say_it():
    """They typed their address a moment ago. Telling them we read it is noise."""
    from agent import announced

    conversation = talking(f"it is {MINE}", **{START_ATTRIBUTE: ANONYMOUS})
    assert announced("here you go", conversation) == "here you go"


def test_a_conversation_that_never_registered_is_left_alone():
    """Absent settings mean nobody chose this, so nothing is announced."""
    from agent import announced

    assert announced("here you go", talking("hello")) == "here you go"
