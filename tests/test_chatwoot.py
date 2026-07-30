"""Tests for the Chatwoot adapter.

These run against canned Chatwoot JSON rather than a live instance, because
the two bugs they guard against live in the mapping between Chatwoot's
payload and our own types, not in the policy.
"""

import httpx
import pytest

from agent import ChatwootInbox, needs_reply

ACCOUNT = "1"
TOKEN = "test-token"

INCOMING = {"id": 1, "content": "I want a refund", "message_type": 0, "private": False}
OUTGOING = {"id": 2, "content": "on it", "message_type": 1, "private": False}
PRIVATE_NOTE = {"id": 3, "content": "held for review", "message_type": 1, "private": True}
ACTIVITY = {"id": 4, "content": "Label added", "message_type": 2, "private": False}
# What the widget appends after a visitor's first message, asking for an email.
TEMPLATE = {
    "id": 5,
    "content": "Give the team a way to reach you.",
    "message_type": 3,
    "private": False,
}

CONVERSATION_ENTRY = {
    "id": 7,
    "meta": {"sender": {"email": "customer@example.com"}},
}


def inbox_for(messages) -> tuple[ChatwootInbox, list[httpx.Request]]:
    """A ChatwootInbox wired to canned responses. Also returns the requests
    made, so we can assert on what was sent."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/messages") and request.method == "GET":
            return httpx.Response(200, json={"payload": list(messages)})
        if request.url.path.endswith("/messages") and request.method == "POST":
            return httpx.Response(200, json={"id": 99})
        if request.url.path.endswith("/conversations"):
            return httpx.Response(
                200, json={"data": {"payload": [CONVERSATION_ENTRY]}}
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = ChatwootInbox(
        "http://chatwoot.test",
        ACCOUNT,
        TOKEN,
        transport=httpx.MockTransport(handler),
    )
    return client, seen


def only_conversation(messages):
    client, _ = inbox_for(messages)
    conversations = client.open_conversations()
    assert len(conversations) == 1
    return conversations[0]


# ------------------------------------------------------------------ mapping


def test_contact_email_is_read_from_the_sender():
    assert only_conversation([INCOMING]).contact_email == "customer@example.com"


def test_incoming_and_outgoing_are_distinguished():
    messages = only_conversation([INCOMING, OUTGOING]).messages
    assert [m.incoming for m in messages] == [True, False]


def test_private_notes_are_kept_and_flagged():
    messages = only_conversation([INCOMING, PRIVATE_NOTE]).messages
    assert len(messages) == 2, "private notes must survive the mapping"
    assert messages[-1].private


def test_activity_entries_are_marked_as_activity():
    messages = only_conversation([INCOMING, ACTIVITY]).messages
    assert messages[-1].activity


# -------------------------------------------------------------- idempotency
# Both of these failed before the mapping was fixed.


def test_a_held_refund_is_not_picked_up_again():
    """A private note is our turn. Without this the agent re-flags the same
    conversation on every poll, once every few seconds, forever."""
    conversation = only_conversation([INCOMING, PRIVATE_NOTE])
    assert not needs_reply(conversation)


def test_an_activity_entry_does_not_mute_the_conversation():
    """A label or assignment landing after the customer writes in must not
    look like a reply."""
    conversation = only_conversation([INCOMING, ACTIVITY])
    assert needs_reply(conversation)


def test_a_widget_template_prompt_does_not_mute_the_conversation():
    """The real widget appends "give the team a way to reach you" as a template
    message straight after a visitor's first message. Counting that as our turn
    meant no widget visitor was ever answered."""
    conversation = only_conversation([INCOMING, TEMPLATE])
    assert needs_reply(conversation)


def test_templates_are_marked_as_templates():
    messages = only_conversation([INCOMING, TEMPLATE]).messages
    assert messages[-1].template
    assert not messages[-1].activity


def test_a_real_reply_after_a_template_still_closes_the_turn():
    conversation = only_conversation([INCOMING, TEMPLATE, OUTGOING])
    assert not needs_reply(conversation)


def test_customer_writing_again_after_a_note_reopens_it():
    followup = {**INCOMING, "id": 5, "content": "any update?"}
    conversation = only_conversation([INCOMING, PRIVATE_NOTE, followup])
    assert needs_reply(conversation)


# ------------------------------------------------------------------ writing


@pytest.mark.parametrize(
    ("method", "expect_private"),
    [("send_reply", False), ("add_private_note", True)],
)
def test_posting_sets_the_private_flag_correctly(method, expect_private):
    client, seen = inbox_for([INCOMING])
    getattr(client, method)(7, "hello")
    posts = [r for r in seen if r.method == "POST"]
    assert len(posts) == 1
    import json

    body = json.loads(posts[0].content)
    assert body == {
        "content": "hello",
        "message_type": "outgoing",
        "private": expect_private,
    }


def test_requests_carry_the_access_token():
    client, seen = inbox_for([INCOMING])
    client.open_conversations()
    assert all(r.headers["api_access_token"] == TOKEN for r in seen)


# ------------------------------------------------- tapping one of the buttons


def tapped(chosen: str) -> dict:
    """The question we asked, after somebody tapped an answer on it."""
    return {
        "id": 42,
        "content": "Which would you like refunded?",
        "message_type": 1,
        "content_type": "input_select",
        "content_attributes": {
            "items": [{"title": chosen, "value": chosen}],
            "submitted_values": [{"title": chosen, "value": chosen}],
        },
    }


def test_a_tapped_option_becomes_something_the_customer_said():
    """Tapping sends no message. Chatwoot writes it onto the question, which is
    ours, so without this who spoke last never changes and the conversation
    stops with the customer having already answered."""
    inbox, _ = inbox_for([tapped("$20.00 on 30 July 2026")])
    conversation = inbox.open_conversations()[0]
    assert conversation.latest is not None
    assert conversation.latest.incoming, "the tap is the customer's turn"
    assert conversation.latest.content == "$20.00 on 30 July 2026"


def test_a_tap_needs_a_reply():
    inbox, _ = inbox_for([tapped("$20.00 on 30 July 2026")])
    assert needs_reply(inbox.open_conversations()[0])


def test_a_tap_reads_as_a_turn_the_model_can_see():
    inbox, _ = inbox_for([tapped("$35.00 on 23 July 2026")])
    turns = inbox.open_conversations()[0].turns()
    assert turns[-1].role == "user"
    assert turns[-1].content == "$35.00 on 23 July 2026"


def test_an_untouched_question_is_still_only_ours():
    waiting = tapped("$20.00 on 30 July 2026")
    del waiting["content_attributes"]["submitted_values"]
    inbox, _ = inbox_for([waiting])
    assert not needs_reply(inbox.open_conversations()[0]), "nobody has answered yet"


def test_a_tap_gets_an_id_that_cannot_collide_with_a_real_message():
    """Acting on it twice is prevented by the same means as anything else, so
    the id has to be unique and the same on every poll."""
    inbox, _ = inbox_for([tapped("$20.00 on 30 July 2026")])
    messages = inbox.open_conversations()[0].messages
    assert [m.id for m in messages] == [42, -42]
