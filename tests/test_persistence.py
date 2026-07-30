"""Tests for remembering handled messages in Chatwoot rather than in memory.

An in-process set cannot manage this. A crash forgets it, and a second copy of
the agent has its own, which is exactly how one message got answered twice: the
running agent and a check script both handled the same conversations and posted
duplicate replies.

Chatwoot's conversation custom_attributes are the memory instead, so the marker
survives a restart and is shared by anything polling the same account. The list
payload already carries those attributes, so reading it costs nothing.
"""

import json

import httpx

from agent import (
    HANDLED_ATTRIBUTE,
    ChatwootInbox,
    Conversation,
    HandledMessages,
    Message,
    Proposal,
    handle_conversation,
)

INCOMING = {"id": 11, "content": "do you have decaf?", "message_type": 0, "private": False}


def conversation(handled_message_id=None, message_id=11) -> Conversation:
    return Conversation(
        id=7,
        contact_email="a@b.com",
        messages=(Message(id=message_id, content="do you have decaf?", incoming=True),),
        handled_message_id=handled_message_id,
    )


class FakeInbox:
    def __init__(self):
        self.replies = []
        self.notes = []
        self.resolved = []
        self.recorded = []

    def open_conversations(self):
        return []

    def send_reply(self, conversation_id, content):
        self.replies.append((conversation_id, content))

    def add_private_note(self, conversation_id, content):
        self.notes.append((conversation_id, content))

    def resolve(self, conversation_id):
        self.resolved.append(conversation_id)

    def record_handled(self, conversation_id, message_id):
        self.recorded.append((conversation_id, message_id))


class FakePayments:
    def latest_charge(self, email):
        return None

    def refund(self, charge_id, idempotency_key):
        raise AssertionError("should not refund in these tests")


def understander(reply="here you go"):
    return lambda turns, knowledge, articles=(), tools=None: Proposal(
        reply=reply,
        refund_requested=False,
        clear_request=True,
        charge_identified=True,
        hedging=False,
    )


def act(conv, inbox=None, handled=None):
    inbox = inbox or FakeInbox()
    action = handle_conversation(
        conv,
        inbox=inbox,
        payments=FakePayments(),
        understand=understander(),
        knowledge="guidance",
        handled=handled,
    )
    return action, inbox


# ------------------------------------------------------ using the marker


def test_a_message_chatwoot_says_we_handled_is_skipped():
    """The case an in-memory set cannot cover: fresh process, same message."""
    action, inbox = act(conversation(handled_message_id=11))
    assert action == "already handled"
    assert inbox.replies == [], "must not answer it a second time"


def test_a_newer_message_is_still_handled():
    action, inbox = act(conversation(handled_message_id=10, message_id=11))
    assert action == "answered"
    assert len(inbox.replies) == 1


def test_no_marker_means_it_has_not_been_handled():
    action, inbox = act(conversation(handled_message_id=None))
    assert action == "answered"


def test_the_marker_is_written_after_handling():
    action, inbox = act(conversation())
    assert action == "answered"
    assert inbox.recorded == [(7, 11)], "Chatwoot must be told, or a restart repeats it"


def test_the_marker_is_not_written_when_nothing_was_done():
    action, inbox = act(conversation(handled_message_id=11))
    assert action == "already handled"
    assert inbox.recorded == []


def test_the_in_memory_set_still_short_circuits_first():
    """Kept as the cheap layer: it saves a round trip within a single run."""
    handled = HandledMessages()
    handled.add(11)
    action, inbox = act(conversation(), handled=handled)
    assert action == "already handled"
    assert inbox.recorded == []


# ------------------------------------------------------ the chatwoot side


def chatwoot(attributes=None, calls=None):
    calls = calls if calls is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.content))
        if request.url.path.endswith("/conversations"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "payload": [
                            {
                                "id": 7,
                                "meta": {"sender": {"email": "a@b.com"}},
                                "messages": [INCOMING],
                                "custom_attributes": attributes or {},
                            }
                        ]
                    }
                },
            )
        return httpx.Response(200, json={"payload": {}})

    inbox = ChatwootInbox(
        "http://chatwoot.test", "1", "token", transport=httpx.MockTransport(handler)
    )
    return inbox, calls


def test_the_marker_is_read_from_the_list_payload():
    inbox, calls = chatwoot({HANDLED_ATTRIBUTE: 11})
    conversation = inbox.open_conversations()[0]
    assert conversation.handled_message_id == 11
    assert len(calls) == 1, "the attributes come with the list, no extra request"


def test_a_string_marker_is_read_as_a_number():
    """Chatwoot round-trips custom attributes as JSON and can hand back "11"."""
    inbox, _ = chatwoot({HANDLED_ATTRIBUTE: "11"})
    assert inbox.open_conversations()[0].handled_message_id == 11


def test_a_missing_or_junk_marker_reads_as_none():
    for attributes in ({}, {HANDLED_ATTRIBUTE: None}, {HANDLED_ATTRIBUTE: "not a number"}):
        inbox, _ = chatwoot(attributes)
        assert inbox.open_conversations()[0].handled_message_id is None


def test_other_custom_attributes_are_left_alone():
    inbox, calls = chatwoot()
    inbox.record_handled(7, 11)
    method, path, content = calls[-1]
    assert method == "POST"
    assert path.endswith("/conversations/7/custom_attributes")
    assert json.loads(content) == {"custom_attributes": {HANDLED_ATTRIBUTE: 11}}
