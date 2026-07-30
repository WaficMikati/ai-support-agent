"""Tests for how much the poll costs.

Refunds held for a human stay open by design, so the queue of open
conversations grows. Fetching each one's messages every poll made the agent
measurably slower as that queue grew: measured on a real instance, eight held
refunds took a reply from about 2 seconds to about 4.

The list response already carries the newest message, which is all the loop
needs. These tests pin down that the extra fetch happens only when it must.
"""

import httpx

from agent import ChatwootInbox, needs_reply

INCOMING = {"id": 1, "content": "hello", "message_type": 0, "private": False}
OUTGOING = {"id": 2, "content": "answered", "message_type": 1, "private": False}
NOTE = {"id": 3, "content": "held for review", "message_type": 1, "private": True}
ACTIVITY = {"id": 4, "content": "Label added", "message_type": 2, "private": False}
TEMPLATE = {"id": 5, "content": "give us your email", "message_type": 3, "private": False}


def inbox_for(conversations, full_threads=None):
    """conversations: list of (id, inline_messages). full_threads: what the
    per-conversation endpoint returns, keyed by id."""
    full_threads = full_threads or {}
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        seen.append(path)
        if path.endswith("/conversations"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "payload": [
                            {
                                "id": cid,
                                "meta": {"sender": {"email": f"c{cid}@example.com"}},
                                "messages": inline,
                            }
                            for cid, inline in conversations
                        ]
                    }
                },
            )
        if path.endswith("/messages"):
            cid = int(path.split("/conversations/")[1].split("/")[0])
            return httpx.Response(200, json={"payload": full_threads.get(cid, [])})
        raise AssertionError(f"unexpected {path}")

    client = ChatwootInbox(
        "http://chatwoot.test", "1", "token", transport=httpx.MockTransport(handler)
    )
    return client, seen


def message_fetches(seen: list[str]) -> int:
    return sum(1 for path in seen if path.endswith("/messages"))


# ------------------------------------------------------------- the fast path


def test_one_request_serves_a_whole_inbox():
    inbox, seen = inbox_for([(n, [INCOMING]) for n in range(1, 21)])
    conversations = inbox.open_conversations()
    assert len(conversations) == 20
    assert message_fetches(seen) == 0, "twenty conversations must not cost twenty fetches"
    assert len(seen) == 1


def test_the_newest_message_is_read_from_the_list():
    inbox, seen = inbox_for([(7, [INCOMING])])
    conversation = inbox.open_conversations()[0]
    assert conversation.latest is not None
    assert conversation.latest.content == "hello"
    assert needs_reply(conversation)
    assert message_fetches(seen) == 0


def test_a_conversation_we_answered_needs_no_fetch_either():
    inbox, seen = inbox_for([(7, [OUTGOING])])
    assert not needs_reply(inbox.open_conversations()[0])
    assert message_fetches(seen) == 0


def test_a_held_refund_needs_no_fetch():
    """These are the ones that pile up, so they must be the cheap case."""
    inbox, seen = inbox_for([(7, [NOTE])])
    conversation = inbox.open_conversations()[0]
    assert not needs_reply(conversation), "a private note is our turn"
    assert message_fetches(seen) == 0


def test_the_contact_email_still_comes_through():
    inbox, _ = inbox_for([(7, [INCOMING])])
    assert inbox.open_conversations()[0].contact_email == "c7@example.com"


# ------------------------------------------------------------- the fall back


def test_a_trailing_activity_entry_forces_a_fetch():
    """The list only carries the newest message, and an activity entry says
    nothing about who spoke last."""
    inbox, seen = inbox_for(
        [(7, [ACTIVITY])], full_threads={7: [INCOMING, ACTIVITY]}
    )
    conversation = inbox.open_conversations()[0]
    assert message_fetches(seen) == 1
    assert needs_reply(conversation), "the customer spoke before the label was added"


def test_a_trailing_template_entry_forces_a_fetch():
    inbox, seen = inbox_for(
        [(7, [TEMPLATE])], full_threads={7: [INCOMING, TEMPLATE]}
    )
    conversation = inbox.open_conversations()[0]
    assert message_fetches(seen) == 1
    assert needs_reply(conversation), "the widget prompt is not a reply"


def test_no_messages_in_the_list_forces_a_fetch():
    inbox, seen = inbox_for([(7, [])], full_threads={7: [INCOMING]})
    assert needs_reply(inbox.open_conversations()[0])
    assert message_fetches(seen) == 1


def test_only_the_conversations_that_need_it_are_fetched():
    inbox, seen = inbox_for(
        [(1, [INCOMING]), (2, [ACTIVITY]), (3, [NOTE]), (4, [TEMPLATE])],
        full_threads={2: [INCOMING, ACTIVITY], 4: [INCOMING, TEMPLATE]},
    )
    inbox.open_conversations()
    assert message_fetches(seen) == 2, "only the activity and template ones"


def test_malformed_rows_are_ignored_rather_than_crashing():
    inbox, seen = inbox_for([(7, [{"no_id": True}])], full_threads={7: [INCOMING]})
    assert needs_reply(inbox.open_conversations()[0])
    assert message_fetches(seen) == 1
