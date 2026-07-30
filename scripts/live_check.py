"""Drive the real Chatwoot adapter against a running instance.

The unit tests cover the mapping with canned JSON. This checks the same
behaviour against a live Chatwoot, which is where the message_type and
private-note semantics actually get decided.

    uv run python scripts/live_check.py

Reads deploy/admin.local.txt for the token and inbox ids. Needs no model
and no payment provider: it only exercises the inbox.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ChatwootInbox, needs_reply  # noqa: E402

CONFIG = Path(__file__).resolve().parent.parent / "deploy" / "admin.local.txt"

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


def load_config() -> dict[str, str]:
    values = {}
    for line in CONFIG.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def main() -> int:
    config = load_config()
    base_url = config["chatwoot_url"]
    token = config["access_token"]
    account_id = config["account_id"]
    api_inbox_id = int(config["api_inbox_id"])

    api = httpx.Client(
        base_url=f"{base_url}/api/v1/accounts/{account_id}",
        headers={"api_access_token": token},
        timeout=20,
    )

    print("setting up a conversation in the API inbox")
    contact = api.post(
        "/contacts",
        json={
            "name": "Live Check",
            "email": "live-check@example.com",
            "inbox_id": api_inbox_id,
        },
    )
    if contact.status_code == 422:  # already exists from an earlier run
        search = api.get("/contacts/search", params={"q": "live-check@example.com"})
        search.raise_for_status()
        payload = search.json()["payload"][0]
        contact_id = payload["id"]
        source_id = payload["contact_inboxes"][0]["source_id"]
    else:
        contact.raise_for_status()
        body = contact.json()["payload"]
        record = body.get("contact", body)
        contact_id = record["id"]
        source_id = body["contact_inbox"]["source_id"]

    conversation = api.post(
        "/conversations",
        json={
            "source_id": source_id,
            "inbox_id": api_inbox_id,
            "contact_id": contact_id,
        },
    )
    conversation.raise_for_status()
    conversation_id = conversation.json()["id"]
    print(f"  conversation {conversation_id}")

    def customer_says(text: str) -> None:
        response = api.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": text, "message_type": "incoming"},
        )
        response.raise_for_status()

    inbox = ChatwootInbox(base_url, account_id, token)

    def reload():
        for candidate in inbox.open_conversations():
            if candidate.id == conversation_id:
                return candidate
        return None

    print("\ncustomer asks for a refund")
    customer_says("I want a refund for my bootcamp payment")
    current = reload()
    check("conversation is visible to the agent", current is not None)
    if current is None:
        return 1
    check("customer message is read as incoming", current.messages[-1].incoming)
    check("agent sees it as needing a reply", needs_reply(current))

    print("\nagent holds it for a human (private note only)")
    inbox.add_private_note(conversation_id, "Refund held for review. Reason: test")
    current = reload()
    assert current is not None
    check(
        "private note survives the mapping",
        any(message.private for message in current.messages),
    )
    check(
        "conversation is NOT picked up again (the bug this guards)",
        not needs_reply(current),
    )

    print("\ncustomer follows up")
    customer_says("any update?")
    current = reload()
    assert current is not None
    check("follow-up reopens it", needs_reply(current))

    print("\nagent replies to the customer")
    inbox.send_reply(conversation_id, "Looking into it now.")
    current = reload()
    assert current is not None
    check("public reply closes the turn", not needs_reply(current))

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("all live checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
