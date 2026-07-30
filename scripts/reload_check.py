"""Prove that editing knowledge.md changes the answers with no restart.

This is the claim the whole design rests on: support guidance lives in a text
file that a non-developer maintains. So it gets checked against a running
agent rather than asserted.

    uv run python scripts/reload_check.py

Starts agent.py, asks something the knowledge file does not cover, adds a
section while the agent is still running, asks again, and compares. The
knowledge file is restored afterwards either way.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTION = "Do you offer student discounts?"
NEW_SECTION = """

## Student discounts

Yes. There is 30% off for students with a valid student ID. Ask them to email
a photo of the ID and we will apply it to the next payment.
"""

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
    if not ok:
        failures.append(label)


def config() -> dict[str, str]:
    values: dict[str, str] = {}
    for source in (ROOT / ".env", ROOT / "deploy" / "admin.local.txt"):
        for line in source.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


class Customer:
    def __init__(self, settings: dict[str, str]):
        self.inbox_id = int(settings["api_inbox_id"])
        self._client = httpx.Client(
            base_url=f"{settings['CHATWOOT_URL']}/api/v1/accounts"
            f"/{settings['CHATWOOT_ACCOUNT_ID']}",
            headers={"api_access_token": settings["CHATWOOT_TOKEN"]},
            timeout=30,
        )

    def ask(self, email: str, text: str) -> int:
        contact = self._client.post(
            "/contacts",
            json={"name": email.split("@")[0], "email": email, "inbox_id": self.inbox_id},
        )
        contact.raise_for_status()
        body = contact.json()["payload"]
        record = body.get("contact", body)
        conversation = self._client.post(
            "/conversations",
            json={
                "source_id": body["contact_inbox"]["source_id"],
                "inbox_id": self.inbox_id,
                "contact_id": record["id"],
            },
        )
        conversation.raise_for_status()
        conversation_id = conversation.json()["id"]
        message = self._client.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": text, "message_type": "incoming"},
        )
        message.raise_for_status()
        return conversation_id

    def wait_for_reply(self, conversation_id: int, timeout: float = 90.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self._client.get(f"/conversations/{conversation_id}/messages")
            response.raise_for_status()
            for entry in response.json()["payload"]:
                if entry["message_type"] == 1 and not entry.get("private"):
                    return entry.get("content") or ""
            time.sleep(2)
        return ""


def main() -> int:
    settings = config()
    knowledge = ROOT / settings.get("KNOWLEDGE_FILE", "knowledge.md")
    original = knowledge.read_text()
    stamp = str(int(time.time()))
    customer = Customer(settings)
    log_path = ROOT / "reload_check.log"

    print("starting the agent")
    with log_path.open("w") as log_file:
        agent = subprocess.Popen(
            [sys.executable, "agent.py"],
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    try:
        print(f"\nasking something knowledge.md does not cover: {QUESTION!r}")
        first_id = customer.ask(f"reload-a-{stamp}@example.com", QUESTION)
        first = customer.wait_for_reply(first_id)
        check("the agent replied", bool(first), first[:60])
        check(
            "it declines to invent an answer",
            "colleague" in first.lower(),
            first[:80],
        )

        print("\nadding a Student discounts section while the agent keeps running")
        knowledge.write_text(original.rstrip() + NEW_SECTION)
        check("the agent is still the same process", agent.poll() is None)

        print(f"\nasking the same thing again: {QUESTION!r}")
        second_id = customer.ask(f"reload-b-{stamp}@example.com", QUESTION)
        second = customer.wait_for_reply(second_id)
        check("the agent replied", bool(second), second[:60])
        check(
            "it now answers from the new section",
            "30" in second or "student id" in second.lower(),
            second[:100],
        )
        check(
            "and no longer defers to a colleague",
            "colleague" not in second.lower(),
            second[:80],
        )
        check("no restart happened", agent.poll() is None)
    finally:
        agent.terminate()
        try:
            agent.wait(timeout=10)
        except subprocess.TimeoutExpired:
            agent.kill()
        knowledge.write_text(original)
        print(f"\nknowledge.md restored; agent log at {log_path.name}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("live reload confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
