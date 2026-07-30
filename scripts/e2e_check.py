"""End-to-end check: real Chatwoot, real model, real Stripe test mode.

Every dependency is the live one. The model really classifies, Stripe really
issues a refund, Chatwoot really receives the reply.

    uv run python scripts/e2e_check.py

Needs .env with MODEL_API_KEY, STRIPE_API_KEY, CHATWOOT_* and, for the refund
scenarios, STRIPE_SETUP_KEY. The model is resolved exactly as agent.py does it,
so this always checks the provider that is actually running.

Why a second Stripe key: the agent's restricted key is deliberately
Customers=Read, so it cannot create the customer and charge a refund test
needs. Fixture setup therefore uses the sandbox's ordinary secret key
(sk_test_...). Scenarios that do not need a charge run without it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import (  # noqa: E402
    MAX_AUTO_REFUND_CENTS,
    Brain,
    ChatwootInbox,
    StripePayments,
    Turn,
    env_value,
    handle_conversation,
    load_env_file,
    refund_idempotency_key,
)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
    if not ok:
        failures.append(label)


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def load_local(name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / "deploy" / name).read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


# ------------------------------------------------------------------ stripe


class StripeSetup:
    """Creates fixtures with a broader key than the agent's."""

    def __init__(self, secret_key: str):
        self._client = httpx.Client(
            base_url="https://api.stripe.com/v1",
            headers={"Authorization": f"Bearer {secret_key}"},
            timeout=30,
        )

    def customer_with_charge(self, email: str, amount_cents: int) -> str:
        found = self._client.get("/customers", params={"email": email, "limit": 1})
        found.raise_for_status()
        existing = found.json()["data"]
        if existing:
            customer_id = existing[0]["id"]
        else:
            created = self._client.post(
                "/customers", data={"email": email, "source": "tok_visa"}
            )
            created.raise_for_status()
            customer_id = created.json()["id"]

        charges = self._client.get(
            "/charges", params={"customer": customer_id, "limit": 1}
        )
        charges.raise_for_status()
        rows = charges.json()["data"]
        if rows and not rows[0]["refunded"] and rows[0]["amount"] == amount_cents:
            return rows[0]["id"]

        charge = self._client.post(
            "/charges",
            data={
                "amount": amount_cents,
                "currency": "usd",
                "customer": customer_id,
                "description": "e2e fixture",
            },
        )
        charge.raise_for_status()
        return charge.json()["id"]


# ---------------------------------------------------------------- chatwoot


class ChatwootAdmin:
    def __init__(self, base_url: str, account_id: str, token: str, inbox_id: int):
        self._inbox_id = inbox_id
        self._client = httpx.Client(
            base_url=f"{base_url}/api/v1/accounts/{account_id}",
            headers={"api_access_token": token},
            timeout=30,
        )

    def conversation_with(self, email: str, text: str) -> int:
        contact = self._client.post(
            "/contacts",
            json={"name": email.split("@")[0], "email": email, "inbox_id": self._inbox_id},
        )
        if contact.status_code == 422:
            search = self._client.get("/contacts/search", params={"q": email})
            search.raise_for_status()
            record = search.json()["payload"][0]
            contact_id = record["id"]
            source_id = record["contact_inboxes"][0]["source_id"]
        else:
            contact.raise_for_status()
            body = contact.json()["payload"]
            record = body.get("contact", body)
            contact_id = record["id"]
            source_id = body["contact_inbox"]["source_id"]

        conversation = self._client.post(
            "/conversations",
            json={
                "source_id": source_id,
                "inbox_id": self._inbox_id,
                "contact_id": contact_id,
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

    def messages(self, conversation_id: int) -> list[dict]:
        response = self._client.get(f"/conversations/{conversation_id}/messages")
        response.raise_for_status()
        return response.json()["payload"]

    def status(self, conversation_id: int) -> str:
        response = self._client.get(f"/conversations/{conversation_id}")
        response.raise_for_status()
        body = response.json()
        return (body.get("payload") or body).get("status", "")


# --------------------------------------------------------------------- run


def agent_already_running() -> bool:
    """True if a polling agent is up.

    This script drives handle_conversation itself, so a running agent handles
    the same conversations in parallel and every assertion about "one reply" or
    "one note" fails. Worth refusing outright: the failure otherwise looks like
    a bug in the agent rather than two processes doing the same work. The money
    stays safe either way, because the Stripe idempotency key catches it, which
    is exactly what that key is for.
    """
    try:
        found = subprocess.run(
            ["pgrep", "-f", "python agent.py"], capture_output=True, text=True, timeout=10
        )
        return found.returncode == 0 and bool(found.stdout.strip())
    except Exception:
        return False


def main() -> int:
    if agent_already_running() and "--allow-agent" not in sys.argv:
        print(
            "REFUSING TO RUN: agent.py is polling.\n"
            "  It would handle these conversations too, so every assertion about\n"
            "  a single reply or a single note would fail. Stop the agent, or pass\n"
            "  --allow-agent if you know what you are looking at."
        )
        return 1

    env = load_env()
    local = load_local("admin.local.txt")

    base_url = env["CHATWOOT_URL"]
    account_id = env["CHATWOOT_ACCOUNT_ID"]
    token = env["CHATWOOT_TOKEN"]
    api_inbox = int(local["api_inbox_id"])

    stripe_key = env["STRIPE_API_KEY"]
    if not stripe_key.startswith("rk_test_") and not stripe_key.startswith("sk_test_"):
        print(f"REFUSING TO RUN: STRIPE_API_KEY is not a test key ({stripe_key[:8]}...)")
        return 1

    knowledge = (ROOT / env.get("KNOWLEDGE_FILE", "knowledge.md")).read_text()
    inbox = ChatwootInbox(base_url, account_id, token)
    payments = StripePayments(stripe_key)
    # Resolve the model exactly as agent.py does, or this checks a different
    # provider than the one actually running.
    load_env_file(ROOT / ".env")
    brain = Brain(
        env_value("MODEL_API_KEY", "OPENROUTER_API_KEY"),
        model=env_value("MODEL_NAME", "OPENROUTER_MODEL", default=Brain.DEFAULT_MODEL),
        base_url=env_value(
            "MODEL_BASE_URL", "OPENROUTER_BASE_URL", default=Brain.DEFAULT_BASE_URL
        ),
    )
    print(f"  model: {brain._model} via {env_value('MODEL_BASE_URL', 'OPENROUTER_BASE_URL', default=Brain.DEFAULT_BASE_URL)}")
    admin = ChatwootAdmin(base_url, account_id, token, api_inbox)
    stamp = str(int(time.time()))

    def act(conversation_id: int) -> str:
        for candidate in inbox.open_conversations():
            if candidate.id == conversation_id:
                return handle_conversation(
                    candidate,
                    inbox=inbox,
                    payments=payments,
                    understand=brain.understand,
                    knowledge=knowledge,
                )
        raise AssertionError(f"conversation {conversation_id} not visible")

    # ---------------------------------------------------------- model only
    print("\n1. the model classifies (real call)")
    refund_call = brain.understand(
        (Turn(role="user", content="I want my money back for the bootcamp"),),
        knowledge,
    )
    check("a refund request is recognised", refund_call.refund_requested,
          f"refund_requested={refund_call.refund_requested} @ {refund_call.confidence:.2f}")
    support_call = brain.understand(
        (Turn(role="user", content="my delivery has not arrived"),), knowledge
    )
    check("a support question is not", not support_call.refund_requested,
          f"refund_requested={support_call.refund_requested}")

    # ------------------------------------------------------- support path
    print("\n2. support question answered from knowledge.md")
    conversation_id = admin.conversation_with(
        f"support-{stamp}@example.com", "I can't log in and the reset email never arrives"
    )
    action = act(conversation_id)
    check("action is 'answered'", action == "answered", action)
    posted = [m for m in admin.messages(conversation_id) if m["message_type"] == 1]
    check("a public reply was posted", len(posted) == 1)
    if posted:
        body = (posted[0].get("content") or "").lower()
        check("reply is not private", not posted[0].get("private"))
        check("reply draws on the knowledge file",
              any(word in body for word in ("spam", "reset", "sign-in", "sign in", "colleague")),
              (posted[0].get("content") or "")[:70])
    check("answered conversation is resolved", admin.status(conversation_id) == "resolved",
          admin.status(conversation_id))

    # ------------------------------------------------- refund, no customer
    print("\n3. refund request with no matching Stripe customer is held")
    conversation_id = admin.conversation_with(
        f"ghost-{stamp}@example.com", "Please refund my payment, I want my money back"
    )
    action = act(conversation_id)
    check("action is 'flagged'", action == "flagged", action)
    notes = [m for m in admin.messages(conversation_id) if m.get("private")]
    check("a private note was left", len(notes) == 1)
    if notes:
        check("note explains why", "no charge found" in (notes[0].get("content") or ""),
              (notes[0].get("content") or "")[:60])
    check("held conversation stays open for a human",
          admin.status(conversation_id) == "open", admin.status(conversation_id))

    # ----------------------------------------------------- refund fixtures
    setup_key = env.get("STRIPE_SETUP_KEY", "")
    if not setup_key:
        print("\n4-5. SKIPPED: no STRIPE_SETUP_KEY in .env")
        print("     The agent key is Customers=Read, so it cannot create a test")
        print("     customer or charge. Add the sandbox secret key (sk_test_...)")
        print("     as STRIPE_SETUP_KEY to exercise a real refund.")
    else:
        if not setup_key.startswith("sk_test_"):
            print(f"\nREFUSING: STRIPE_SETUP_KEY is not sk_test_ ({setup_key[:8]}...)")
            return 1
        setup = StripeSetup(setup_key)

        print("\n4. qualifying refund goes through")
        email = f"refund-{stamp}@example.com"
        charge_id = setup.customer_with_charge(email, 2_000)
        conversation_id = admin.conversation_with(
            email, "I want a refund for my bootcamp payment please"
        )
        action = act(conversation_id)
        check("action is 'refunded'", action == "refunded", action)
        charge = httpx.get(
            f"https://api.stripe.com/v1/charges/{charge_id}",
            headers={"Authorization": f"Bearer {stripe_key}"},
            timeout=30,
        )
        charge.raise_for_status()
        check("Stripe shows the charge refunded", charge.json()["refunded"] is True)
        replies = [
            m for m in admin.messages(conversation_id)
            if m["message_type"] == 1 and not m.get("private")
        ]
        check("customer was told", len(replies) == 1,
              (replies[0].get("content") or "")[:60] if replies else "")
        check("refunded conversation is resolved",
              admin.status(conversation_id) == "resolved", admin.status(conversation_id))

        print("\n5. oversized refund is held, money untouched")
        email = f"big-{stamp}@example.com"
        big_charge = setup.customer_with_charge(email, MAX_AUTO_REFUND_CENTS * 18)
        conversation_id = admin.conversation_with(
            email, "Refund me please, I want my money back"
        )
        action = act(conversation_id)
        check("action is 'flagged'", action == "flagged", action)
        charge = httpx.get(
            f"https://api.stripe.com/v1/charges/{big_charge}",
            headers={"Authorization": f"Bearer {stripe_key}"},
            timeout=30,
        )
        charge.raise_for_status()
        check("Stripe shows it NOT refunded", charge.json()["refunded"] is False)
        notes = [m for m in admin.messages(conversation_id) if m.get("private")]
        check("held with a reason", bool(notes) and "limit" in (notes[0].get("content") or ""),
              (notes[0].get("content") or "")[:60] if notes else "")

        # No model calls here: this is the backstop for a crash between issuing
        # the refund and posting the reply, when the remembered message ids are
        # gone and the agent tries the same refund again.
        print("\n6. the same refund sent twice only moves money once")
        email = f"twice-{stamp}@example.com"
        charge_id = setup.customer_with_charge(email, 1_500)
        # Unique per run. Stripe keeps keys for at least 24 hours and rejects a
        # reused key that arrives with different parameters, so a fixed key here
        # fails on the second run of the day against a different charge.
        key = refund_idempotency_key(int(stamp), charge_id)
        first = payments.refund(charge_id, key)
        second = payments.refund(charge_id, key)
        check("Stripe returned the same refund both times", first == second,
              f"{first} vs {second}")
        refunds = httpx.get(
            "https://api.stripe.com/v1/refunds",
            params={"charge": charge_id, "limit": 10},
            headers={"Authorization": f"Bearer {setup_key}"},
            timeout=30,
        )
        refunds.raise_for_status()
        rows = refunds.json()["data"]
        check("exactly one refund exists on the charge", len(rows) == 1, f"{len(rows)} found")
        check("and it is for the full amount once", sum(r["amount"] for r in rows) == 1_500,
              str(sum(r["amount"] for r in rows)))

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("end-to-end checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
