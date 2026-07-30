"""Put the demo back to a clean starting point.

    uv run python scripts/reset_demo.py

Wipes the conversations, contacts and test payment data, then recreates one
refundable charge so the refund path can be shown straight away. Keeps
everything that took setting up: the account, the three inboxes and their
tokens, the IMAP settings, the admin user and its online availability.

Why server side: the widget remembers a visitor in a signed token, and neither
$chatwoot.reset() nor deleting the page's cookie clears it, because the widget
iframe is its own origin and keeps its own copy. Deleting the contact is what
actually gives you a new visitor. Failing that, use a private window.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"
DEMO_EMAIL = "demo@example.com"
DEMO_AMOUNT_CENTS = 2_000

# Test fixtures created by the check scripts, safe to delete.
TEST_PREFIXES = (
    "refund-", "big-", "twice-", "support-", "ghost-", "live-check",
    "reload-a-", "reload-b-", "loop-", "demo@", "customer@",
)


def settings() -> dict[str, str]:
    values: dict[str, str] = {}
    for source in (ROOT / ".env", DEPLOY / "admin.local.txt"):
        if not source.exists():
            continue
        for line in source.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def rails(ruby: str) -> str:
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "rails",
         "bundle", "exec", "rails", "runner", ruby],
        cwd=DEPLOY, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise SystemExit(f"rails runner failed:\n{result.stderr[-800:]}")
    return result.stdout


def wipe_chatwoot() -> None:
    out = rails(
        'before = [Conversation.count, Contact.count, Message.count];'
        'Conversation.destroy_all; Contact.destroy_all;'
        'puts "WAS #{before.join(",")}";'
        'puts "NOW #{[Conversation.count, Contact.count, Message.count].join(",")}";'
        'puts "KEPT inboxes=#{Inbox.count} users=#{User.count}"'
    )
    for line in out.splitlines():
        if line.startswith(("WAS", "NOW", "KEPT")):
            print(f"  {line.lower()}")


def clear_mailboxes() -> None:
    result = subprocess.run(
        ["docker", "compose", "restart", "greenmail"],
        cwd=DEPLOY, capture_output=True, text=True, timeout=180,
    )
    print("  mailboxes cleared" if result.returncode == 0 else "  greenmail restart failed")


def reset_stripe(setup_key: str) -> None:
    client = httpx.Client(
        base_url="https://api.stripe.com/v1",
        headers={"Authorization": f"Bearer {setup_key}"},
        timeout=30,
    )
    deleted = 0
    starting_after = None
    while True:
        params: dict[str, object] = {"limit": 100}
        if starting_after:
            params["starting_after"] = starting_after
        page = client.get("/customers", params=params)
        page.raise_for_status()
        body = page.json()
        for customer in body["data"]:
            if (customer.get("email") or "").startswith(TEST_PREFIXES):
                if client.delete(f"/customers/{customer['id']}").status_code == 200:
                    deleted += 1
        if not body.get("has_more"):
            break
        starting_after = body["data"][-1]["id"]
    print(f"  stripe test customers deleted={deleted}")

    created = client.post(
        "/customers", data={"email": DEMO_EMAIL, "source": "tok_visa"}
    )
    created.raise_for_status()
    charge = client.post(
        "/charges",
        data={
            "amount": DEMO_AMOUNT_CENTS,
            "currency": "usd",
            "customer": created.json()["id"],
            "description": "demo bootcamp payment",
        },
    )
    charge.raise_for_status()
    print(
        f"  refundable charge ready: {DEMO_EMAIL} "
        f"${DEMO_AMOUNT_CENTS / 100:.2f} ({charge.json()['id']})"
    )


def main() -> int:
    config = settings()

    print("chatwoot")
    wipe_chatwoot()

    print("\nemail")
    clear_mailboxes()

    print("\nstripe")
    setup_key = config.get("STRIPE_SETUP_KEY", "")
    if setup_key.startswith("sk_test_"):
        reset_stripe(setup_key)
    else:
        print("  skipped: STRIPE_SETUP_KEY is not an sk_test_ key")

    print(
        "\nready. In the browser, use a private window or clear site data for "
        f"localhost;\nthen ask a question, and enter {DEMO_EMAIL} to try a refund."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
