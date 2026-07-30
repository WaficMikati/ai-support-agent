"""Serve the demo page, and give it a working reset button.

    uv run python demo/serve.py          # then open http://localhost:8080

A static server is not enough, because resetting the chat needs Stripe. Four
things were tried before this shape:

  * $chatwoot.reset() leaves the visitor token byte for byte identical
  * deleting the cookie does nothing, because the chat runs in an iframe on its
    own origin and keeps its own copy
  * rotating the identifier with setUser does give a new conversation, but the
    new contact cannot reuse the previous email, since Chatwoot allows one
    contact per email per account, and without an email the refund path has
    nothing to match against Stripe
  * deleting the contacts frees the email, but then the browser holds a token
    pointing at something gone, setUser fails silently, and the next visitor is
    anonymous

What works is minting a whole new customer: a unique address, a refundable
charge for it in Stripe, and the page identifying itself as that person. The
Stripe key stays in this process rather than being handed to the browser.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import reset_demo  # noqa: E402

PORT = 8080


def clear_queue(config: dict[str, str], email: str | None = None) -> int:
    """Resolve open conversations, and report how many.

    Scoped to one contact when an email is given, and that is what makes the
    page safe to hand round. Clearing the whole account is right for a single
    presenter and wrong for a room: everybody shares one Chatwoot, so an
    unscoped reset closes strangers' conversations mid-sentence.

    `scripts/reset_demo.py` still clears everything, for tidying up afterwards.

    Resolved rather than deleted: the history stays visible in the Chatwoot
    dashboard, which is worth showing, and deleting contacts would be slower
    for no gain. What matters is that the queue is empty, because every open
    conversation is one the agent reconsiders on each poll.

    Done over the API rather than through rails runner, which takes several
    seconds to boot and would make the button feel broken.
    """
    client = httpx.Client(
        base_url=f"{config['CHATWOOT_URL']}/api/v1/accounts/{config['CHATWOOT_ACCOUNT_ID']}",
        headers={"api_access_token": config["CHATWOOT_TOKEN"]},
        timeout=20,
    )
    listing = client.get("/conversations", params={"status": "open"})
    listing.raise_for_status()
    body = listing.json()
    rows = body.get("data", body).get("payload", [])

    cleared = 0
    for row in rows:
        if email:
            sender = (row.get("meta") or {}).get("sender") or {}
            if (sender.get("email") or "").lower() != email.lower():
                continue
        response = client.post(
            f"/conversations/{row['id']}/toggle_status", json={"status": "resolved"}
        )
        if response.status_code < 300:
            cleared += 1
    return cleared


# Each of these builds an account that fails a different check in
# refund_decision, so the policy can be demonstrated rather than described.
#
# There is one condition that cannot be set up this way: MAX_CHARGE_AGE_DAYS.
# Stripe stamps `created` itself and will not accept one, so a charge is always
# from today and "too old" cannot be reached with a real test charge.
HISTORIES = ("clean", "refunded", "disputed", "prior", "multiple", "none")

# Stripe's own test token for a charge the cardholder immediately disputes.
# A real card number cannot be posted here: /v1/tokens answers 402 unless the
# card was tokenised in a browser.
DISPUTE_SOURCE = "tok_createDispute"


def new_customer(
    amount_cents: int = 0,
    history: str = "clean",
    backdate_days: int = 0,
    amount_cents_2: int = 0,
    backdate_days_2: int = 0,
) -> dict[str, str]:
    """Provision a brand new customer for a fresh conversation.

    Deliberately not by deleting anything. Two constraints force this shape:

      * Chatwoot allows one contact per email per account, so a new visitor
        cannot reuse the previous one's address
      * deleting contacts leaves the browser holding a token that points at
        something gone, and setUser against a stale token fails silently, so
        the next visitor ends up anonymous and refunds stop matching

    So each reset mints a unique address, gives it whatever payment history was
    asked for, and the page identifies itself as that person.
    """
    config = reset_demo.settings()
    stamp = str(int(time.time()))
    email = f"demo+{stamp}@example.com"
    amount = amount_cents or reset_demo.DEMO_AMOUNT_CENTS
    if history not in HISTORIES:
        history = "clean"

    setup_key = config.get("STRIPE_SETUP_KEY", "")
    if setup_key.startswith("sk_test_"):
        client = httpx.Client(
            base_url="https://api.stripe.com/v1",
            headers={"Authorization": f"Bearer {setup_key}"},
            timeout=30,
        )
        source = DISPUTE_SOURCE if history == "disputed" else "tok_visa"
        customer = client.post(
            "/customers", data={"email": email, "source": source}
        )
        customer.raise_for_status()
        customer_id = customer.json()["id"]

        def charge_for(cents: int, description: str, aged: int = -1) -> str:
            days = backdate_days if aged < 0 else aged
            data = {
                "amount": cents,
                "currency": "usd",
                "customer": customer_id,
                "description": description,
            }
            if days:
                # Stripe stamps `created` itself and a test clock backdates the
                # customer but not their charges, so the age is carried in
                # metadata and applied when the charge is read back.
                data["metadata[demo_backdate_days]"] = str(days)
            made = client.post("/charges", data=data)
            made.raise_for_status()
            return made.json()["id"]

        def refund(charge_id: str) -> None:
            done = client.post("/refunds", data={"charge": charge_id})
            done.raise_for_status()

        if history == "none":
            pass  # A customer with nothing on file.
        elif history == "disputed":
            # Paid on a card that disputes it. The charge comes back from the
            # create call with disputed still false; it is true by the time
            # anything reads it again, which is all the agent ever does.
            charge_for(amount, "demo subscription payment")
        elif history == "refunded":
            refund(charge_for(amount, "demo subscription payment"))
        elif history == "prior":
            # An older payment already refunded, and a fresh one to ask about.
            refund(charge_for(amount, "demo subscription payment (earlier)"))
            charge_for(amount, "demo subscription payment")
        elif history == "multiple":
            # Two candidates, so the policy refuses to guess which is meant.
            # They get their own amount and date: identical payments made the
            # ambiguity abstract, since there was nothing to tell them apart.
            #
            # Created oldest first, because the backdating is metadata Stripe
            # knows nothing about. Stripe orders charges by when they were
            # really made, and the adapter takes the newest, so creating them
            # out of order would hand the agent the one the page calls older.
            second = amount_cents_2 or amount
            pair = sorted(
                [(backdate_days, amount), (backdate_days_2, second)],
                key=lambda made: -made[0],
            )
            for index, (aged, cents) in enumerate(pair, start=1):
                charge_for(cents, f"demo subscription payment ({index})", aged=aged)
        else:
            charge_for(amount, "demo subscription payment")

    return {
        "email": email,
        "identifier": f"demo-{stamp}",
        "name": "Demo Customer",
        "history": history,
        "amount_cents": str(amount),
        "backdate_days": str(backdate_days),
    }


def page_settings() -> dict[str, str]:
    """The two values the page needs, from the same config as everything else.

    They used to be typed into the HTML. That is not a leak, since a widget
    token appears in the page source of every site running Chatwoot, but it is
    configuration hardcoded in a second place: anybody else running this got a
    dead widget until they hand-edited the file.
    """
    config = reset_demo.settings()
    return {
        "__CHATWOOT_URL__": config.get("CHATWOOT_URL", "http://localhost:3000"),
        "__WIDGET_TOKEN__": config.get(
            "CHATWOOT_WIDGET_TOKEN", config.get("widget_token", "")
        ),
    }


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "demo"), **kwargs)

    def do_GET(self) -> None:  # noqa: N802  (http.server naming)
        if self.path.rstrip("/") not in ("", "/index.html"):
            super().do_GET()
            return

        page = (ROOT / "demo" / "index.html").read_text()
        values = page_settings()
        if not values["__WIDGET_TOKEN__"]:
            # Say so on the page. A blank token gives a widget that silently
            # never appears, which is a miserable thing to debug.
            page = page.replace(
                "</h1>",
                "</h1><p style='color:#b00'>No widget token found. Set "
                "CHATWOOT_WIDGET_TOKEN in .env, or widget_token in "
                "deploy/admin.local.txt.</p>",
            )
        for placeholder, value in values.items():
            page = page.replace(placeholder, value)

        body = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802  (http.server naming)
        route = self.path.rstrip("/")
        # /visitor provisions somebody who has never been here, /reset replaces
        # somebody who has. The difference is only whether there is an existing
        # conversation to close first, but it matters on a shared instance: a
        # first arrival must not clear anything, because everything open belongs
        # to somebody else.
        if route not in ("/reset", "/visitor"):
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            asked = json.loads(self.rfile.read(length) or "{}") if length else {}
            # Only ever this visitor's own conversations, identified by the
            # address they were given last time. A first arrival has none.
            mine = str(asked.get("email") or "").strip()
            cleared = (
                clear_queue(reset_demo.settings(), email=mine)
                if route == "/reset" and mine
                else 0
            )
            body = {
                "ok": True,
                "cleared": cleared,
                **new_customer(
                    amount_cents=int(asked.get("amount_cents") or 0),
                    history=str(asked.get("history") or "clean"),
                    backdate_days=int(asked.get("backdate_days") or 0),
                    amount_cents_2=int(asked.get("amount_cents_2") or 0),
                    backdate_days_2=int(asked.get("backdate_days_2") or 0),
                ),
            }
        except Exception as error:  # the page should show what went wrong
            body = {"ok": False, "error": f"{type(error).__name__}: {error}"}

        payload = json.dumps(body).encode()
        self.send_response(200 if body["ok"] else 500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if body["ok"] and route == "/reset":
            # Delete the widget's visitor token from here rather than from the
            # page. This is the only thing that actually starts a new
            # conversation: setUser creates the new contact server-side but does
            # not re-issue this cookie, and the cookie is what binds the browser
            # to a conversation, so the old thread kept coming back. The widget
            # mints a fresh token on load when it finds no cookie.
            #
            # Cookies ignore the port, so clearing it for host "localhost" from
            # :8080 also clears the one the widget set from :3000.
            #
            # Only on /reset. A first arrival is already holding a token the
            # widget minted moments ago, and clearing it would orphan the
            # contact it belongs to before setUser has claimed it.
            self.send_header(
                "Set-Cookie",
                "cw_conversation=; Max-Age=0; Path=/; SameSite=Lax",
            )
        self.end_headers()
        self.wfile.write(payload)

    def end_headers(self) -> None:
        # Never let the browser cache the page. Editing the demo and having a tab
        # keep running the previous JavaScript is very hard to diagnose from the
        # outside: the reset button still calls the server and still succeeds, so
        # the logs look perfectly healthy while the page behaves like an older
        # version of itself.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Quiet about page assets; the reset is the only interesting request.
        if args and "reset" in str(args[0]):
            super().log_message(format, *args)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), DemoHandler)
    print(f"demo page on http://localhost:{PORT}  (POST /reset clears the demo)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
