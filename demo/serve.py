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
import re
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import reset_demo  # noqa: E402

PORT = 8080


def release_contact(config: dict[str, str], email: str) -> None:
    """Delete this person's Chatwoot contact so the identity can be claimed again.

    A reset clears the widget's cookie, so the browser returns as a brand new
    contact and then tries to say who it is. Chatwoot allows one contact per
    identifier and per email, and the old contact still holds both, so setUser
    fails silently: the visitor stays anonymous, refunds stop matching, and
    nothing anywhere says why.

    This never came up while every reset invented a fresh address. It appears
    the moment the address is one somebody typed and expects to keep.
    """
    if not email:
        return
    client = httpx.Client(
        base_url=f"{config['CHATWOOT_URL']}/api/v1/accounts/{config['CHATWOOT_ACCOUNT_ID']}",
        headers={"api_access_token": config["CHATWOOT_TOKEN"]},
        timeout=20,
    )
    found = client.get("/contacts/search", params={"q": email})
    if not found.is_success:
        return
    for contact in found.json().get("payload", []):
        if (contact.get("email") or "").lower() == email.lower():
            client.delete(f"/contacts/{contact['id']}")


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

# Deliberately forgiving. This is checking that somebody typed an address
# rather than their name into the wrong box, not policing what an address may
# look like, and the strict-looking patterns reject real ones.
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

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
    email: str = "",
    name: str = "",
) -> dict[str, str]:
    """Build the payment history for one person.

    The address used to be invented here, one per reset. It is now whatever they
    typed on the page, because being handed an identity you never gave is
    exactly what made the agent look like it was reading minds.

    Registering the same address twice replaces what was there rather than
    adding to it. Stripe is happy to hold two customers with one email, and
    since a refund is matched by email, a leftover from an earlier run would be
    picked up as often as the new one. The old customer is deleted first, which
    is why the scenario selectors can be changed and tried again.
    """
    config = reset_demo.settings()
    stamp = str(int(time.time()))
    email = email.strip() or f"demo+{stamp}@example.com"
    name = name.strip() or "Demo Customer"
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
        # Clear out anything registered under this address before, so the
        # history is the one just chosen and not that plus the last attempt.
        existing = client.get("/customers", params={"email": email, "limit": 100})
        existing.raise_for_status()
        for previous in existing.json().get("data", []):
            client.delete(f"/customers/{previous['id']}")

        source = DISPUTE_SOURCE if history == "disputed" else "tok_visa"
        customer = client.post(
            "/customers", data={"email": email, "name": name, "source": source}
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
        # Chatwoot allows one contact per identifier, so this has to be the
        # person rather than the visit: registering again must reach the same
        # contact, not make a second one that cannot claim the address.
        "identifier": email,
        "name": name,
        "history": history,
        "amount_cents": str(amount),
        "backdate_days": str(backdate_days),
    }


def page_settings(host: str = "") -> dict[str, str]:
    """The two values the page needs, from the same config as everything else.

    They used to be typed into the HTML. That is not a leak, since a widget
    token appears in the page source of every site running Chatwoot, but it is
    configuration hardcoded in a second place: anybody else running this got a
    dead widget until they hand-edited the file.
    """
    config = reset_demo.settings()
    # The widget's script, its iframe and its websocket are all fetched by the
    # visitor's browser, so a page served over a tunnel cannot tell them
    # Chatwoot is on localhost: that is their machine, and the bubble never
    # appears. PUBLIC_CHATWOOT_URL is for them.
    #
    # Which is chosen by how this page was reached, not by whether the setting
    # exists. Applying it to everybody meant a stale or stopped tunnel broke
    # the demo on the very machine running it, with a name-resolution error in
    # a console nobody had open. The agent is unaffected either way; it talks
    # to CHATWOOT_URL directly.
    local = host.split(":")[0] in ("localhost", "127.0.0.1", "[::1]", "")
    return {
        "__CHATWOOT_URL__": config.get("CHATWOOT_URL", "http://localhost:3000")
        if local
        else (
            config.get("PUBLIC_CHATWOOT_URL")
            or config.get("CHATWOOT_URL", "http://localhost:3000")
        ),
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
        values = page_settings(self.headers.get("Host", ""))
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
        if route not in ("/reset", "/register"):
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            asked = json.loads(self.rfile.read(length) or "{}") if length else {}
            # Only ever this visitor's own conversations, identified by the
            # address they were given last time. A first arrival has none.
            mine = str(asked.get("email") or "").strip()
            wanted = str(asked.get("register_email") or "").strip()
            if route == "/register" and not EMAIL.match(wanted):
                raise ValueError(f"{wanted!r} does not look like an email address")
            given = " ".join(
                part for part in (
                    str(asked.get("first_name") or "").strip(),
                    str(asked.get("last_name") or "").strip(),
                ) if part
            )
            if route == "/register" and not given:
                raise ValueError("a first and last name are needed")

            cleared = 0
            if route == "/reset" and mine:
                cleared = clear_queue(reset_demo.settings(), email=mine)

            # Let go of any contact already holding these details, so the browser
            # that comes back after the reload can claim them. Chatwoot keeps one
            # contact per email and per identifier, and setUser against a taken
            # one fails without saying so: the visitor stays nameless, the agent
            # cannot look anything up, and the only symptom is being asked for an
            # address it then refuses to accept.
            #
            # Registering matters as much as resetting here. Somebody arriving in
            # a fresh browser and typing an address they used earlier takes the
            # register path, and the contact from that earlier visit is still
            # holding it.
            for address in {mine, wanted} - {""}:
                release_contact(reset_demo.settings(), address)
            body = {
                "ok": True,
                "cleared": cleared,
                **new_customer(
                    amount_cents=int(asked.get("amount_cents") or 0),
                    history=str(asked.get("history") or "clean"),
                    backdate_days=int(asked.get("backdate_days") or 0),
                    amount_cents_2=int(asked.get("amount_cents_2") or 0),
                    backdate_days_2=int(asked.get("backdate_days_2") or 0),
                    email=wanted or mine,
                    name=given,
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
