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


def new_customer() -> dict[str, str]:
    """Provision a brand new customer for a fresh conversation.

    Deliberately not by deleting anything. Two constraints force this shape:

      * Chatwoot allows one contact per email per account, so a new visitor
        cannot reuse the previous one's address
      * deleting contacts leaves the browser holding a token that points at
        something gone, and setUser against a stale token fails silently, so
        the next visitor ends up anonymous and refunds stop matching

    So each reset mints a unique address, gives it a refundable charge in
    Stripe, and the page identifies itself as that person.
    """
    config = reset_demo.settings()
    stamp = str(int(time.time()))
    email = f"demo+{stamp}@example.com"

    setup_key = config.get("STRIPE_SETUP_KEY", "")
    if setup_key.startswith("sk_test_"):
        client = httpx.Client(
            base_url="https://api.stripe.com/v1",
            headers={"Authorization": f"Bearer {setup_key}"},
            timeout=30,
        )
        customer = client.post(
            "/customers", data={"email": email, "source": "tok_visa"}
        )
        customer.raise_for_status()
        charge = client.post(
            "/charges",
            data={
                "amount": reset_demo.DEMO_AMOUNT_CENTS,
                "currency": "usd",
                "customer": customer.json()["id"],
                "description": "demo subscription payment",
            },
        )
        charge.raise_for_status()

    return {"email": email, "identifier": f"demo-{stamp}", "name": "Demo Customer"}


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "demo"), **kwargs)

    def do_POST(self) -> None:  # noqa: N802  (http.server naming)
        if self.path.rstrip("/") != "/reset":
            self.send_error(404)
            return

        try:
            body = {"ok": True, **new_customer()}
        except Exception as error:  # the page should show what went wrong
            body = {"ok": False, "error": f"{type(error).__name__}: {error}"}

        payload = json.dumps(body).encode()
        self.send_response(200 if body["ok"] else 500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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
