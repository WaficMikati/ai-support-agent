"""Create a mock company's help centre inside Chatwoot.

    uv run python scripts/seed_helpcenter.py

The workshop needs support documentation to answer from, and it must not look
like any real company's policy. So this seeds a portal for a fictional coffee
subscription, Acme Coffee, with the kinds of articles a subscription business
actually publishes.

Everything here is invented. That is the point: nothing in it can be mistaken
for a real refund window or a real price. Swap COMPANY and ARTICLES for real
documentation and nothing else has to change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

COMPANY = "Acme Coffee"
PORTAL_SLUG = "acme-coffee-help"

CATEGORIES = {
    "subscription": "Subscription and deliveries",
    "billing": "Billing and payments",
    "account": "Account and access",
    "coffee": "Our coffee",
}

ARTICLES: list[tuple[str, str, str]] = [
    (
        "subscription",
        "How your subscription works",
        f"""{COMPANY} sends one bag of freshly roasted coffee every month. Your
first bag ships within two working days of signing up, and after that on the
same date each month.

You can switch between the 250g and 500g bag at any time from Settings, under
Subscription. The change applies to your next delivery, not the one already on
its way.""",
    ),
    (
        "subscription",
        "Skipping a delivery",
        """You can skip a single month from Settings, under Subscription. Choose
Skip next delivery. You are not charged for a skipped month and your
subscription picks up again the following month.

Skipping is available until the day before your delivery date. After that the
bag has already been roasted and packed, so the next available skip is the
month after.""",
    ),
    (
        "subscription",
        "Cancelling your subscription",
        """Cancellation is in Settings, under Subscription. It stops the next
payment immediately.

You keep any delivery you have already paid for. So if you cancel the day after
being charged, that month's bag still arrives, and nothing is charged after
that. There is no cancellation fee and no minimum term.""",
    ),
    (
        "subscription",
        "Changing your delivery address",
        """Update your address in Settings, under Delivery. If you change it
before your delivery date the next bag goes to the new address.

If a bag has already shipped we cannot redirect it. Contact support and we will
arrange a replacement to the correct address.""",
    ),
    (
        "subscription",
        "My delivery has not arrived",
        """Deliveries usually arrive within five working days of your delivery
date. Before contacting us, check whether a neighbour has taken it in and look
for a card from the courier.

If it is more than five working days late, contact support with your delivery
date and we will send a replacement. We do not ask you to wait longer than
that.""",
    ),
    (
        "billing",
        "When you are charged",
        """You are charged on the same date each month, the date you first
subscribed. The charge appears as ACME COFFEE on your statement.

If that date does not exist in a given month, for example the 31st, you are
charged on the last day of that month instead.""",
    ),
    (
        "billing",
        "Updating your card",
        """Add or change a card in Settings, under Billing. The new card is used
from your next charge onwards.

If a payment fails we retry once after three days. If it fails again the
subscription pauses rather than cancels, so you can add a working card and pick
up where you left off.""",
    ),
    (
        "billing",
        "Refunds",
        """If something is wrong with a bag you have received, contact support
and we will refund that month. Tell us what was wrong and we do not need the
coffee back.

Refunds go to the card you paid with and usually appear within a few days,
depending on your bank. We do not refund months that were delivered and
enjoyed, but if you cancelled and were charged anyway that is our mistake and
we refund it.""",
    ),
    (
        "billing",
        "Why was I charged after cancelling",
        """If you were charged after cancelling, one of two things happened. If
the charge came on the same day you cancelled, it was already in flight and we
will refund it. If it came later, the cancellation may not have completed.

Either way, contact support with the date of the charge and we will sort it
out.""",
    ),
    (
        "billing",
        "Do you offer gift subscriptions",
        """Yes. Choose Gift a subscription when signing up and pick three, six
or twelve months. Gift subscriptions are paid for up front and do not renew
automatically, so the recipient is never charged.

You can have the first bag delivered on a chosen date if you are giving it for
an occasion.""",
    ),
    (
        "account",
        "Signing in",
        """Sign in with the email address you subscribed with. If you have
forgotten your password use the Reset password link on the sign-in page.

If the reset email does not arrive, check your spam folder first. If it is
still missing, contact support and we will look at the account directly.""",
    ),
    (
        "account",
        "Changing the email on your account",
        """Change your email in Settings, under Account. You will be asked to
confirm the new address before it takes effect, so it needs to be one you can
receive mail at.

Your subscription and delivery history stay with the account.""",
    ),
    (
        "account",
        "Deleting your account",
        """Cancel your subscription first, then choose Delete account in
Settings, under Account. Deletion removes your delivery history and cannot be
undone.

If you only want to stop deliveries, cancel instead of deleting. That way your
address and preferences are still there if you come back.""",
    ),
    (
        "coffee",
        "How fresh is the coffee",
        """Every bag is roasted the week it ships. The roast date is printed on
the bag rather than a best-before date, because coffee is at its best in the
two to four weeks after roasting.

We do not roast in advance and we do not hold stock.""",
    ),
    (
        "coffee",
        "Choosing whole bean or ground",
        """Choose whole bean or ground in Settings, under Subscription. If you
choose ground, tell us your brew method and we grind for it.

If you are not sure, whole bean keeps its flavour longer, but only if you have
a grinder.""",
    ),
    (
        "coffee",
        "Do you offer decaf",
        """Yes. Decaf is available in the same bag sizes and on the same
schedule, and it costs the same. Switch to it in Settings, under Subscription.

Our decaf is decaffeinated with water and carbon dioxide rather than solvents.""",
    ),
]


def settings() -> dict[str, str]:
    values: dict[str, str] = {}
    for source in (ROOT / ".env", ROOT / "deploy" / "admin.local.txt"):
        if not source.exists():
            continue
        for line in source.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def main() -> int:
    config = settings()
    client = httpx.Client(
        base_url=f"{config['CHATWOOT_URL']}/api/v1/accounts/{config['CHATWOOT_ACCOUNT_ID']}",
        headers={"api_access_token": config["CHATWOOT_TOKEN"]},
        timeout=30,
    )
    author_id = int(config.get("author_id", 1))

    existing = client.get("/portals")
    existing.raise_for_status()
    portals = existing.json()
    rows = portals.get("payload", portals) if isinstance(portals, dict) else portals
    portal = next((p for p in rows if p.get("slug") == PORTAL_SLUG), None)

    if portal is None:
        created = client.post(
            "/portals",
            json={
                "name": f"{COMPANY} Help Centre",
                "slug": PORTAL_SLUG,
                "custom_domain": "",
                "homepage_link": "",
            },
        )
        created.raise_for_status()
        portal = created.json()
        print(f"  created portal {PORTAL_SLUG}")
    else:
        print(f"  portal {PORTAL_SLUG} already exists")

    # Nested help centre routes look the portal up by slug, not by id:
    # /portals/1/categories is a 404 while /portals/<slug>/categories is fine.
    portal_ref = PORTAL_SLUG

    # Every help centre response wraps the record in "payload".
    def unwrap(response: httpx.Response):
        body = response.json()
        return body.get("payload", body) if isinstance(body, dict) else body

    def existing_categories() -> dict[str, int]:
        listing = client.get(f"/portals/{portal_ref}/categories")
        listing.raise_for_status()
        return {row["slug"]: row["id"] for row in unwrap(listing)}

    # Re-runnable: reuse categories that are already there rather than failing
    # on a duplicate slug.
    category_ids = existing_categories()
    for slug, name in CATEGORIES.items():
        if slug in category_ids:
            continue
        response = client.post(
            f"/portals/{portal_ref}/categories",
            json={"name": name, "slug": slug, "locale": "en", "description": name},
        )
        if response.status_code in (200, 201):
            category_ids[slug] = unwrap(response)["id"]
        else:
            print(f"  category {slug} failed: {response.status_code} {response.text[:120]}")
    category_ids = {**category_ids, **existing_categories()}
    print(f"  categories: {sorted(set(CATEGORIES) & set(category_ids))}")

    known = {
        row["title"]
        for row in unwrap(client.get(f"/portals/{portal_ref}/articles"))
        if isinstance(row, dict) and row.get("title")
    }

    made = existing = failed = 0
    for category_slug, title, content in ARTICLES:
        if title in known:
            existing += 1
            continue
        response = client.post(
            f"/portals/{portal_ref}/articles",
            json={
                "title": title,
                "content": " ".join(content.split()),
                "category_id": category_ids.get(category_slug),
                "author_id": author_id,
                "status": "published",
                "locale": "en",
            },
        )
        if response.status_code in (200, 201):
            made += 1
        else:
            failed += 1
            if failed == 1:
                print(f"  first failure: {response.status_code} {response.text[:200]}")
    print(f"  articles created={made} already_present={existing} failed={failed}")
    print(f"\n{COMPANY} help centre seeded. Everything in it is invented.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
