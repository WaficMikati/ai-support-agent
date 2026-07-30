"""Tests for the Stripe adapter.

This is the code that decides which charge gets refunded, so it is worth
pinning down precisely. Everything runs against canned Stripe JSON.
"""

import httpx

from agent import StripePayments, refund_decision

NOW_TS = 1_785_000_000
CUSTOMER = {"id": "cus_1", "email": "a@b.com"}


def charge_json(
    charge_id,
    amount=2_000,
    refunded=False,
    amount_refunded=0,
    created=NOW_TS,
    disputed=False,
    metadata=None,
    currency="usd",
):
    return {
        "id": charge_id,
        "amount": amount,
        "created": created,
        "refunded": refunded,
        "amount_refunded": amount_refunded,
        "disputed": disputed,
        "metadata": metadata or {},
        "currency": currency,
    }


def payments_for(charges, customers=(CUSTOMER,)):
    """A StripePayments wired to canned responses, plus the requests made."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/v1/customers":
            return httpx.Response(200, json={"data": list(customers)})
        if path == "/v1/charges":
            return httpx.Response(200, json={"data": list(charges)})
        if path == "/v1/refunds":
            return httpx.Response(200, json={"id": "re_1"})
        raise AssertionError(f"unexpected {request.method} {path}")

    return StripePayments("rk_test_x", transport=httpx.MockTransport(handler)), seen


# ------------------------------------------------------------------ lookup


def test_no_customer_means_no_charge():
    payments, _ = payments_for([], customers=())
    assert payments.latest_charge("a@b.com") is None


def test_customer_with_no_charges_means_no_charge():
    payments, _ = payments_for([])
    assert payments.latest_charge("a@b.com") is None


def test_a_single_charge_is_returned():
    payments, _ = payments_for([charge_json("ch_1")])
    charge = payments.latest_charge("a@b.com")
    assert charge is not None
    assert charge.id == "ch_1"
    assert charge.amount_cents == 2_000
    assert not charge.refunded
    assert charge.sibling_unrefunded_count == 0


def test_partial_refund_counts_as_refunded():
    """Stripe leaves `refunded` false when only part of a charge is returned.
    Treating that as untouched would hand back the remaining balance."""
    payments, _ = payments_for(
        [charge_json("ch_1", amount=2_000, refunded=False, amount_refunded=500)]
    )
    charge = payments.latest_charge("a@b.com")
    assert charge is not None
    assert charge.refunded, "a partly refunded charge must not look refundable"
    assert not refund_decision(charge, 0.95).auto_approve


def test_the_refundable_charge_is_chosen_not_merely_the_newest():
    """Newest first, as Stripe returns them: the newest is already refunded,
    so the older untouched one is the only candidate."""
    payments, _ = payments_for(
        [
            charge_json("ch_new", refunded=True, amount_refunded=2_000, created=NOW_TS),
            charge_json("ch_old", created=NOW_TS - 86_400),
        ]
    )
    charge = payments.latest_charge("a@b.com")
    assert charge is not None
    assert charge.id == "ch_old"
    assert charge.prior_refund_count == 1


def test_several_refundable_charges_report_siblings():
    payments, _ = payments_for(
        [charge_json("ch_1"), charge_json("ch_2"), charge_json("ch_3")]
    )
    charge = payments.latest_charge("a@b.com")
    assert charge is not None
    assert charge.id == "ch_1", "newest of the candidates"
    assert charge.sibling_unrefunded_count == 2
    assert not refund_decision(charge, 0.95).auto_approve


def test_everything_already_refunded_reports_refunded_not_missing():
    payments, _ = payments_for(
        [
            charge_json("ch_1", refunded=True, amount_refunded=2_000),
            charge_json("ch_2", refunded=True, amount_refunded=2_000),
        ]
    )
    charge = payments.latest_charge("a@b.com")
    assert charge is not None, "reporting no charge would give a misleading reason"
    assert charge.refunded
    assert charge.prior_refund_count == 1, "the other one, not counting itself"
    decision = refund_decision(charge, 0.95)
    assert not decision.auto_approve
    assert "already been refunded" in decision.reason


def test_amount_and_created_are_mapped():
    payments, _ = payments_for([charge_json("ch_1", amount=12_345, created=NOW_TS)])
    charge = payments.latest_charge("a@b.com")
    assert charge is not None
    assert charge.amount_cents == 12_345
    assert int(charge.created.timestamp()) == NOW_TS
    assert charge.created.tzinfo is not None, "must be timezone aware for age checks"


def test_a_dispute_is_carried_through():
    payments, _ = payments_for([charge_json("ch_1", disputed=True)])
    charge = payments.latest_charge("a@b.com")
    assert charge is not None and charge.disputed


def test_a_charge_is_not_disputed_unless_stripe_says_so():
    payments, _ = payments_for([charge_json("ch_1")])
    charge = payments.latest_charge("a@b.com")
    assert charge is not None and not charge.disputed


# ------------------------------------------------------- the demo backdate


def test_a_demo_backdate_ages_the_payment():
    """Stripe stamps `created` itself and a test clock backdates the customer
    but not their charges, so the age check could not otherwise be shown at all.
    The fiction lives in the data; refund_decision still just compares two
    timestamps."""
    payments, _ = payments_for(
        [charge_json("ch_1", metadata={"demo_backdate_days": "45"})]
    )
    charge = payments.latest_charge("a@b.com")
    assert charge is not None
    assert int(charge.created.timestamp()) == NOW_TS - 45 * 86_400


def test_a_charge_without_the_marker_is_untouched():
    payments, _ = payments_for([charge_json("ch_1")])
    charge = payments.latest_charge("a@b.com")
    assert charge is not None and int(charge.created.timestamp()) == NOW_TS


def test_a_nonsense_backdate_is_ignored_rather_than_crashing():
    payments, _ = payments_for(
        [charge_json("ch_1", metadata={"demo_backdate_days": "soon"})]
    )
    charge = payments.latest_charge("a@b.com")
    assert charge is not None and int(charge.created.timestamp()) == NOW_TS


# ----------------------------------------------------------------- refunds


def test_refund_posts_the_charge_id_and_returns_the_refund_id():
    payments, seen = payments_for([charge_json("ch_1")])
    assert payments.refund("ch_1", "refund-conv1-msg2") == "re_1"
    posts = [r for r in seen if r.method == "POST"]
    assert len(posts) == 1
    assert posts[0].url.path == "/v1/refunds"
    assert b"charge=ch_1" in posts[0].content


def test_refund_sends_the_idempotency_key_as_a_header():
    """Without this header a retry issues a second refund."""
    payments, seen = payments_for([charge_json("ch_1")])
    payments.refund("ch_1", "refund-conv7-msg42")
    post = next(r for r in seen if r.method == "POST")
    assert post.headers["Idempotency-Key"] == "refund-conv7-msg42"


def test_requests_are_authorised():
    payments, seen = payments_for([charge_json("ch_1")])
    payments.latest_charge("a@b.com")
    assert all(r.headers["Authorization"] == "Bearer rk_test_x" for r in seen)


def test_lookup_never_writes():
    payments, seen = payments_for([charge_json("ch_1")])
    payments.latest_charge("a@b.com")
    assert [r.method for r in seen] == ["GET", "GET"], "reading must not POST"


# ----------------------------------------------------------------- currency


def test_the_currency_is_carried_through():
    payments, _ = payments_for([charge_json("ch_1", currency="eur")])
    charge = payments.latest_charge("a@b.com")
    assert charge is not None and charge.currency == "eur"


def test_a_charge_without_a_currency_is_assumed_to_be_dollars():
    """Rather than crashing on a field Stripe always sends but a fake might not."""
    payments, _ = payments_for([{"id": "ch_1", "amount": 2_000, "created": NOW_TS,
                                 "refunded": False, "amount_refunded": 0}])
    charge = payments.latest_charge("a@b.com")
    assert charge is not None and charge.currency == "usd"
