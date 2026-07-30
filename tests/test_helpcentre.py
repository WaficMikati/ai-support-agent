"""Tests for finding the right help centre article.

Chatwoot's own ?query= is a phrase match, not a search: it returns nothing for
"How do I cancel my subscription?" even with an article called "Cancelling your
subscription". So selection happens here, and these tests pin down that the
obvious customer phrasings actually find the obvious article.
"""

import httpx
import pytest

from agent import Article, ChatwootHelpCentre, keywords

ARTICLES = [
    ("Cancelling your subscription",
     "Cancellation is in Settings, under Subscription. It stops the next payment."),
    ("Skipping a delivery",
     "You can skip a single month from Settings. You are not charged for a skipped month."),
    ("Refunds",
     "If something is wrong with a bag you received we refund that month."),
    ("Do you offer decaf",
     "Decaf is available in the same bag sizes and costs the same."),
    ("How fresh is the coffee",
     "Every bag is roasted the week it ships, and the roast date is on the bag."),
    ("Updating your card",
     "Add or change a card in Settings, under Billing."),
]


def help_centre(articles=ARTICLES, ttl_seconds=300, clock=None):
    calls: list[httpx.Request] = []
    ticks = iter(clock or [0.0] * 50)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "payload": [
                    {"title": title, "content": content} for title, content in articles
                ]
            },
        )

    centre = ChatwootHelpCentre(
        "http://chatwoot.test", "1", "token", "acme-coffee-help",
        ttl_seconds=ttl_seconds,
        transport=httpx.MockTransport(handler),
        now=lambda: next(ticks),
    )
    return centre, calls


def titles(articles: list[Article]) -> list[str]:
    return [article.title for article in articles]


# ------------------------------------------------------------------ keywords


def test_common_words_are_ignored():
    assert keywords("How do I cancel my subscription?") == {"cancel", "subscription"}


def test_punctuation_and_case_do_not_matter():
    assert keywords("DECAF?!") == {"decaf"}


def test_a_question_of_only_common_words_has_no_keywords():
    assert keywords("what is it about") == set()


# ----------------------------------------------------------------- selection


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How do I cancel my subscription?", "Cancelling your subscription"),
        ("do you have decaf", "Do you offer decaf"),
        ("can I skip a month", "Skipping a delivery"),
        ("I want a refund for a bad bag", "Refunds"),
        ("how do I change my card", "Updating your card"),
        ("how fresh is it roasted", "How fresh is the coffee"),
    ],
)
def test_the_obvious_phrasing_finds_the_obvious_article(question, expected):
    centre, _ = help_centre()
    found = titles(centre.relevant(question))
    assert found, f"nothing matched {question!r}"
    assert found[0] == expected, f"{question!r} matched {found}"


def test_a_title_match_outranks_a_body_match():
    """"Settings" appears in three bodies; the question is about cancelling."""
    centre, _ = help_centre()
    assert titles(centre.relevant("cancel"))[0] == "Cancelling your subscription"


def test_nothing_relevant_returns_nothing():
    centre, _ = help_centre()
    assert centre.relevant("what is the capital of France") == []


def test_a_question_with_no_keywords_returns_nothing_without_fetching():
    centre, calls = help_centre()
    assert centre.relevant("what is it about") == []
    assert calls == [], "should not have called the API at all"


def test_the_limit_is_respected():
    centre, _ = help_centre()
    assert len(centre.relevant("subscription settings card month bag", limit=2)) == 2


# --------------------------------------------------------------------- cache


def test_articles_are_fetched_once_within_the_ttl():
    centre, calls = help_centre()
    centre.relevant("cancel")
    centre.relevant("decaf")
    centre.relevant("refund")
    assert len(calls) == 1, "the corpus should be cached, not refetched per question"


def test_the_cache_expires_so_edits_are_picked_up():
    """Editing an article in Chatwoot must take effect without a restart."""
    centre, calls = help_centre(ttl_seconds=60, clock=[0.0, 120.0, 120.0, 120.0])
    centre.relevant("cancel")
    centre.relevant("cancel")
    assert len(calls) == 2, "should have refetched after the ttl passed"


# ------------------------------------------------------------------ mapping


def test_articles_are_mapped_from_the_payload():
    centre, _ = help_centre()
    found = centre.relevant("decaf")
    assert found[0] == Article(
        title="Do you offer decaf",
        content="Decaf is available in the same bag sizes and costs the same.",
    )


def test_empty_rows_are_dropped():
    centre, _ = help_centre(articles=[("", ""), ("Refunds", "we refund that month")])
    assert len(centre.articles()) == 1


def test_requests_carry_the_access_token():
    centre, calls = help_centre()
    centre.articles()
    assert calls[0].headers["api_access_token"] == "token"
    assert "/portals/acme-coffee-help/articles" in str(calls[0].url)
