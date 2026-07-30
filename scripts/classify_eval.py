"""Check the classifier against labelled messages.

    uv run python scripts/classify_eval.py            # every case
    uv run python scripts/classify_eval.py --quick    # the six that matter most
    uv run python scripts/classify_eval.py --bare     # guidance only, as it used to

The classifier decides whether code is allowed to move money, so its rubric
needs a regression test. Unit tests cannot do this: the behaviour lives in a
prompt and a model, not in a branch.

The distinction that matters most here is between asking *for* a refund and
asking *about* refunds. "How do I get a refund?" is a question, and answering it
means quoting the refund article. An earlier rubric read it as a request and
refunded twenty dollars for it.

It sends what the agent sends, which it did not always do. Given only the
guidance, with no help centre and no tools, it was testing an arrangement the
agent is never in, and the difference is not cosmetic: that same "how do i get a
refund?" reads as a request with nothing else in the prompt and as a question
once the Refunds article is beside it. So it reported failures that could not be
reproduced in the running agent, and could not have caught anything the articles
or the lookup cause. --bare keeps the old behaviour for comparing the two.

A full run is not free. Every case now costs two or three model calls carrying
the guidance, the matching articles and the tool definitions, which is a real
slice of a day's tokens on a free tier. --quick is for when the answer is wanted
before a demo rather than after one.

Failures are reported by direction, because they are not equally bad. Answering
a refund request with a reply is an annoyance. Refunding somebody who only asked
a question is a real problem.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import (  # noqa: E402
    MIN_CONFIDENCE,
    Brain,
    ChatwootHelpCentre,
    Turn,
    env_value,
    load_env_file,
)

# (message, expected intent). Keep adding real phrasings as they turn up.
CASES: list[tuple[str, str]] = [
    # Actual requests.
    ("I want my money back", "refund"),
    ("please refund my last payment", "refund"),
    ("refund me", "refund"),
    ("this bag was stale, I want a refund", "refund"),
    ("I was charged twice, please send one back", "refund"),
    ("my order arrived damaged, I want my money back", "refund"),
    ("i need a refund", "refund"),
    ("Can you refund the payment from last week?", "refund"),
    ("I'd like a refund for this month please", "refund"),
    # Questions about refunds. These must not move money.
    ("how do i get a refund?", "support"),
    ("what is your refund policy?", "support"),
    ("do you offer refunds?", "support"),
    ("can I get a refund if I cancel?", "support"),
    ("are refunds possible after a month?", "support"),
    ("how long do refunds take?", "support"),
    ("if my coffee arrives stale can I get my money back?", "support"),
    ("if I cancel today do I get this month back?", "support"),
    ("would I be refunded if the delivery never turns up?", "support"),
    # Ordinary support.
    ("can I skip a month?", "support"),
    ("do you have decaf?", "support"),
    ("my delivery has not arrived", "support"),
    ("I want to cancel my subscription", "support"),
    ("how do I change my card?", "support"),
    ("I can't log in", "support"),
    ("I need help with a purchase", "support"),
]


KNOWLEDGE = (ROOT / "knowledge.md").read_text()
PAUSE_SECONDS = 8

# The ones that have actually gone wrong, for a run before a demo. Three of each
# direction, including the two phrasings that read as requests when they are not.
QUICK = (
    "I want my money back",
    "i need a refund",
    "please refund my last payment",
    "how do i get a refund?",
    "if my coffee arrives stale can I get my money back?",
    "can I skip a month?",
)

# A payment the agent can look up, so the lookup answers the way it would in a
# real conversation rather than reporting an empty account. Fixed rather than
# fetched: this is testing the classifier, and a Stripe round trip per case
# would make a slow script slower without changing what is being measured.
DEMO_CHARGE = "Most recent payment: 20.00 on 30 July 2026."


def main() -> int:
    load_env_file(ROOT / ".env")
    bare = "--bare" in sys.argv
    cases = (
        [case for case in CASES if case[0] in QUICK]
        if "--quick" in sys.argv
        else CASES
    )

    brain = Brain(
        env_value("MODEL_API_KEY", "OPENROUTER_API_KEY"),
        model=env_value("MODEL_NAME", "OPENROUTER_MODEL", default=Brain.DEFAULT_MODEL),
        base_url=env_value(
            "MODEL_BASE_URL", "OPENROUTER_BASE_URL", default=Brain.DEFAULT_BASE_URL
        ),
    )

    # What the agent has beside it on every message. Without these the script
    # measures something the agent never does.
    portal = env_value("HELP_CENTRE_PORTAL")
    help_centre = (
        ChatwootHelpCentre(
            base_url=env_value("CHATWOOT_URL"),
            account_id=env_value("CHATWOOT_ACCOUNT_ID", default="1"),
            token=env_value("CHATWOOT_TOKEN"),
            portal_slug=portal,
        )
        if portal and not bare
        else None
    )
    tools = {} if bare else {"get_last_purchase": lambda: DEMO_CHARGE}

    print(f"model: {brain._model}")
    print(
        "sending: guidance"
        + ("" if help_centre is None else " + matching articles")
        + ("" if not tools else " + the lookup tool")
        + f"  ({len(cases)} cases)\n"
    )

    dangerous: list[str] = []
    annoying: list[str] = []
    low_confidence: list[str] = []

    for index, (message, expected) in enumerate(cases):
        # Paced deliberately. Each call carries the guidance and any matching
        # documentation, so a free tier limited by tokens per minute rather than
        # requests runs out well before this list does.
        if index:
            time.sleep(PAUSE_SECONDS)
        turns = (Turn(role="user", content=message),)
        articles = help_centre.relevant(message) if help_centre else ()
        result = brain.understand(turns, KNOWLEDGE, articles, tools)
        got = "refund" if result.refund_requested else "support"
        ok = got == expected
        if not ok and got == "refund":
            dangerous.append(message)
        elif not ok:
            annoying.append(message)
        # A correct refund below the threshold gets held for a human, which is
        # safe but means the demo shows a hand-off instead of a refund.
        if ok and expected == "refund" and result.confidence < MIN_CONFIDENCE:
            low_confidence.append(message)

        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {got:<8} @{result.confidence:.2f}  {message!r}")

    total = len(cases)
    wrong = len(dangerous) + len(annoying)
    print(f"\n{total - wrong}/{total} correct")

    if dangerous:
        print(f"\nWOULD HAVE MOVED MONEY on a question ({len(dangerous)}):")
        for message in dangerous:
            print(f"  {message!r}")
    if annoying:
        print(f"\nMissed a real request, would reply instead ({len(annoying)}):")
        for message in annoying:
            print(f"  {message!r}")
    if low_confidence:
        print(f"\nCorrect but under the {MIN_CONFIDENCE} threshold, so held for a human:")
        for message in low_confidence:
            print(f"  {message!r}")

    # Only the dangerous direction fails the run.
    return 1 if dangerous else 0


if __name__ == "__main__":
    raise SystemExit(main())
