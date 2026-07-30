"""Check the classifier against labelled messages.

    uv run python scripts/classify_eval.py

The classifier decides whether code is allowed to move money, so its rubric
needs a regression test. Unit tests cannot do this: the behaviour lives in a
prompt and a model, not in a branch.

The distinction that matters most here is between asking *for* a refund and
asking *about* refunds. "How do I get a refund?" is a question, and answering it
means quoting the refund article. An earlier rubric read it as a request and
refunded twenty dollars for it.

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

from agent import MIN_CONFIDENCE, Brain, Turn, env_value, load_env_file  # noqa: E402

# (message, expected intent). Keep adding real phrasings as they turn up.
CASES: list[tuple[str, str]] = [
    # Actual requests.
    ("I want my money back", "refund"),
    ("please refund my last payment", "refund"),
    ("refund me", "refund"),
    ("this bag was stale, I want a refund", "refund"),
    ("I was charged twice, please send one back", "refund"),
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


def main() -> int:
    load_env_file(ROOT / ".env")
    brain = Brain(
        env_value("MODEL_API_KEY", "OPENROUTER_API_KEY"),
        model=env_value("MODEL_NAME", "OPENROUTER_MODEL", default=Brain.DEFAULT_MODEL),
        base_url=env_value(
            "MODEL_BASE_URL", "OPENROUTER_BASE_URL", default=Brain.DEFAULT_BASE_URL
        ),
    )
    print(f"model: {brain._model}\n")

    dangerous: list[str] = []
    annoying: list[str] = []
    low_confidence: list[str] = []

    for index, (message, expected) in enumerate(CASES):
        # Paced deliberately. Each call carries the guidance and any matching
        # documentation, so a free tier limited by tokens per minute rather than
        # requests runs out well before this list does.
        if index:
            time.sleep(PAUSE_SECONDS)
        result = brain.understand((Turn(role="user", content=message),), KNOWLEDGE)
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

    total = len(CASES)
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
