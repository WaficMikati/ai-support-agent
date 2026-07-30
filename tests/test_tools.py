"""Tests for the tools the model may call while working an answer out.

The model can look things up before it replies. Two properties matter more than
the mechanics, and both are asserted here rather than assumed:

  * every tool is read-only, so a tool loop can never move money
  * every tool takes no arguments, so the customer whose record is read is fixed
    by the caller and cannot be steered by anything typed into the chat

The second is the one worth being careful about. A lookup that accepted an email
address would let the model pass along whatever the customer wrote, and anybody
could read a stranger's payment history by naming their address.

Because Groq refuses `tools` and `response_format` in one request, this runs in
two phases: gather with tools, then ask for the proposal. The seam between them
is where things can go wrong, so the tests check what actually reaches the second
call.
"""

import json
from datetime import datetime, timezone

import httpx

from agent import (
    TOOL_SPECS,
    Brain,
    Charge,
    Conversation,
    Message,
    Proposal,
    State,
    Turn,
    account_tools,
    describe_charge,
    understand_node,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)

CONVERSATION = (Turn(role="user", content="how much was my last payment?"),)

PROPOSAL = json.dumps(
    {
        "reply": "It was 20.00 on 30 July.",
        "refund_requested": False,
        "clear_request": False,
        "charge_identified": False,
        "hedging": False,
    }
)


def answered(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def called(name: str, arguments: str = "{}", call_id: str = "call_1") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }


def brain_for(responses):
    remaining = list(responses)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=remaining.pop(0))

    brain = Brain("gsk_test", transport=httpx.MockTransport(handler), sleep=lambda _: None)
    return brain, seen


def charge(**overrides) -> Charge:
    return Charge(
        **{
            "id": "ch_1",
            "amount_cents": 2_000,
            "created": NOW,
            "refunded": False,
            "prior_refund_count": 0,
            **overrides,
        }
    )


# ------------------------------------------------------- the safety property


def test_no_tool_can_move_money():
    """The model may read. Whether a refund happens is refund_decision's, in
    code, after the model has finished."""
    for name, spec in TOOL_SPECS.items():
        assert "refund" not in name, f"{name} would let the model spend money"
        assert "cancel" not in name and "delete" not in name


def test_no_tool_takes_arguments():
    """The whole point: the model chooses whether to look, never whose record to
    look at."""
    for name, spec in TOOL_SPECS.items():
        parameters = spec["function"]["parameters"]
        assert parameters.get("properties") == {}, f"{name} accepts input"
        assert not parameters.get("required"), f"{name} requires input"


def test_the_lookup_is_bound_to_the_conversations_own_contact():
    """Not to anything said in the chat."""
    asked_for: list[str] = []

    class Payments:
        def latest_charge(self, email):
            asked_for.append(email)
            return charge()

        def refund(self, charge_id, idempotency_key):
            raise AssertionError("not in this test")

    state = State(
        conversation=Conversation(
            id=1,
            contact_email="real@customer.com",
            # The customer naming somebody else must change nothing.
            messages=(Message(id=1, content="check victim@example.com", incoming=True),),
        ),
        message=Message(id=1, content="check victim@example.com", incoming=True),
        knowledge="guidance",
        inbox=None,
        payments=Payments(),
        understand=lambda *a, **k: None,
        now=NOW,
    )
    tools = account_tools(state)
    tools["get_last_purchase"]()
    assert asked_for == ["real@customer.com"]


def test_nothing_is_offered_when_the_visitor_is_not_identified():
    """An anonymous visitor has no record to read, and offering the tool anyway
    invites an answer built on a failed lookup."""
    state = State(
        conversation=Conversation(id=1, contact_email=None, messages=()),
        message=Message(id=1, content="hi", incoming=True),
        knowledge="guidance",
        inbox=None,
        payments=None,
        understand=lambda *a, **k: None,
        now=NOW,
    )
    assert account_tools(state) == {}


# ------------------------------------------------------------- the two phases


def test_a_tool_result_reaches_the_call_that_writes_the_reply():
    """The gathering phase is worthless if what it found is dropped before the
    model is asked for the proposal."""
    brain, seen = brain_for(
        [called("get_last_purchase"), answered("found it"), answered(PROPOSAL)]
    )
    proposal = brain.understand(
        CONVERSATION,
        "be helpful",
        (),
        {"get_last_purchase": lambda: "Most recent payment: 20.00 on 30 July 2026."},
    )
    assert isinstance(proposal, Proposal)

    final = seen[-1]["messages"]
    tool_results = [m for m in final if m.get("role") == "tool"]
    assert tool_results, "what the tool returned never reached the second call"
    assert "20.00" in tool_results[0]["content"]


def test_the_gathering_phase_is_not_asked_for_json():
    """Offered a strict schema and tools together the model answers immediately
    and never looks anything up. Groq refuses the combination outright."""
    brain, seen = brain_for(
        [called("get_last_purchase"), answered("found it"), answered(PROPOSAL)]
    )
    brain.understand(CONVERSATION, "be helpful", (), {"get_last_purchase": lambda: "ok"})

    gather, final = seen[0], seen[-1]
    assert "tools" in gather and "response_format" not in gather
    assert "response_format" in final and "tools" not in final


def test_no_tools_means_a_single_call():
    """Nothing to look up, nothing to pay for."""
    brain, seen = brain_for([answered(PROPOSAL)])
    brain.understand(CONVERSATION, "be helpful")
    assert len(seen) == 1
    assert "tools" not in seen[0]


def test_the_model_can_answer_without_calling_anything():
    brain, seen = brain_for([answered("just answering"), answered(PROPOSAL)])
    brain.understand(CONVERSATION, "be helpful", (), {"get_last_purchase": lambda: "ok"})
    assert len(seen) == 2, "one gather that called nothing, then the proposal"
    assert not [m for m in seen[-1]["messages"] if m.get("role") == "tool"]


def test_the_loop_is_bounded():
    """A model that keeps asking for the same thing must not loop forever."""
    calls = 0

    def tool():
        nonlocal calls
        calls += 1
        return "same answer again"

    brain, seen = brain_for(
        [called("get_last_purchase", call_id=f"call_{n}") for n in range(3)]
        + [answered(PROPOSAL)]
    )
    brain.understand(CONVERSATION, "be helpful", (), {"get_last_purchase": tool})
    assert calls == Brain.MAX_TOOL_ROUNDS
    assert len(seen) == Brain.MAX_TOOL_ROUNDS + 1, "the proposal is still asked for"


def test_an_unknown_tool_is_reported_rather_than_crashing():
    """A model inventing a name should cost a wasted round, not the whole reply."""
    brain, seen = brain_for(
        [called("read_their_emails"), answered("hm"), answered(PROPOSAL)]
    )
    proposal = brain.understand(
        CONVERSATION, "be helpful", (), {"get_last_purchase": lambda: "ok"}
    )
    assert isinstance(proposal, Proposal)
    result = [m for m in seen[-1]["messages"] if m.get("role") == "tool"][0]
    assert "no tool called" in result["content"]


def test_the_node_offers_the_lookup_to_the_model():
    """The wiring, rather than the model client on its own."""
    offered: list[dict] = []

    class Payments:
        def latest_charge(self, email):
            return charge()

        def refund(self, charge_id, idempotency_key):
            raise AssertionError("not in this test")

    def understand(turns, knowledge, articles=(), tools=None):
        offered.append(dict(tools or {}))
        return Proposal(
            reply="ok",
            refund_requested=False,
            clear_request=False,
            charge_identified=False,
            hedging=False,
        )

    state = State(
        conversation=Conversation(
            id=1,
            contact_email="a@b.com",
            messages=(Message(id=1, content="what did I pay?", incoming=True),),
        ),
        message=Message(id=1, content="what did I pay?", incoming=True),
        knowledge="guidance",
        inbox=None,
        payments=Payments(),
        understand=understand,
        now=NOW,
    )
    understand_node(state)
    assert list(offered[0]) == ["get_last_purchase"]


# --------------------------------------------------------- what it says back


def test_a_payment_is_described_in_plain_words():
    said = describe_charge(charge())
    assert "20.00" in said and "30 July 2026" in said


def test_a_payment_that_stands_says_nothing_about_refunds():
    """Told a payment "has not been refunded" the model repeats it, so somebody
    who asked what they had paid is answered with a refund status they never
    raised. Whether a refund can happen is settled in code regardless."""
    assert "refund" not in describe_charge(charge()).lower()


def test_an_already_refunded_payment_says_so():
    assert "already been refunded" in describe_charge(charge(refunded=True))


def test_other_payments_are_mentioned_because_they_block_a_refund():
    said = describe_charge(charge(sibling_unrefunded_count=2))
    assert "2 other unrefunded" in said


def test_no_payment_is_stated_plainly_rather_than_left_blank():
    assert "no payments on record" in describe_charge(None)


def test_the_charge_id_is_never_shown():
    """Meaningless to a customer, and anything in front of the model can be
    quoted back to them."""
    assert "ch_1" not in describe_charge(charge())
