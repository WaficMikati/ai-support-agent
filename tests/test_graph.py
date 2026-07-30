"""Tests for the graph itself.

The flow is a dict of nodes and a dict of edges rather than nested ifs, which
means the shape can be checked directly: every node has a way out, every route
leads somewhere real, and the one edge that matters is chosen by the policy
rather than by the model.

Nodes are `state -> state`, so each can be run on its own with fakes.
"""

from datetime import datetime, timezone

import pytest

from agent import (
    EDGES,
    NODES,
    Charge,
    Conversation,
    Message,
    Proposal,
    State,
    answer_node,
    execute_refund_node,
    hold_node,
    refund_node,
    run_graph,
    understand_node,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


class FakeInbox:
    def __init__(self):
        self.replies, self.notes, self.resolved, self.recorded = [], [], [], []

    def open_conversations(self):
        return []

    def send_reply(self, conversation_id, content):
        self.replies.append((conversation_id, content))

    def add_private_note(self, conversation_id, content):
        self.notes.append((conversation_id, content))

    def resolve(self, conversation_id):
        self.resolved.append(conversation_id)

    def record_handled(self, conversation_id, message_id):
        self.recorded.append((conversation_id, message_id))


class FakePayments:
    def __init__(self, charge=None):
        self.charge, self.refunded = charge, []

    def latest_charge(self, email):
        return self.charge

    def charges_for(self, email):
        return [self.charge] if self.charge else []

    def refund(self, charge_id, idempotency_key):
        self.refunded.append((charge_id, idempotency_key))
        return "re_1"


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


def proposal(refund=False, reply="here you go", clear=True, named=True, hedging=False):
    return Proposal(
        reply=reply,
        refund_requested=refund,
        clear_request=clear,
        charge_identified=named,
        hedging=hedging,
    )


def state(*, refund=False, charge_on_file=None, reply="here you go") -> State:
    message = Message(id=1, content="hello", incoming=True)
    return State(
        conversation=Conversation(
            id=7, contact_email="a@b.com", messages=(message,)
        ),
        message=message,
        knowledge="guidance",
        inbox=FakeInbox(),
        payments=FakePayments(charge_on_file),
        understand=lambda turns, knowledge, articles=(), tools=None: proposal(
            refund=refund, reply=reply
        ),
        now=NOW,
    )


# --------------------------------------------------------------- the shape


def test_every_node_has_an_edge():
    assert set(NODES) == set(EDGES), "a node with no edge would hang the walk"


def test_every_route_leads_to_a_real_node():
    routes = {
        "understand": [state(refund=True), state(refund=False)],
        "refund": [
            state(refund=True, charge_on_file=charge()),
            state(refund=True, charge_on_file=None),
        ],
    }
    for name, cases in routes.items():
        for case in cases:
            # Every node before this one has to have run, or its state is missing.
            understand_node(case)
            NODES[name](case)
            destination = EDGES[name](case)
            assert destination in NODES, f"{name} routed to {destination!r}"


def test_the_terminal_nodes_end_the_walk():
    for name in ("answer", "execute_refund", "hold"):
        assert EDGES[name](state()) is None


# -------------------------------------------------------------- the routing


@pytest.mark.parametrize(
    ("refund_requested", "expected"),
    [(True, "refund"), (False, "answer")],
)
def test_the_model_chooses_between_answering_and_the_refund_path(refund_requested, expected):
    current = state(refund=refund_requested)
    understand_node(current)
    assert EDGES["understand"](current) == expected


def test_the_policy_chooses_whether_money_moves_not_the_model():
    """The model asked for a refund in both cases. The edge differs because the
    charge does."""
    approved = state(refund=True, charge_on_file=charge())
    understand_node(approved)
    refund_node(approved)
    assert EDGES["refund"](approved) == "execute_refund"

    too_big = state(refund=True, charge_on_file=charge(amount_cents=90_000))
    understand_node(too_big)
    refund_node(too_big)
    assert EDGES["refund"](too_big) == "hold"


def test_a_confident_model_cannot_force_a_refund():
    """Even at a perfect rubric score, an ineligible charge routes to hold."""
    current = state(refund=True, charge_on_file=charge(refunded=True))
    current.understand = lambda turns, knowledge, articles=(), tools=None: proposal(
        refund=True, clear=True, named=True, hedging=False
    )
    understand_node(current)
    assert current.proposal is not None and current.proposal.confidence == 1.0
    refund_node(current)
    assert EDGES["refund"](current) == "hold"


# ------------------------------------------------------ nodes on their own


def test_the_answer_node_replies_and_closes():
    current = state()
    understand_node(current)
    answer_node(current)
    assert current.inbox.replies == [(7, "here you go")]
    assert current.inbox.resolved == [7]
    assert current.action == "answered"


def test_the_refund_node_decides_but_does_nothing():
    current = state(refund=True, charge_on_file=charge())
    understand_node(current)
    refund_node(current)
    assert current.decision is not None and current.decision.auto_approve
    assert current.payments.refunded == [], "deciding is not doing"
    assert current.inbox.replies == []


def test_the_execute_node_refunds_and_states_the_amount():
    current = state(refund=True, charge_on_file=charge())
    understand_node(current)
    refund_node(current)
    execute_refund_node(current)
    assert current.payments.refunded == [("ch_1", "refund-conv7-ch_1")]
    assert "20.00" in current.inbox.replies[0][1]
    assert current.action == "refunded"


def test_the_hold_node_leaves_it_open_for_a_person():
    current = state(refund=True, charge_on_file=charge(amount_cents=90_000))
    understand_node(current)
    refund_node(current)
    hold_node(current)
    assert current.inbox.notes, "a colleague needs the reason"
    assert current.inbox.resolved == [], "a person still has to act"
    assert current.action == "flagged"


# ------------------------------------------------------------ walking it


def test_a_support_question_walks_understand_then_answer():
    current = run_graph(state())
    assert current.action == "answered"
    assert current.inbox.replies == [(7, "here you go")]


def test_a_qualifying_refund_walks_all_the_way_to_the_money():
    current = run_graph(state(refund=True, charge_on_file=charge()))
    assert current.action == "refunded"
    assert current.payments.refunded


def test_an_ineligible_refund_walks_to_hold():
    current = run_graph(state(refund=True, charge_on_file=None))
    assert current.action == "flagged"
    assert current.payments.refunded == []


def test_the_walk_can_start_anywhere():
    """Useful for retrying a conversation from part way through."""
    current = state(refund=True, charge_on_file=charge())
    understand_node(current)
    finished = run_graph(current, start="refund")
    assert finished.action == "refunded"
