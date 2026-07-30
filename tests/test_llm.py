"""Tests for the model client: request shape, parsing and retries.

The client makes one call per message and returns a proposal: the reply to send,
and whether the customer is asking for a refund. Whether money actually moves is
decided afterwards in code, so what matters here is that the proposal is well
formed and that a bad one is never passed on.

A one-off 400 has been seen for a request that then succeeds unchanged, so the
retry behaviour is pinned down here rather than discovered during a demo.
"""

import json

import httpx
import pytest

from agent import Article, Brain, Turn

PROPOSAL = '{"reply":"here you go","refund_requested":false,"confidence":0.9}'
REFUND_PROPOSAL = '{"reply":"looking into it","refund_requested":true,"confidence":0.95}'

CONVERSATION = (
    Turn(role="user", content="my delivery is late"),
    Turn(role="assistant", content="sorry about that, when was it due?"),
    Turn(role="user", content="last Tuesday"),
)


def completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def brain_for(responses, **kwargs):
    """A Brain returning the given responses in order. Sleeping is replaced so
    retry tests do not actually wait."""
    remaining = list(responses)
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = remaining.pop(0)
        return httpx.Response(status, json=body)

    client = Brain(
        "gsk_test",
        transport=httpx.MockTransport(handler),
        sleep=slept.append,
        **kwargs,
    )
    return client, seen, slept


def body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)


# ------------------------------------------------------------ request shape


def test_the_whole_conversation_is_sent_as_message_history():
    """The point of the redesign: a label carries no context, a thread does."""
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(CONVERSATION, "be nice")
    turns = [m for m in body_of(seen[0])["messages"] if m["role"] != "system"]
    assert turns == [
        {"role": "user", "content": "my delivery is late"},
        {"role": "assistant", "content": "sorry about that, when was it due?"},
        {"role": "user", "content": "last Tuesday"},
    ]


def test_the_behaviour_guidance_goes_in_the_system_prompt():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(CONVERSATION, "ALWAYS BE KIND")
    system = body_of(seen[0])["messages"][0]
    assert system["role"] == "system"
    assert "ALWAYS BE KIND" in system["content"]


def test_articles_are_offered_as_facts():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(
        CONVERSATION,
        "be nice",
        [Article(title="Refunds", content="We refund a bad bag.")],
    )
    system = body_of(seen[0])["messages"][0]["content"]
    assert "Refunds" in system
    assert "We refund a bad bag." in system


def test_with_no_articles_it_is_told_not_to_state_product_facts():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(CONVERSATION, "be nice", [])
    assert "no documentation matched" in body_of(seen[0])["messages"][0]["content"].lower()


def test_the_output_contract_asks_for_a_strict_schema():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(CONVERSATION, "be nice")
    fmt = body_of(seen[0])["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert sorted(schema["required"]) == ["confidence", "refund_requested", "reply"]
    assert schema["additionalProperties"] is False


def test_the_temperature_is_pinned():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(CONVERSATION, "be nice")
    assert body_of(seen[0])["temperature"] == 0.0


def test_the_model_is_the_fast_free_one_by_default():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(CONVERSATION, "be nice")
    assert body_of(seen[0])["model"] == "nvidia/nemotron-3-nano-30b-a3b:free"


# --------------------------------------------------------- provider routing


def test_by_default_no_provider_constraints_are_sent():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(CONVERSATION, "be nice")
    assert "provider" not in body_of(seen[0])


def test_schema_support_can_be_insisted_on():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))], require_schema=True)
    brain.understand(CONVERSATION, "be nice")
    assert body_of(seen[0])["provider"]["require_parameters"] is True


def test_a_pinned_provider_refuses_fallbacks():
    brain, seen, _ = brain_for(
        [(200, completion(PROPOSAL))], provider_order=("Groq", "Fireworks")
    )
    brain.understand(CONVERSATION, "be nice")
    provider = body_of(seen[0])["provider"]
    assert provider["order"] == ["Groq", "Fireworks"]
    assert provider["allow_fallbacks"] is False


def test_provider_routing_is_not_sent_to_non_openrouter_endpoints():
    """`provider` is an OpenRouter extension; a plain endpoint may reject it."""
    brain, seen, _ = brain_for(
        [(200, completion(PROPOSAL))],
        base_url="https://api.groq.com/openai/v1",
        provider_order=("Groq",),
    )
    brain.understand(CONVERSATION, "be nice")
    assert "provider" not in body_of(seen[0])


# ------------------------------------------------------------------ parsing


def test_a_proposal_is_parsed():
    brain, _, _ = brain_for([(200, completion(REFUND_PROPOSAL))])
    proposal = brain.understand(CONVERSATION, "be nice")
    assert proposal.reply == "looking into it"
    assert proposal.refund_requested is True
    assert proposal.confidence == pytest.approx(0.95)


def test_an_integer_confidence_is_coerced():
    brain, _, _ = brain_for(
        [(200, completion('{"reply":"hi","refund_requested":false,"confidence":1}'))]
    )
    assert brain.understand(CONVERSATION, "be nice").confidence == pytest.approx(1.0)


def test_surrounding_whitespace_is_trimmed_from_the_reply():
    brain, _, _ = brain_for(
        [(200, completion('{"reply":"  hi  ","refund_requested":false,"confidence":1}'))]
    )
    assert brain.understand(CONVERSATION, "be nice").reply == "hi"


# ------------------------------------------------ proposals we cannot trust
# Endpoints that generate JSON by instruction rather than constrained decoding
# are right most of the time and malformed some of the time.


@pytest.mark.parametrize(
    ("label", "reply"),
    [
        ("prose", "sure, I can help with that"),
        ("truncated json", '{"reply":"hi","refund_requested":false'),
        ("malformed json", '{"reply":"hi" "refund_requested":false}'),
        ("missing reply", '{"refund_requested":false,"confidence":0.9}'),
        ("missing flag", '{"reply":"hi","confidence":0.9}'),
        ("empty reply", '{"reply":"","refund_requested":false,"confidence":0.9}'),
        ("blank reply", '{"reply":"   ","refund_requested":false,"confidence":0.9}'),
        ("flag as string", '{"reply":"hi","refund_requested":"yes","confidence":0.9}'),
        ("confidence too high", '{"reply":"hi","refund_requested":true,"confidence":42}'),
        (
            "confidence not a number",
            '{"reply":"hi","refund_requested":true,"confidence":"high"}',
        ),
        ("empty", ""),
    ],
)
def test_an_unusable_proposal_is_retried_then_accepted(label, reply):
    brain, seen, _ = brain_for([(200, completion(reply)), (200, completion(PROPOSAL))])
    assert brain.understand(CONVERSATION, "be nice").reply == "here you go", label
    assert len(seen) == 2, "should have asked again"


def test_a_string_flag_is_never_treated_as_a_refund_request():
    """"yes" is truthy in Python; letting it through would move money on a
    malformed reply."""
    bad = '{"reply":"hi","refund_requested":"yes","confidence":0.99}'
    brain, _, _ = brain_for([(200, completion(bad))] * Brain.MAX_ATTEMPTS)
    with pytest.raises(RuntimeError):
        brain.understand(CONVERSATION, "be nice")


def test_a_confidence_outside_zero_to_one_is_never_returned():
    """refund_decision compares this to a threshold, so 42 would auto-approve
    everything."""
    bad = '{"reply":"hi","refund_requested":true,"confidence":42}'
    brain, _, _ = brain_for([(200, completion(bad))] * Brain.MAX_ATTEMPTS)
    with pytest.raises(RuntimeError):
        brain.understand(CONVERSATION, "be nice")


def test_giving_up_reports_the_last_reply():
    attempts = Brain.MAX_ATTEMPTS
    brain, seen, _ = brain_for([(200, completion("still not json"))] * attempts)
    with pytest.raises(RuntimeError) as failure:
        brain.understand(CONVERSATION, "be nice")
    assert len(seen) == attempts
    assert "still not json" in str(failure.value)


# ------------------------------------------------------------------ retries


@pytest.mark.parametrize("status", [400, 408, 429, 500, 503])
def test_transient_failures_are_retried(status):
    brain, seen, slept = brain_for(
        [(status, {"error": "transient"}), (200, completion(PROPOSAL))]
    )
    assert brain.understand(CONVERSATION, "be nice").reply == "here you go"
    assert len(seen) == 2
    assert slept == [1]


def test_retries_back_off_then_give_up_reporting_the_body():
    attempts = Brain.MAX_ATTEMPTS
    brain, seen, slept = brain_for([(503, {"error": "still down"})] * attempts)
    with pytest.raises(RuntimeError) as failure:
        brain.understand(CONVERSATION, "be nice")
    assert len(seen) == attempts
    assert slept == [2 ** n for n in range(attempts - 1)], "exponential backoff"
    assert "503" in str(failure.value)
    assert "still down" in str(failure.value), "the body must not be swallowed"


def test_a_permanent_failure_is_not_retried():
    brain, seen, _ = brain_for([(401, {"error": "bad key"})])
    with pytest.raises(RuntimeError) as failure:
        brain.understand(CONVERSATION, "be nice")
    assert len(seen) == 1, "401 will not fix itself"
    assert "bad key" in str(failure.value)


def test_requests_are_authorised():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(CONVERSATION, "be nice")
    assert seen[0].headers["Authorization"] == "Bearer gsk_test"


def test_a_numeric_string_confidence_is_accepted():
    """Some inputs make the model write "0.95" with quotes every single time.
    The schema allows it and the value is coerced, rather than failing the
    request five times over a JSON type."""
    brain, _, _ = brain_for(
        [(200, completion('{"reply":"hi","refund_requested":true,"confidence":"0.95"}'))]
    )
    proposal = brain.understand(CONVERSATION, "be nice")
    assert proposal.confidence == pytest.approx(0.95)
    assert proposal.refund_requested is True


def test_the_schema_allows_confidence_as_either_type():
    brain, seen, _ = brain_for([(200, completion(PROPOSAL))])
    brain.understand(CONVERSATION, "be nice")
    schema = body_of(seen[0])["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["confidence"]["type"] == ["number", "string"]
