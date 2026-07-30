"""Tests for the model client: request shape, parsing and retries.

A one-off 400 has been seen for a request that then succeeds
unchanged, so the retry behaviour is pinned down here rather than discovered
during a demo.
"""

import json

import httpx
import pytest

from agent import Brain


def completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def brain_for(responses):
    """A Brain that returns the given responses in order. Sleeping is
    replaced so retry tests do not actually wait."""
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
    )
    return client, seen, slept


# ------------------------------------------------------------ request shape


def test_classification_asks_for_a_strict_schema():
    brain, seen, _ = brain_for([(200, completion('{"intent":"refund","confidence":0.9}'))])
    brain.classify("I want my money back")
    body = json.loads(seen[0].content)
    assert body["model"] == "nvidia/nemotron-3-nano-30b-a3b:free"
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True, "sent as a hint; honoured only by some endpoints"
    schema = fmt["json_schema"]["schema"]
    assert schema["properties"]["intent"]["enum"] == ["refund", "support"]
    assert schema["additionalProperties"] is False


def test_both_calls_pin_the_temperature():
    """Left unset, a small model drops stray tokens into otherwise fine
    sentences and the classifier's confidence wanders on identical input."""
    brain, seen, _ = brain_for(
        [
            (200, completion('{"intent":"refund","confidence":0.9}')),
            (200, completion("an answer")),
        ]
    )
    brain.classify("money back")
    brain.answer("q", "kb")
    assert [json.loads(r.content)["temperature"] for r in seen] == [0.0, 0.0]


def test_the_temperature_is_configurable():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=completion("hi"))

    brain = Brain(
        "gsk_test",
        temperature=0.4,
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    brain.answer("q", "kb")
    assert json.loads(seen[0].content)["temperature"] == 0.4


def test_the_customer_message_is_the_user_turn():
    brain, seen, _ = brain_for([(200, completion('{"intent":"support","confidence":0.7}'))])
    brain.classify("I cannot log in")
    body = json.loads(seen[0].content)
    assert body["messages"][-1] == {"role": "user", "content": "I cannot log in"}


def test_answering_passes_the_knowledge_and_asks_for_no_schema():
    brain, seen, _ = brain_for([(200, completion("Try the reset link."))])
    assert brain.answer("help", "## Cannot log in\nUse the reset link.") == "Try the reset link."
    body = json.loads(seen[0].content)
    assert "response_format" not in body, "free text, not JSON"
    assert "Use the reset link." in body["messages"][0]["content"]


def test_a_custom_model_is_honoured():
    brain, seen, _ = brain_for([(200, completion("hi"))])
    brain._model = "openai/gpt-oss-120b"
    brain.answer("q", "kb")
    assert json.loads(seen[0].content)["model"] == "openai/gpt-oss-120b"


# --------------------------------------------------------- provider routing


def test_by_default_no_provider_constraints_are_sent():
    """Requiring advertised schema support narrows the free catalogue from
    fourteen models to four, and does not actually guarantee enforcement."""
    brain, seen, _ = brain_for([(200, completion('{"intent":"refund","confidence":0.9}'))])
    brain.classify("money back")
    assert "provider" not in json.loads(seen[0].content)


def test_schema_support_can_be_insisted_on():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200, json=completion('{"intent":"refund","confidence":0.9}')
        )

    brain = Brain(
        "gsk_test",
        require_schema=True,
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    brain.classify("money back")
    assert json.loads(seen[0].content)["provider"]["require_parameters"] is True


def _brain_with(**kwargs):
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200, json=completion('{"intent":"support","confidence":0.9}')
        )

    brain = Brain(
        "gsk_test",
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        **kwargs,
    )
    return brain, seen


def test_a_pinned_provider_refuses_fallbacks():
    """A provider can advertise strict schema support and still return prose,
    so the provider itself has to be nameable."""
    brain, seen = _brain_with(provider_order=("Groq", "Fireworks"))
    brain.classify("hello")
    provider = json.loads(seen[0].content)["provider"]
    assert provider["order"] == ["Groq", "Fireworks"]
    assert provider["allow_fallbacks"] is False
    assert "require_parameters" not in provider, "pinning is independent of insisting"


def test_pinning_also_applies_when_writing_an_answer():
    brain, seen = _brain_with(provider_order=("Groq",))
    brain.answer("q", "kb")
    assert json.loads(seen[0].content)["provider"]["order"] == ["Groq"]


def test_provider_routing_is_not_sent_to_non_openrouter_endpoints():
    """`provider` is an OpenRouter extension; a plain endpoint may reject it."""
    brain, seen = _brain_with(
        base_url="https://api.groq.com/openai/v1", provider_order=("Groq",)
    )
    brain.classify("hello")
    assert "provider" not in json.loads(seen[0].content)


# ---------------------------------------------------------------- parsing


def test_classification_is_parsed():
    brain, _, _ = brain_for([(200, completion('{"intent":"refund","confidence":0.83}'))])
    result = brain.classify("money back please")
    assert result.intent == "refund"
    assert result.confidence == pytest.approx(0.83)


def test_integer_confidence_is_coerced_to_float():
    brain, _, _ = brain_for([(200, completion('{"intent":"support","confidence":1}'))])
    assert brain.classify("hello").confidence == pytest.approx(1.0)


# ------------------------------------------------- unusable classifications
# Endpoints that generate JSON by instruction rather than constrained decoding
# are right most of the time and malformed some of the time.


@pytest.mark.parametrize(
    ("label", "reply"),
    [
        ("prose", "refund\nconfidence: 0.99"),
        ("truncated json", '{"intent":"refund","confidence":0.9'),
        ("malformed json", '{"intent":"refund" "confidence":0.9}'),
        ("missing field", '{"intent":"refund"}'),
        ("unknown intent", '{"intent":"chargeback","confidence":0.9}'),
        ("confidence out of range", '{"intent":"refund","confidence":42}'),
        ("confidence not a number", '{"intent":"refund","confidence":"high"}'),
        ("empty", ""),
    ],
)
def test_an_unusable_reply_is_retried_then_accepted(label, reply):
    brain, seen, _ = brain_for(
        [(200, completion(reply)), (200, completion('{"intent":"refund","confidence":0.9}'))]
    )
    result = brain.classify("money back")
    assert result.intent == "refund", label
    assert len(seen) == 2, "should have asked again"


def test_an_unknown_intent_is_never_returned():
    """The refund path branches on this string; anything else must not survive."""
    brain, _, _ = brain_for([(200, completion('{"intent":"chargeback","confidence":0.9}'))] * 3)
    with pytest.raises(RuntimeError) as failure:
        brain.classify("hello")
    assert "usable classification" in str(failure.value)


def test_a_confidence_outside_zero_to_one_is_never_returned():
    """refund_decision compares this to a threshold, so 42 would auto-approve
    everything."""
    brain, _, _ = brain_for([(200, completion('{"intent":"refund","confidence":42}'))] * 3)
    with pytest.raises(RuntimeError):
        brain.classify("hello")


def test_giving_up_reports_the_last_reply():
    brain, seen, _ = brain_for([(200, completion("still not json"))] * 3)
    with pytest.raises(RuntimeError) as failure:
        brain.classify("hello")
    assert len(seen) == 3
    assert "still not json" in str(failure.value)


def test_boundary_confidences_are_accepted():
    for value in ("0", "1", "0.8"):
        brain, _, _ = brain_for(
            [(200, completion(f'{{"intent":"support","confidence":{value}}}'))]
        )
        assert brain.classify("hello").confidence == pytest.approx(float(value))


# ---------------------------------------------------------------- retries


@pytest.mark.parametrize("status", [400, 429, 500, 503])
def test_transient_failures_are_retried(status):
    brain, seen, slept = brain_for(
        [(status, {"error": "transient"}), (200, completion('{"intent":"support","confidence":0.9}'))]
    )
    assert brain.classify("hello").intent == "support"
    assert len(seen) == 2, "should have retried once"
    assert slept == [1], "one second before the second attempt"


def test_retries_back_off_then_give_up_reporting_the_body():
    brain, seen, slept = brain_for([(503, {"error": "still down"})] * 3)
    with pytest.raises(RuntimeError) as failure:
        brain.classify("hello")
    assert len(seen) == 3, "three attempts, then stop"
    assert slept == [1, 2], "exponential backoff between attempts"
    assert "503" in str(failure.value)
    assert "still down" in str(failure.value), "the body must not be swallowed"


def test_a_permanent_failure_is_not_retried():
    brain, seen, _ = brain_for([(401, {"error": "bad key"})])
    with pytest.raises(RuntimeError) as failure:
        brain.classify("hello")
    assert len(seen) == 1, "401 will not fix itself"
    assert "bad key" in str(failure.value)


def test_requests_are_authorised():
    brain, seen, _ = brain_for([(200, completion("hi"))])
    brain.answer("q", "kb")
    assert seen[0].headers["Authorization"] == "Bearer gsk_test"
