import anthropic
import httpx
import pytest
from pydantic import BaseModel

from atlas_agents.bedrock import HAIKU, SONNET, StructuredOutputError
from atlas_core.costs import usd
from tests.conftest import make_bedrock_client as make_client
from tests.conftest import make_message


def throttled() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://bedrock.test/messages")
    response = httpx.Response(429, request=request)
    return anthropic.RateLimitError("throttled", response=response, body=None)


class Answer(BaseModel):
    answer: str
    confidence: float


def test_complete_returns_text_and_cost() -> None:
    client, fake = make_client([make_message("hello", input_tokens=1000, output_tokens=200)])
    result = client.complete(model=HAIKU, system="be brief", prompt="hi")
    assert result.text == "hello"
    assert result.model == HAIKU
    assert result.cost_usd == pytest.approx((1000 * 1.00 + 200 * 5.00) / 1_000_000)
    assert fake.calls[0]["model"] == HAIKU
    assert fake.calls[0]["system"] == "be brief"


def test_sonnet_degrades_to_haiku_on_throttle() -> None:
    client, fake = make_client([throttled(), make_message("degraded")])
    result = client.complete(model=SONNET, system="s", prompt="p")
    assert result.model == HAIKU
    assert [call["model"] for call in fake.calls] == [SONNET, HAIKU]


def test_haiku_throttle_has_no_fallback() -> None:
    client, _ = make_client([throttled()])
    with pytest.raises(anthropic.RateLimitError):
        client.complete(model=HAIKU, system="s", prompt="p")


def test_structured_happy_path_sends_strict_schema() -> None:
    client, fake = make_client([make_message('{"answer": "42", "confidence": 0.9}')])
    parsed, completion = client.complete_structured(
        model=HAIKU, system="s", prompt="p", output_type=Answer
    )
    assert parsed == Answer(answer="42", confidence=0.9)
    assert completion.output_tokens == 50
    output_config = fake.calls[0]["output_config"]
    assert isinstance(output_config, dict)
    assert output_config["format"]["schema"]["additionalProperties"] is False


def test_structured_repair_aggregates_usage() -> None:
    client, fake = make_client(
        [make_message("not json"), make_message('{"answer": "ok", "confidence": 1.0}')]
    )
    parsed, completion = client.complete_structured(
        model=HAIKU, system="s", prompt="p", output_type=Answer
    )
    assert parsed.answer == "ok"
    assert completion.input_tokens == 200
    assert completion.output_tokens == 100
    assert completion.cost_usd == pytest.approx(2 * (100 * 1.00 + 50 * 5.00) / 1_000_000)
    # The repair turn shows the model its own bad output plus the validation error.
    repair_messages = fake.calls[1]["messages"]
    assert isinstance(repair_messages, list)
    assert repair_messages[1] == {"role": "assistant", "content": "not json"}
    assert "failed validation" in repair_messages[2]["content"]


def test_structured_gives_up_after_one_repair() -> None:
    client, _ = make_client([make_message("bad"), make_message("still bad")])
    with pytest.raises(StructuredOutputError):
        client.complete_structured(model=HAIKU, system="s", prompt="p", output_type=Answer)


def test_unknown_model_is_never_free() -> None:
    with pytest.raises(ValueError, match="no price table entry"):
        usd("anthropic.claude-future-9", 1, 1)


def test_cache_tokens_priced_at_multipliers() -> None:
    cost = usd(HAIKU, 0, 0, cache_write_tokens=1_000_000, cache_read_tokens=1_000_000)
    assert cost == pytest.approx(1.00 * 1.25 + 1.00 * 0.10)
