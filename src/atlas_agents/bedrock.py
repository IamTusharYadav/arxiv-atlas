import logging
import os
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import anthropic
from anthropic.types import Message, MessageParam
from pydantic import BaseModel, ValidationError

from atlas_core.costs import usd

log = logging.getLogger(__name__)

# Newer Claude models on Bedrock are inference-profile-only (bare on-demand ids 400),
# and this account is not entitled to the bedrock-mantle endpoint, so we use the native
# InvokeModel client (AnthropicBedrock) with us-region inference-profile ids. Sonnet 5 is
# gated for this account (AWS Sales approval); Sonnet 4.6 is the working stand-in.
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "us.anthropic.claude-sonnet-4-6"

_DEGRADE = {SONNET: HAIKU}

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Model output failed schema validation even after one repair attempt."""


@dataclass(frozen=True)
class Completion:
    """One model response plus the usage that budget accounting needs.

    After a structured-output repair, token and cost fields aggregate both
    attempts so the caller charges the full spend against the budget.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


# Bedrock structured outputs reject JSON-schema constraint keywords beyond the core type
# system (e.g. integer minimum/maximum). Pydantic still enforces these on
# model_validate_json, so stripping them from the wire schema is safe: an out-of-range
# value fails validation client-side and triggers the existing repair round trip.
_UNSUPPORTED_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Add additionalProperties:false on every object node and drop constraint keywords
    Bedrock structured outputs reject (Pydantic re-checks them client-side)."""
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    for key in _UNSUPPORTED_KEYS & schema.keys():
        del schema[key]
    for value in schema.values():
        if isinstance(value, dict):
            _strict_schema(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _strict_schema(item)
    return schema


class BedrockClient:
    def __init__(
        self,
        region: str | None = None,
        client: anthropic.AnthropicBedrock | None = None,
    ) -> None:
        # Explicit arg wins; otherwise honor AWS_REGION so .env can pick the Bedrock
        # region (the SDK would otherwise not see it once we pass aws_region).
        region = region or os.environ.get("AWS_REGION") or "us-east-1"
        self._client = client or anthropic.AnthropicBedrock(aws_region=region)

    def complete(
        self, *, model: str, system: str, prompt: str, max_tokens: int = 1024
    ) -> Completion:
        """Plain text completion on the given tier."""
        return self._call(
            model=model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

    def complete_structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        output_type: type[T],
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> tuple[T, Completion]:
        """Completion constrained to output_type's JSON schema, validated by pydantic.

        The API guarantees schema-valid JSON, so validation only fails on truncation
        or semantic validators; one repair round trip covers those, then we abort.
        """
        output_config = {
            "format": {
                "type": "json_schema",
                "schema": _strict_schema(output_type.model_json_schema()),
            }
        }
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        first = self._call(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            output_config=output_config,
            temperature=temperature,
        )
        try:
            return output_type.model_validate_json(first.text), first
        except ValidationError as err:
            log.warning("structured output failed validation, attempting repair: %s", err)
            messages += [
                {"role": "assistant", "content": first.text},
                {
                    "role": "user",
                    "content": f"That JSON failed validation:\n{err}\n"
                    "Return only the corrected JSON object.",
                },
            ]
            second = self._call(
                model=model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                output_config=output_config,
                temperature=temperature,
            )
            combined = replace(
                second,
                input_tokens=first.input_tokens + second.input_tokens,
                output_tokens=first.output_tokens + second.output_tokens,
                cost_usd=first.cost_usd + second.cost_usd,
            )
            try:
                return output_type.model_validate_json(second.text), combined
            except ValidationError as final_err:
                raise StructuredOutputError(
                    f"output still invalid after repair: {final_err}"
                ) from final_err

    def _call(
        self,
        *,
        model: str,
        system: str,
        messages: list[MessageParam],
        max_tokens: int,
        output_config: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> Completion:
        extra: dict[str, Any] = {"output_config": output_config} if output_config else {}
        if temperature is not None:
            extra["temperature"] = temperature
        try:
            msg = self._client.messages.create(
                model=model, system=system, messages=messages, max_tokens=max_tokens, **extra
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError):
            fallback = _DEGRADE.get(model)
            if fallback is None:
                raise
            log.warning("bedrock %s unavailable after retries, degrading to %s", model, fallback)
            model = fallback
            msg = self._client.messages.create(
                model=model, system=system, messages=messages, max_tokens=max_tokens, **extra
            )
        return _completion(msg, model)


def _completion(msg: Message, model: str) -> Completion:
    # Cost is keyed on the model we called, not msg.model: Bedrock may echo the id
    # in a different form and the price lookup fails closed on unknown ids.
    if msg.stop_reason == "max_tokens":
        log.warning("bedrock response truncated at max_tokens for %s", model)
    text = "".join(block.text for block in msg.content if block.type == "text")
    u = msg.usage
    return Completion(
        text=text,
        model=model,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cost_usd=usd(
            model,
            u.input_tokens,
            u.output_tokens,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
            cache_read_tokens=u.cache_read_input_tokens or 0,
        ),
    )
