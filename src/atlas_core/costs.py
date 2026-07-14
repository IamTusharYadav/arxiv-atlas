"""Token to USD for the Bedrock models we call. Prices are USD per million tokens, checked
against Anthropic pricing on 2026-07-12; re-verify when adding a model."""

_PER_MTOK: dict[str, tuple[float, float]] = {
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.00, 5.00),
    "us.anthropic.claude-sonnet-4-6": (3.00, 15.00),
}

# Cache pricing multipliers relative to the input price.
_CACHE_WRITE = 1.25
_CACHE_READ = 0.10


def usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Cost of one call. Unknown models raise so a new model can never be silently free."""
    try:
        input_price, output_price = _PER_MTOK[model]
    except KeyError:
        raise ValueError(f"no price table entry for model {model!r}") from None
    return (
        input_tokens * input_price
        + output_tokens * output_price
        + cache_write_tokens * input_price * _CACHE_WRITE
        + cache_read_tokens * input_price * _CACHE_READ
    ) / 1_000_000
