"""One-shot live smoke test of the Bedrock client. Costs well under a cent.

Run: uv run python scripts/smoke_bedrock.py [region]
Needs AWS credentials (env or ~/.aws/credentials) with bedrock:InvokeModel.
"""

import sys

from pydantic import BaseModel

from atlas_agents.bedrock import HAIKU, SONNET, BedrockClient


class Capital(BaseModel):
    country: str
    capital: str


def main() -> None:
    region = sys.argv[1] if len(sys.argv) > 1 else "us-east-1"
    client = BedrockClient(region=region)

    print(f"region: {region}")

    haiku = client.complete(
        model=HAIKU, system="Answer in one word.", prompt="Say ok.", max_tokens=20
    )
    print(f"haiku text:       {haiku.text!r}")
    print(f"haiku tokens:     {haiku.input_tokens} in / {haiku.output_tokens} out")
    print(f"haiku cost:       ${haiku.cost_usd:.6f}")

    parsed, structured = client.complete_structured(
        model=HAIKU,
        system="Extract the fact.",
        prompt="France's capital is Paris.",
        output_type=Capital,
        max_tokens=100,
    )
    print(f"structured:       {parsed!r}")
    print(f"structured cost:  ${structured.cost_usd:.6f}")

    sonnet = client.complete(
        model=SONNET, system="Answer in one word.", prompt="Say ok.", max_tokens=20
    )
    print(f"sonnet text:      {sonnet.text!r}")
    print(f"sonnet cost:      ${sonnet.cost_usd:.6f}")

    total = haiku.cost_usd + structured.cost_usd + sonnet.cost_usd
    print(f"total smoke cost: ${total:.6f}")


if __name__ == "__main__":
    main()
