import json

from atlas_agents.harness import RunContext
from atlas_agents.steps.extractor import extract
from atlas_core.vectorstore import ScoredPaper
from tests.conftest import make_bedrock_client, make_message, make_paper


def papers(*ids: str) -> list[ScoredPaper]:
    return [ScoredPaper(paper=make_paper(arxiv_id=i), score=8.0) for i in ids]


def extraction_json(claims_by_id: dict[str, list[str]]) -> str:
    return json.dumps({"papers": [{"arxiv_id": k, "claims": v} for k, v in claims_by_id.items()]})


def test_extract_batches_and_returns_claims() -> None:
    client, fake = make_bedrock_client(
        [
            make_message(
                extraction_json(
                    {
                        "2607.00001": ["proposes 4-bit kv quantization"],
                        "2607.00002": ["evicts low-attention tokens"],
                    }
                )
            )
        ]
    )
    ctx = RunContext()

    claims = extract(client, ctx, "kv cache?", papers("2607.00001", "2607.00002"))

    assert [c.arxiv_id for c in claims] == ["2607.00001", "2607.00002"]
    assert claims[0].claims == ["proposes 4-bit kv quantization"]
    prompt = fake.calls[0]["messages"][0]["content"]  # type: ignore[index]
    assert prompt.count("<paper id=") == 2
    assert ctx.trace[0].summary == "claims for 2 of 2 papers"


def test_extract_drops_invented_ids_and_empty_claims() -> None:
    client, _ = make_bedrock_client(
        [
            make_message(
                extraction_json(
                    {"2607.00001": ["real claim"], "9999.99999": ["fake"], "2607.00002": []}
                )
            )
        ]
    )
    claims = extract(client, RunContext(), "q", papers("2607.00001", "2607.00002"))
    assert [c.arxiv_id for c in claims] == ["2607.00001"]


def test_extract_empty_input_skips_model_call() -> None:
    client, fake = make_bedrock_client([])
    ctx = RunContext()
    assert extract(client, ctx, "q", []) == []
    assert fake.calls == []
