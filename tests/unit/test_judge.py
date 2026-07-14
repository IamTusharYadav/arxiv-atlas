import json

from evals.golden_set import GoldenQuery
from evals.judge import _build_prompt, judge

from atlas_agents.ask import Answer
from atlas_core.vectorstore import ScoredPaper
from tests.conftest import make_bedrock_client, make_message, make_paper


def scores_json(relevance: int, faithfulness: int, citation: int, rationale: str = "ok") -> str:
    return json.dumps(
        {
            "relevance": relevance,
            "faithfulness": faithfulness,
            "citation_correctness": citation,
            "rationale": rationale,
        }
    )


def make_query(in_scope: bool = True) -> GoldenQuery:
    return GoldenQuery(
        id="kv-cache-compression",
        category="technical-survey",
        question="What are current approaches to KV cache compression?",
        in_scope=in_scope,
        expected_topics=("kv cache", "quantization"),
        rubric_notes="Name several method families with a cited paper each.",
    )


def make_answer() -> Answer:
    paper = make_paper(arxiv_id="2607.00001", title="KV Cache Quantization")
    return Answer(
        brief="Quantization shrinks the cache [2607.00001].",
        papers=[ScoredPaper(paper=paper, score=9.0)],
        trace=[],
        cost_usd=0.01,
    )


def test_judge_returns_scores_at_temperature_zero() -> None:
    client, fake = make_bedrock_client([make_message(scores_json(4, 5, 3))])
    result = judge(client, make_query(), make_answer())

    assert result.query_id == "kv-cache-compression"
    assert (result.scores.relevance, result.scores.faithfulness) == (4, 5)
    assert result.scores.citation_correctness == 3
    assert fake.calls[0]["temperature"] == 0.0
    assert result.cost_usd > 0


def test_median_of_three_samples_per_dimension() -> None:
    client, fake = make_bedrock_client(
        [
            make_message(scores_json(3, 5, 1)),
            make_message(scores_json(5, 5, 2)),
            make_message(scores_json(4, 2, 3)),
        ]
    )
    result = judge(client, make_query(), make_answer(), samples=3)

    assert result.scores.relevance == 4  # median(3, 5, 4)
    assert result.scores.faithfulness == 5  # median(5, 5, 2)
    assert result.scores.citation_correctness == 2  # median(1, 2, 3)
    assert len(fake.calls) == 3


def test_prompt_marks_scope_and_includes_answer_and_citations() -> None:
    prompt = _build_prompt(make_query(in_scope=False), make_answer())
    assert "OUT OF SCOPE" in prompt
    assert "KV cache compression" in prompt
    assert "2607.00001" in prompt  # cited paper block


def test_prompt_handles_answer_with_no_citations() -> None:
    answer = Answer(brief="I cannot answer.", papers=[], trace=[], cost_usd=0.0)
    prompt = _build_prompt(make_query(in_scope=False), answer)
    assert "cited no papers" in prompt
