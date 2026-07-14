"""LLM-as-judge: score one agent answer against its golden query on a rubric-anchored 1-5
scale (plan section G). The judge sees only the question, the answer, and the cited papers'
abstracts, so it scores grounding rather than its own world knowledge.

Runs on Haiku at temperature 0 for reproducibility; `samples>1` takes the per-dimension
median across independent judgments to damp the residual nondeterminism on borderline answers.
"""

import statistics
from dataclasses import dataclass

from pydantic import BaseModel, Field

from atlas_agents.ask import Answer
from atlas_agents.bedrock import BedrockClient
from atlas_agents.prompts import JUDGE
from evals.golden_set import GoldenQuery

# Dimensions the CI gate blocks a merge on; citation_correctness is reported but not gated.
GATED_DIMS = ("relevance", "faithfulness")


class Scores(BaseModel):
    relevance: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    citation_correctness: int = Field(ge=1, le=5)
    rationale: str


@dataclass(frozen=True)
class Judgement:
    query_id: str
    scores: Scores
    cost_usd: float


def _cited_block(answer: Answer) -> str:
    if not answer.papers:
        return "(the answer cited no papers)"
    return "\n\n".join(
        f"<paper id={s.paper.arxiv_id!r}>\n{s.paper.title}\n{s.paper.abstract}\n</paper>"
        for s in answer.papers
    )


def _build_prompt(query: GoldenQuery, answer: Answer) -> str:
    scope = "in scope" if query.in_scope else "OUT OF SCOPE (a correct answer declines)"
    topics = ", ".join(query.expected_topics) or "(none given)"
    return (
        f"<scope>{scope}</scope>\n"
        f"<question>{query.question}</question>\n"
        f"<expected_topics>{topics}</expected_topics>\n"
        f"<rubric_notes>{query.rubric_notes}</rubric_notes>\n\n"
        f"<answer>\n{answer.brief}\n</answer>\n\n"
        f"<cited_papers>\n{_cited_block(answer)}\n</cited_papers>"
    )


def _median(judgements: list[Scores]) -> Scores:
    return Scores(
        relevance=round(statistics.median(s.relevance for s in judgements)),
        faithfulness=round(statistics.median(s.faithfulness for s in judgements)),
        citation_correctness=round(statistics.median(s.citation_correctness for s in judgements)),
        rationale=judgements[0].rationale,
    )


def judge(
    client: BedrockClient, query: GoldenQuery, answer: Answer, *, samples: int = 1
) -> Judgement:
    runs: list[Scores] = []
    cost = 0.0
    for _ in range(samples):
        scores, completion = client.complete_structured(
            model=JUDGE.model,
            system=JUDGE.system,
            prompt=_build_prompt(query, answer),
            output_type=Scores,
            max_tokens=500,
            temperature=0.0,
        )
        runs.append(scores)
        cost += completion.cost_usd
    return Judgement(query_id=query.id, scores=_median(runs), cost_usd=cost)
