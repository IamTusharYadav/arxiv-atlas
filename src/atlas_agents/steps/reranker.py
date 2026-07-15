from pydantic import BaseModel, Field

from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import RunContext
from atlas_agents.prompts import RERANKER
from atlas_agents.steps.evidence import paper_block
from atlas_core.vectorstore import ScoredPaper

KEEP = 8
SCORE_FLOOR = 5
# Abstract lead per candidate, half the extractor's window. Reranking only scores coarse
# relevance over vector-prefiltered candidates, so the lead is enough, and since the candidate
# count grows each round this is the step's main input cost. Watch the reranker relevance
# score in evals before trimming further.
_ABSTRACT_CHARS = 500
# The retriever widens per round, so late rounds can hand over 100+ candidates. Each score is
# ~20 output tokens, so past ~75 the response outgrows max_tokens below, truncates, and fails
# both validation and its repair (which truncates identically). Candidates arrive best-first,
# so capping keeps the strongest and bounds the output well under the token ceiling.
MAX_CANDIDATES = 60


class CandidateScore(BaseModel):
    arxiv_id: str
    score: int = Field(ge=0, le=10)


class Reranking(BaseModel):
    scores: list[CandidateScore]


def rerank(
    client: BedrockClient,
    ctx: RunContext,
    question: str,
    candidates: list[ScoredPaper],
    keep: int = KEEP,
    score_floor: int = SCORE_FLOOR,
) -> list[ScoredPaper]:
    """Top `keep` candidates scoring at least `score_floor`, best first. Papers the
    model does not score are dropped; ids it invents are ignored."""
    if not candidates:
        ctx.record("reranker", "no candidates to score")
        return []
    candidates = candidates[:MAX_CANDIDATES]

    blocks = "\n\n".join(
        paper_block(s.paper.arxiv_id, s.paper.title, s.paper.abstract, _ABSTRACT_CHARS)
        for s in candidates
    )
    reranking, completion = client.complete_structured(
        model=RERANKER.model,
        system=RERANKER.system,
        prompt=f"<question>{question}</question>\n\n{blocks}",
        output_type=Reranking,
        max_tokens=1500,
    )

    by_id = {s.paper.arxiv_id: s.paper for s in candidates}
    scored = [
        ScoredPaper(paper=by_id[cs.arxiv_id], score=float(cs.score))
        for cs in reranking.scores
        if cs.arxiv_id in by_id and cs.score >= score_floor
    ]
    kept = sorted(scored, key=lambda s: s.score, reverse=True)[:keep]
    ctx.record(
        "reranker",
        f"kept {len(kept)} of {len(candidates)}",
        completion,
        model=RERANKER.model,
        version=RERANKER.version,
    )
    return kept
