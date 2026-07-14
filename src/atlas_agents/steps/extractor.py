"""Extractor step: per-paper key claims relevant to the question, one batched Haiku call.

The claims are the synthesizer's working material; sending compressed claims instead
of full abstracts is what keeps the Sonnet synthesis call cheap.
"""

from pydantic import BaseModel

from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import RunContext
from atlas_agents.prompts import EXTRACTOR
from atlas_agents.steps.evidence import paper_block
from atlas_core.vectorstore import ScoredPaper

_ABSTRACT_CHARS = 1500


class PaperClaims(BaseModel):
    arxiv_id: str
    claims: list[str]


class Extraction(BaseModel):
    papers: list[PaperClaims]


def extract(
    client: BedrockClient,
    ctx: RunContext,
    question: str,
    papers: list[ScoredPaper],
) -> list[PaperClaims]:
    """Claims per paper, in the input order; ids the model invents are dropped."""
    if not papers:
        ctx.record("extractor", "no papers to extract from")
        return []

    blocks = "\n\n".join(
        paper_block(s.paper.arxiv_id, s.paper.title, s.paper.abstract, _ABSTRACT_CHARS)
        for s in papers
    )
    extraction, completion = client.complete_structured(
        model=EXTRACTOR.model,
        system=EXTRACTOR.system,
        prompt=f"<question>{question}</question>\n\n{blocks}",
        output_type=Extraction,
        max_tokens=2000,
    )

    known = {s.paper.arxiv_id for s in papers}
    claims = [p for p in extraction.papers if p.arxiv_id in known and p.claims]
    ctx.record(
        "extractor",
        f"claims for {len(claims)} of {len(papers)} papers",
        completion,
        model=EXTRACTOR.model,
        version=EXTRACTOR.version,
    )
    return claims
