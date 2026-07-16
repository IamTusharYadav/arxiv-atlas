from pydantic import BaseModel

from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import RunContext
from atlas_agents.prompts import DIRECTION
from atlas_agents.steps.evidence import paper_block
from atlas_core.models import Paper

# Coarse naming needs less context than extraction; reranker-sized leads keep k calls cheap.
_ABSTRACT_CHARS = 500


class Direction(BaseModel):
    name: str
    problem: str
    representative_ids: list[str]


def label_direction(
    client: BedrockClient, ctx: RunContext, topic: str, papers: list[Paper]
) -> Direction:
    """Name one cluster of papers; invented representative ids fall back to cluster order."""
    blocks = "\n\n".join(
        paper_block(p.arxiv_id, p.title, p.abstract, _ABSTRACT_CHARS) for p in papers
    )
    direction, completion = client.complete_structured(
        model=DIRECTION.model,
        system=DIRECTION.system,
        prompt=f"<topic>{topic}</topic>\n\n{blocks}",
        output_type=Direction,
        max_tokens=300,
    )
    known = {p.arxiv_id for p in papers}
    kept = [i for i in direction.representative_ids if i in known]
    if not kept:
        kept = [p.arxiv_id for p in papers[:3]]  # papers arrive most-central-first
    direction = direction.model_copy(update={"representative_ids": kept[:3]})
    ctx.record(
        "direction",
        direction.name,
        completion,
        model=DIRECTION.model,
        version=DIRECTION.version,
    )
    return direction
