import re

from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import AgentError, RunContext
from atlas_agents.prompts import SYNTHESIZER
from atlas_agents.steps.evidence import paper_block
from atlas_agents.steps.extractor import PaperClaims
from atlas_core.vectorstore import ScoredPaper

# New-style (2507.01234) and old-style (cs/0112017) arXiv ids in [brackets].
_CITATION = re.compile(r"\[([a-zA-Z.-]+/\d{7}|\d{4}\.\d{4,5})\]")


class UngroundedCitations(AgentError):
    """The brief cited ids outside the evidence even after one repair attempt."""


def _invented(markdown: str, known: set[str]) -> set[str]:
    return {m.group(1) for m in _CITATION.finditer(markdown)} - known


def synthesize(
    client: BedrockClient,
    ctx: RunContext,
    question: str,
    papers: list[ScoredPaper],
    claims: list[PaperClaims],
) -> str:
    titles = {s.paper.arxiv_id: s.paper.title for s in papers}
    evidence = "\n\n".join(
        paper_block(
            c.arxiv_id,
            titles.get(c.arxiv_id, ""),
            "\n".join(f"- {claim}" for claim in c.claims),
        )
        for c in claims
    )
    prompt = f"<question>{question}</question>\n\n{evidence}"
    known = {c.arxiv_id for c in claims}

    completion = client.complete(
        model=SYNTHESIZER.model, system=SYNTHESIZER.system, prompt=prompt, max_tokens=2000
    )
    invented = _invented(completion.text, known)
    if invented:
        repair = client.complete(
            model=SYNTHESIZER.model,
            system=SYNTHESIZER.system,
            prompt=f"{prompt}\n\nYour previous draft cited ids that are not in the "
            f"evidence: {sorted(invented)}. Rewrite the brief citing only provided ids.",
            max_tokens=2000,
        )
        ctx.record(
            "synthesizer",
            "repaired invented citations",
            completion,
            model=SYNTHESIZER.model,
            version=SYNTHESIZER.version,
        )
        completion = repair
        invented = _invented(completion.text, known)
        if invented:
            ctx.record(
                "synthesizer",
                f"ungrounded citations: {sorted(invented)}",
                completion,
                model=SYNTHESIZER.model,
                version=SYNTHESIZER.version,
            )
            raise UngroundedCitations(
                f"brief cites unknown ids after repair: {sorted(invented)}",
                spent_usd=ctx.spent_usd,
            )

    cited = {m.group(1) for m in _CITATION.finditer(completion.text)}
    ctx.record(
        "synthesizer",
        f"brief citing {len(cited)} papers",
        completion,
        model=SYNTHESIZER.model,
        version=SYNTHESIZER.version,
    )
    return completion.text
