"""Synthesizer step: the one Sonnet call, turning extracted claims into a cited brief.

Citations are validated against the evidence after the call: an id the model invents
gets one repair round trip, then the run fails rather than ship a hallucinated
citation (the eval rubric penalizes those hardest).
"""

import re

from atlas_agents.bedrock import SONNET, BedrockClient
from atlas_agents.harness import RunContext
from atlas_agents.steps.extractor import PaperClaims
from atlas_core.vectorstore import ScoredPaper

# New-style (2507.01234) and old-style (cs/0112017) arXiv ids in [brackets].
_CITATION = re.compile(r"\[([a-zA-Z.-]+/\d{7}|\d{4}\.\d{4,5})\]")

SYNTHESIZER_SYSTEM = """\
You write a research brief answering a question from extracted paper claims.

Rules:
- Markdown, at most ~500 words: a one-paragraph answer first, then short sections
  grouping the papers by approach, then one line on open tradeoffs.
- Every factual statement carries an inline citation like [2507.01234].
- Cite only the paper ids provided. Never invent an id, and never cite a paper for
  a claim it does not make.
- If the evidence only partially covers the question, say what is missing instead
  of papering over the gap."""


class UngroundedCitations(RuntimeError):
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
        f"<paper id={c.arxiv_id!r}>\n{titles.get(c.arxiv_id, '')}\n"
        + "\n".join(f"- {claim}" for claim in c.claims)
        + "\n</paper>"
        for c in claims
    )
    prompt = f"<question>{question}</question>\n\n{evidence}"
    known = {c.arxiv_id for c in claims}

    completion = client.complete(
        model=SONNET, system=SYNTHESIZER_SYSTEM, prompt=prompt, max_tokens=2000
    )
    invented = _invented(completion.text, known)
    if invented:
        repair = client.complete(
            model=SONNET,
            system=SYNTHESIZER_SYSTEM,
            prompt=f"{prompt}\n\nYour previous draft cited ids that are not in the "
            f"evidence: {sorted(invented)}. Rewrite the brief citing only provided ids.",
            max_tokens=2000,
        )
        ctx.record("synthesizer", "repaired invented citations", completion)
        completion = repair
        invented = _invented(completion.text, known)
        if invented:
            ctx.record("synthesizer", f"ungrounded citations: {sorted(invented)}", completion)
            raise UngroundedCitations(f"brief cites unknown ids after repair: {sorted(invented)}")

    cited = {m.group(1) for m in _CITATION.finditer(completion.text)}
    ctx.record("synthesizer", f"brief citing {len(cited)} papers", completion)
    return completion.text
