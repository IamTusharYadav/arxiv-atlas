import pytest

from atlas_agents.bedrock import SONNET
from atlas_agents.harness import RunContext
from atlas_agents.steps.extractor import PaperClaims
from atlas_agents.steps.synthesizer import UngroundedCitations, synthesize
from atlas_core.vectorstore import ScoredPaper
from tests.conftest import make_bedrock_client, make_message, make_paper

PAPERS = [ScoredPaper(paper=make_paper(arxiv_id="2607.00001"), score=9.0)]
CLAIMS = [PaperClaims(arxiv_id="2607.00001", claims=["proposes 4-bit kv quantization"])]

GROUNDED = "Quantization dominates [2607.00001]."
INVENTED = "Quantization dominates [2607.00001], see also [1234.56789]."


def test_synthesize_returns_grounded_brief_on_sonnet() -> None:
    client, fake = make_bedrock_client([make_message(GROUNDED, model=SONNET)])
    ctx = RunContext()

    brief = synthesize(client, ctx, "kv cache?", PAPERS, CLAIMS)

    assert brief == GROUNDED
    assert fake.calls[0]["model"] == SONNET
    prompt = fake.calls[0]["messages"][0]["content"]  # type: ignore[index]
    assert "proposes 4-bit kv quantization" in prompt
    assert ctx.trace[-1].summary == "brief citing 1 papers"


def test_invented_citation_gets_one_repair() -> None:
    client, fake = make_bedrock_client(
        [make_message(INVENTED, model=SONNET), make_message(GROUNDED, model=SONNET)]
    )
    ctx = RunContext()

    brief = synthesize(client, ctx, "kv cache?", PAPERS, CLAIMS)

    assert brief == GROUNDED
    assert len(fake.calls) == 2
    repair_prompt = fake.calls[1]["messages"][0]["content"]  # type: ignore[index]
    assert "1234.56789" in repair_prompt
    # Both calls are charged on the trace.
    assert len([r for r in ctx.trace if r.step == "synthesizer"]) == 2


def test_still_invented_after_repair_raises() -> None:
    client, _ = make_bedrock_client(
        [make_message(INVENTED, model=SONNET), make_message(INVENTED, model=SONNET)]
    )
    with pytest.raises(UngroundedCitations, match=r"1234\.56789"):
        synthesize(client, RunContext(), "kv cache?", PAPERS, CLAIMS)


def test_grouped_citations_cannot_bypass_grounding() -> None:
    # The first live landscape grouped ids inside one bracket ([a, b]); a single-id pattern
    # would treat the whole group as non-citation text and let invented ids sail through.
    from atlas_agents.steps.synthesizer import invented_citations

    text = "Grouped [2507.00001, 9999.99999] and old-style [cs/0112017] and prose [see above]."
    assert invented_citations(text, known={"2507.00001", "cs/0112017"}) == {"9999.99999"}
    assert invented_citations("versioned [2507.00001v2] never matches", known=set()) == set()
