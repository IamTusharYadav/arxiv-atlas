"""Topic-in, structured-landscape-out (ADR 0003). A linear pipeline, not the iterative ask()
loop: plan, retrieve wide, cluster the stored embeddings (free), name each cluster with Haiku,
compute the activity timeline (free), then one Sonnet synthesis grounded in retrieved ids."""

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from atlas_agents.ask import SCOPE_PREFIX, _FanOutSink
from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import RunContext, StepRecord, StepSink, run_loop
from atlas_agents.prompts import LANDSCAPE
from atlas_agents.steps.directions import label_direction
from atlas_agents.steps.evidence import neutralize, paper_block
from atlas_agents.steps.planner import plan_query
from atlas_agents.steps.retriever import retrieve
from atlas_agents.steps.synthesizer import UngroundedCitations, invented_citations
from atlas_agents.tracing import query_trace
from atlas_core.cluster import central_order, kmeans, pick_k
from atlas_core.embedding import Embedder
from atlas_core.models import Paper
from atlas_core.vectorstore import VectorStore

# One landscape is k+1 Haiku-class calls plus one Sonnet synthesis; measured runs should sit
# well under this. Wider than the ask() cap because the output replaces hours, not minutes.
LANDSCAPE_BUDGET_USD = 0.30
PER_SUBQUERY = 25
# Below this, clusters are noise and the honest answer is "ask it as a question instead".
MIN_PAPERS = 6
# Papers shown to the direction card (most-central-first) and per direction to the synthesis.
CLUSTER_SAMPLE = 8
SYNTH_SAMPLE = 4
_SYNTH_ABSTRACT_CHARS = 400


class ReadingStep(BaseModel):
    arxiv_id: str
    reason: str


class LandscapeSynthesis(BaseModel):
    overview: str
    key_ideas: list[str]
    reading_order: list[ReadingStep]
    open_problems: list[str]


@dataclass(frozen=True)
class DirectionSummary:
    name: str
    problem: str
    papers: list[Paper]  # every cluster member, most-central-first
    representative_ids: list[str]


@dataclass(frozen=True)
class TimelinePoint:
    month: str  # "YYYY-MM"
    direction: str
    count: int


@dataclass(frozen=True)
class Landscape:
    topic: str
    overview: str
    key_ideas: list[str]
    directions: list[DirectionSummary]
    timeline: list[TimelinePoint]
    reading_order: list[ReadingStep]
    open_problems: list[str]
    trace: list[StepRecord]
    cost_usd: float
    # True when the pipeline stopped before mapping: out of scope, or too few papers.
    # The overview carries the explanation and every list is empty.
    declined: bool = False


def map_topic(
    topic: str,
    *,
    client: BedrockClient,
    store: VectorStore,
    embedder: Embedder,
    budget_usd: float = LANDSCAPE_BUDGET_USD,
    sink: StepSink | None = None,
) -> Landscape:
    with query_trace(topic) as trace_sink:
        loop_sink: StepSink = trace_sink if sink is None else _FanOutSink([trace_sink, sink])

        def task(ctx: RunContext) -> Landscape:
            return _build(topic, ctx, client=client, store=store, embedder=embedder)

        # One pass through run_loop keeps the budget/trace semantics identical to ask();
        # the task never returns None, so the single iteration always suffices.
        landscape, _ctx = run_loop(task, max_iters=1, budget_usd=budget_usd, sink=loop_sink)
        trace_sink.set_output(landscape.overview)
    return landscape


def _build(
    topic: str,
    ctx: RunContext,
    *,
    client: BedrockClient,
    store: VectorStore,
    embedder: Embedder,
) -> Landscape:
    plan = plan_query(client, ctx, topic)
    if not plan.in_scope:
        return _declined(topic, f"{SCOPE_PREFIX} {plan.scope_note}".strip(), ctx)

    hits = retrieve(store, embedder, ctx, plan.subqueries, per_query=PER_SUBQUERY)
    papers = {s.paper.arxiv_id: s.paper for s in hits}  # best-score-first insertion order
    vectors = store.get_vectors(list(papers))
    ids = [i for i in papers if i in vectors]
    if len(ids) < MIN_PAPERS:
        return _declined(
            topic,
            f"{SCOPE_PREFIX} Only {len(ids)} recent papers match this topic, too few to map "
            "a landscape; try asking it as a question instead.",
            ctx,
        )

    matrix = np.array([vectors[i] for i in ids], dtype=np.float32)
    k = pick_k(len(ids))
    labels, centroids = kmeans(matrix, k)
    ctx.record("cluster", f"{len(ids)} papers grouped into {k} candidate directions")

    directions: list[DirectionSummary] = []
    by_size = sorted(range(k), key=lambda j: int((labels == j).sum()), reverse=True)
    for j in by_size:
        members = np.flatnonzero(labels == j)
        if len(members) < 2:  # a singleton is a stray paper, not a direction
            continue
        ordered = [ids[int(members[o])] for o in central_order(matrix[members], centroids[j])]
        cluster_papers = [papers[i] for i in ordered]
        named = label_direction(client, ctx, topic, cluster_papers[:CLUSTER_SAMPLE])
        directions.append(
            DirectionSummary(
                name=named.name,
                problem=named.problem,
                papers=cluster_papers,
                representative_ids=named.representative_ids,
            )
        )
    if not directions:
        return _declined(
            topic,
            f"{SCOPE_PREFIX} The matching papers are too scattered to form research "
            "directions; try a broader topic or ask a question instead.",
            ctx,
        )

    # Only direction members are citable: the synthesis prompt shows nothing else, and a
    # citation the UI cannot resolve to a visible paper would be a dead reference.
    visible = {p.arxiv_id for d in directions for p in d.papers}
    synthesis = _synthesize(client, ctx, topic, directions, known=visible)
    reading = [r for r in synthesis.reading_order if r.arxiv_id in visible]
    return Landscape(
        topic=topic,
        overview=synthesis.overview,
        key_ideas=synthesis.key_ideas,
        directions=directions,
        timeline=_timeline(directions),
        reading_order=reading,
        open_problems=synthesis.open_problems,
        trace=ctx.trace,
        cost_usd=ctx.spent_usd,
    )


def _declined(topic: str, note: str, ctx: RunContext) -> Landscape:
    return Landscape(
        topic=topic,
        overview=note,
        key_ideas=[],
        directions=[],
        timeline=[],
        reading_order=[],
        open_problems=[],
        trace=ctx.trace,
        cost_usd=ctx.spent_usd,
        declined=True,
    )


def _timeline(directions: list[DirectionSummary]) -> list[TimelinePoint]:
    counts: dict[tuple[str, str], int] = {}
    for direction in directions:
        for paper in direction.papers:
            key = (paper.published_at.strftime("%Y-%m"), direction.name)
            counts[key] = counts.get(key, 0) + 1
    return [
        TimelinePoint(month=month, direction=direction, count=count)
        for (month, direction), count in sorted(counts.items())
    ]


def _synthesize(
    client: BedrockClient,
    ctx: RunContext,
    topic: str,
    directions: list[DirectionSummary],
    known: set[str],
) -> LandscapeSynthesis:
    blocks = []
    for d in directions:
        by_id = {p.arxiv_id: p for p in d.papers}
        reps = [by_id[i] for i in d.representative_ids if i in by_id]
        sample = (reps or d.papers)[:SYNTH_SAMPLE]
        papers = "\n".join(
            paper_block(p.arxiv_id, p.title, p.abstract, _SYNTH_ABSTRACT_CHARS) for p in sample
        )
        # Direction names came out of a model reading untrusted abstracts; escape them like
        # every other laundered text.
        blocks.append(
            f"<direction name={neutralize(d.name)!r} papers={len(d.papers)}>\n"
            f"{neutralize(d.problem)}\n{papers}\n</direction>"
        )
    prompt = f"<topic>{neutralize(topic)}</topic>\n\n" + "\n\n".join(blocks)

    synthesis, completion = client.complete_structured(
        model=LANDSCAPE.model,
        system=LANDSCAPE.system,
        prompt=prompt,
        output_type=LandscapeSynthesis,
        max_tokens=2500,
    )
    invented = invented_citations(_cited_text(synthesis), known)
    if invented:
        synthesis, repair = client.complete_structured(
            model=LANDSCAPE.model,
            system=LANDSCAPE.system,
            prompt=f"{prompt}\n\nYour previous draft cited ids that are not in the evidence: "
            f"{sorted(invented)}. Rewrite citing only provided ids.",
            output_type=LandscapeSynthesis,
            max_tokens=2500,
        )
        ctx.record(
            "landscape",
            "repaired invented citations",
            completion,
            model=LANDSCAPE.model,
            version=LANDSCAPE.version,
        )
        completion = repair
        invented = invented_citations(_cited_text(synthesis), known)
        if invented:
            ctx.record(
                "landscape",
                f"ungrounded citations: {sorted(invented)}",
                completion,
                model=LANDSCAPE.model,
                version=LANDSCAPE.version,
            )
            raise UngroundedCitations(
                f"landscape cites unknown ids after repair: {sorted(invented)}",
                spent_usd=ctx.spent_usd,
                trace=list(ctx.trace),
            )
    ctx.record(
        "landscape",
        f"{len(synthesis.key_ideas)} key ideas, {len(synthesis.reading_order)} reading steps",
        completion,
        model=LANDSCAPE.model,
        version=LANDSCAPE.version,
    )
    return synthesis


def _cited_text(synthesis: LandscapeSynthesis) -> str:
    return "\n".join(
        [synthesis.overview, *synthesis.key_ideas, *synthesis.open_problems]
        + [r.reason for r in synthesis.reading_order]
    )
