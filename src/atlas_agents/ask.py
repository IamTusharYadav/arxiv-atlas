from dataclasses import dataclass, field

from pydantic import BaseModel

from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import (
    BUDGET_USD,
    MAX_ITERS,
    AgentError,
    BudgetExceeded,
    IterationsExhausted,
    RunContext,
    StepRecord,
    StepSink,
    run_loop,
)
from atlas_agents.prompts import CHECK
from atlas_agents.steps.evidence import neutralize
from atlas_agents.steps.extractor import PaperClaims, extract
from atlas_agents.steps.planner import MAX_SUBQUERIES, Plan, plan_query
from atlas_agents.steps.reranker import rerank
from atlas_agents.steps.retriever import PER_QUERY, retrieve
from atlas_agents.steps.synthesizer import synthesize
from atlas_agents.tracing import query_trace
from atlas_core.embedding import Embedder
from atlas_core.vectorstore import ScoredPaper, VectorStore

SCOPE_PREFIX = "This corpus covers arXiv abstracts in cs.AI, cs.LG and cs.CL only."
PARTIAL_NOTE = (
    "**This run stopped early, so this is evidence rather than a finished brief.** "
    "The gathered findings are listed below, each attributed to the paper it came from, "
    "but they were never synthesized into an argument."
)


class EvidenceCheck(BaseModel):
    sufficient: bool
    refined_subqueries: list[str] = []


@dataclass(frozen=True)
class Answer:
    brief: str
    papers: list[ScoredPaper]
    trace: list[StepRecord]
    cost_usd: float
    # True when a cap stopped the run and the brief is locally assembled evidence rather than
    # a synthesized answer. Callers must not cache a partial as if it were the real thing.
    partial: bool = False


@dataclass
class _State:
    plan: Plan | None = None
    subqueries: list[str] = field(default_factory=list)
    papers: dict[str, ScoredPaper] = field(default_factory=dict)
    claims: dict[str, PaperClaims] = field(default_factory=dict)


class _FanOutSink:
    def __init__(self, sinks: list[StepSink]) -> None:
        self._sinks = sinks

    def step(self, record: StepRecord, *, model: str | None, version: str | None) -> None:
        for sink in self._sinks:
            sink.step(record, model=model, version=version)


def ask(
    question: str,
    *,
    client: BedrockClient,
    store: VectorStore,
    embedder: Embedder,
    max_iters: int = MAX_ITERS,
    budget_usd: float = BUDGET_USD,
    sink: StepSink | None = None,
) -> Answer:
    state = _State()

    def task(ctx: RunContext) -> str | None:
        if state.plan is None:
            state.plan = plan_query(client, ctx, question)
            if not state.plan.in_scope:
                return f"{SCOPE_PREFIX} {state.plan.scope_note}".strip()
            state.subqueries = state.plan.subqueries
        plan = state.plan
        last_round = ctx.iteration >= max_iters - 1

        # Widen the net a little each round so empty rounds are not identical retries.
        candidates = retrieve(
            store, embedder, ctx, state.subqueries, per_query=PER_QUERY * (ctx.iteration + 1)
        )
        kept = rerank(client, ctx, question, candidates)
        for scored in kept:
            state.papers.setdefault(scored.paper.arxiv_id, scored)
        for claims in extract(client, ctx, question, kept):
            state.claims.setdefault(claims.arxiv_id, claims)

        evidence = list(state.claims.values())
        if not evidence:
            if last_round:
                return (
                    f"{SCOPE_PREFIX} No papers in it substantively address this "
                    f"question (searched: {', '.join(state.subqueries)})."
                )
            return None
        if not last_round:
            check = _check_evidence(client, ctx, plan.stop_criterion, evidence)
            if not check.sufficient:
                state.subqueries = check.refined_subqueries or plan.subqueries
                return None
        return synthesize(client, ctx, question, list(state.papers.values()), evidence)

    with query_trace(question) as trace_sink:
        loop_sink: StepSink = trace_sink if sink is None else _FanOutSink([trace_sink, sink])
        try:
            brief, ctx = run_loop(task, max_iters=max_iters, budget_usd=budget_usd, sink=loop_sink)
        except (BudgetExceeded, IterationsExhausted) as exc:
            # A cap stopped the loop, but the evidence it already paid for is still good.
            # Hand it back instead of throwing the run away. Only these two: an ungrounded
            # brief or a schema failure means the output itself is untrustworthy, so those
            # stay hard errors.
            partial = _partial_answer(exc, state)
            if partial is None:
                raise
            trace_sink.set_output(partial.brief)
            return partial
        trace_sink.set_output(brief)
    cited = [state.papers[c.arxiv_id] for c in state.claims.values()]
    return Answer(brief=brief, papers=cited, trace=ctx.trace, cost_usd=ctx.spent_usd)


def _partial_answer(exc: AgentError, state: _State) -> Answer | None:
    """Evidence gathered before a cap fired, assembled locally. No model call: the budget is
    exactly what ran out. None when nothing was gathered, where a bare failure is honest."""
    claims = [c for c in state.claims.values() if c.arxiv_id in state.papers]
    if not claims:
        return None
    sections = [PARTIAL_NOTE]
    for c in claims:
        title = state.papers[c.arxiv_id].paper.title
        findings = "\n".join(f"- {claim}" for claim in c.claims)
        sections.append(f"### {title} [{c.arxiv_id}]\n{findings}")
    return Answer(
        brief="\n\n".join(sections),
        papers=[state.papers[c.arxiv_id] for c in claims],
        trace=exc.trace,
        cost_usd=exc.spent_usd,
        partial=True,
    )


def _check_evidence(
    client: BedrockClient, ctx: RunContext, stop_criterion: str, claims: list[PaperClaims]
) -> EvidenceCheck:
    # Claims are extractor output derived from untrusted abstracts, so escape them here the
    # same way the synthesizer's paper_block does; laundering through one model call does not
    # make the text trusted.
    listing = "\n".join(f"- [{c.arxiv_id}] " + neutralize("; ".join(c.claims)) for c in claims)
    check, completion = client.complete_structured(
        model=CHECK.model,
        system=CHECK.render(max_subqueries=MAX_SUBQUERIES),
        prompt=f"<criterion>{stop_criterion}</criterion>\n\n<evidence>\n{listing}\n</evidence>",
        output_type=EvidenceCheck,
        max_tokens=300,
    )
    ctx.record(
        "check",
        "sufficient" if check.sufficient else "needs more evidence",
        completion,
        model=CHECK.model,
        version=CHECK.version,
    )
    return check
