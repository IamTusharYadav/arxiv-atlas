"""ask(): the five steps wired into the harness loop.

One iteration = one evidence round: retrieve, rerank, extract, then a cheap Haiku
check of the evidence against the planner's stop criterion. Insufficient evidence
refines the subqueries and loops; sufficient evidence hands off to the synthesizer.
Out-of-scope questions and empty corpora decline without spending Sonnet tokens.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel

from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import BUDGET_USD, MAX_ITERS, RunContext, StepRecord, run_loop
from atlas_agents.prompts import CHECK
from atlas_agents.steps.extractor import PaperClaims, extract
from atlas_agents.steps.planner import MAX_SUBQUERIES, Plan, plan_query
from atlas_agents.steps.reranker import rerank
from atlas_agents.steps.retriever import PER_QUERY, retrieve
from atlas_agents.steps.synthesizer import synthesize
from atlas_agents.tracing import query_trace
from atlas_core.embedding import Embedder
from atlas_core.vectorstore import ScoredPaper, VectorStore

SCOPE_PREFIX = "This corpus covers arXiv abstracts in cs.AI, cs.LG and cs.CL only."


class EvidenceCheck(BaseModel):
    sufficient: bool
    refined_subqueries: list[str] = []


@dataclass(frozen=True)
class Answer:
    brief: str
    papers: list[ScoredPaper]
    trace: list[StepRecord]
    cost_usd: float


@dataclass
class _State:
    plan: Plan | None = None
    subqueries: list[str] = field(default_factory=list)
    papers: dict[str, ScoredPaper] = field(default_factory=dict)
    claims: dict[str, PaperClaims] = field(default_factory=dict)


def ask(
    question: str,
    *,
    client: BedrockClient,
    store: VectorStore,
    embedder: Embedder,
    max_iters: int = MAX_ITERS,
    budget_usd: float = BUDGET_USD,
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

    with query_trace(question) as sink:
        brief, ctx = run_loop(task, max_iters=max_iters, budget_usd=budget_usd, sink=sink)
        sink.set_output(brief)
    cited = [state.papers[c.arxiv_id] for c in state.claims.values()]
    return Answer(brief=brief, papers=cited, trace=ctx.trace, cost_usd=ctx.spent_usd)


def _check_evidence(
    client: BedrockClient, ctx: RunContext, stop_criterion: str, claims: list[PaperClaims]
) -> EvidenceCheck:
    listing = "\n".join(f"- [{c.arxiv_id}] " + "; ".join(c.claims) for c in claims)
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
