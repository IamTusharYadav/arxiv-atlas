import hashlib
import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from atlas_agents.ask import ask
from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import BudgetExceeded, IterationsExhausted
from atlas_agents.steps.synthesizer import UngroundedCitations
from atlas_api.jobs import JobStore
from atlas_core.budget import DailyBudgetExceeded, DailyBudgetGuard
from atlas_core.cache import ResponseCache
from atlas_core.embedding import QUERY_PREFIX, Embedder
from atlas_core.ratelimit import RateLimiter
from atlas_core.vectorstore import VectorStore

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    # Bound the question: unbounded input is unbounded Bedrock spend.
    question: str = Field(min_length=1, max_length=500)


class PaperOut(BaseModel):
    arxiv_id: str
    title: str
    primary_category: str
    score: float


class TraceStep(BaseModel):
    step: str
    summary: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class QueryResponse(BaseModel):
    brief: str
    papers: list[PaperOut]
    trace: list[TraceStep]
    cost_usd: float
    cached: bool = False


class QueryAccepted(BaseModel):
    job_id: str
    status: str


class QueryStatus(BaseModel):
    job_id: str
    status: str
    result: QueryResponse | None = None
    error: str | None = None


class GraphNode(BaseModel):
    arxiv_id: str
    title: str
    primary_category: str


class GraphLink(BaseModel):
    source: str
    target: str
    weight: float


class GraphResponse(BaseModel):
    center: str
    nodes: list[GraphNode]
    links: list[GraphLink]


class StatusResponse(BaseModel):
    status: str
    corpus_size: int


def _client_key(request: Request) -> str:
    # Behind API Gateway the caller is the first X-Forwarded-For hop; hash it so raw IPs never
    # reach the bucket store or logs.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    elif request.client is not None:
        ip = request.client.host
    else:
        ip = "unknown"
    return hashlib.sha256(ip.encode()).hexdigest()


def _answer(
    question: str,
    *,
    client: BedrockClient,
    store: VectorStore,
    embedder: Embedder,
    cache: ResponseCache | None,
    budget: DailyBudgetGuard | None,
) -> QueryResponse:
    answer = ask(question, client=client, store=store, embedder=embedder)
    response = QueryResponse(
        brief=answer.brief,
        papers=[
            PaperOut(
                arxiv_id=s.paper.arxiv_id,
                title=s.paper.title,
                primary_category=s.paper.primary_category,
                score=s.score,
            )
            for s in answer.papers
        ],
        trace=[TraceStep.model_validate(r, from_attributes=True) for r in answer.trace],
        cost_usd=answer.cost_usd,
    )
    if budget is not None:
        # ponytail: only successful runs are charged; a run that aborts on a cap still spent.
        budget.charge(answer.cost_usd)
    if cache is not None:
        cache.put(embedder.embed([QUERY_PREFIX + question])[0], response.model_dump())
    return response


def run_job(
    job_id: str,
    *,
    jobs: JobStore,
    client: BedrockClient,
    store: VectorStore,
    embedder: Embedder,
    cache: ResponseCache | None,
    budget: DailyBudgetGuard | None,
) -> None:
    # Runs out of band from the request, so every outcome (success or failure) is written to the
    # job store rather than raised.
    job = jobs.get(job_id)
    if job is None:
        return
    jobs.mark_running(job_id)
    try:
        response = _answer(
            job.question, client=client, store=store, embedder=embedder, cache=cache, budget=budget
        )
    except (BudgetExceeded, IterationsExhausted, UngroundedCitations) as exc:
        jobs.fail(job_id, str(exc))
        return
    except Exception:
        log.exception("query job %s failed", job_id)
        jobs.fail(job_id, "internal error")
        return
    jobs.finish(job_id, response.model_dump())


def create_app(
    *,
    store: VectorStore,
    embedder: Embedder,
    client: BedrockClient,
    limiter: RateLimiter | None = None,
    cache: ResponseCache | None = None,
    budget: DailyBudgetGuard | None = None,
    jobs: JobStore | None = None,
    dispatch: Callable[[str], None] | None = None,
) -> FastAPI:
    app = FastAPI(title="ArXiv Atlas API", version="0.1.0")

    if limiter is not None:
        rl = limiter  # bound non-optional so the closure below type-checks

        @app.middleware("http")
        async def rate_limit(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            retry_after = rl.allow(_client_key(request))
            if retry_after:
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
            return await call_next(request)

    @app.post("/api/query")
    def query(req: QueryRequest, response: Response) -> QueryResponse | QueryAccepted:
        # Order: rate-limit (middleware) -> cache -> budget -> agent. A cache hit is free, so it
        # answers inline and never enqueues a job.
        if cache is not None:
            hit = cache.get(embedder.embed([QUERY_PREFIX + req.question])[0])
            if hit is not None:
                return QueryResponse.model_validate(hit).model_copy(update={"cached": True})
        if budget is not None:
            try:
                budget.check()
            except DailyBudgetExceeded as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if jobs is None or dispatch is None:
            # Synchronous fallback (local dev, tests): the loop is fast against scripted Bedrock.
            try:
                return _answer(
                    req.question,
                    client=client,
                    store=store,
                    embedder=embedder,
                    cache=cache,
                    budget=budget,
                )
            except (BudgetExceeded, IterationsExhausted, UngroundedCitations) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        # The real loop outruns API Gateway's 30s limit, so run it in a background invocation and
        # hand back a job id to poll.
        job_id = jobs.create(req.question)
        dispatch(job_id)
        response.status_code = 202
        return QueryAccepted(job_id=job_id, status="pending")

    @app.get("/api/query/{job_id}")
    def query_status(job_id: str) -> QueryStatus:
        if jobs is None:
            raise HTTPException(status_code=404, detail="async jobs are not enabled")
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id} not found")
        result = QueryResponse.model_validate(job.result) if job.result is not None else None
        return QueryStatus(job_id=job.id, status=job.status, result=result, error=job.error)

    @app.get("/api/graph/{arxiv_id}")
    def graph(arxiv_id: str) -> GraphResponse:
        found = store.get([arxiv_id])
        if not found:
            raise HTTPException(status_code=404, detail=f"paper {arxiv_id} not found")
        center = found[0]
        # Outgoing edges only; incoming would need a reverse index or the full adjacency artifact.
        neighbors = {s.paper.arxiv_id: s.paper for s in store.get([e.target for e in center.edges])}
        nodes = [
            GraphNode(
                arxiv_id=center.paper.arxiv_id,
                title=center.paper.title,
                primary_category=center.paper.primary_category,
            )
        ]
        links = []
        for edge in center.edges:
            paper = neighbors.get(edge.target)
            if paper is None:
                continue  # neighbor no longer in the corpus; skip the dangling link
            nodes.append(
                GraphNode(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    primary_category=paper.primary_category,
                )
            )
            links.append(GraphLink(source=arxiv_id, target=edge.target, weight=edge.weight))
        return GraphResponse(center=arxiv_id, nodes=nodes, links=links)

    @app.get("/api/status")
    def status() -> StatusResponse:
        return StatusResponse(status="ok", corpus_size=store.count())

    return app
