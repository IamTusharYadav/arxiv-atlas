import hashlib
import logging
from collections.abc import Awaitable, Callable
from time import time

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from atlas_agents.ask import ask
from atlas_agents.bedrock import BedrockClient, StructuredOutputError
from atlas_agents.harness import AgentError, StepRecord, StepSink
from atlas_agents.landscape import Landscape, inner_links, map_topic
from atlas_api.jobs import Job, JobStore
from atlas_api.v1 import register_v1
from atlas_core.budget import DailyBudgetExceeded, DailyBudgetGuard
from atlas_core.cache import ResponseCache
from atlas_core.embedding import QUERY_PREFIX, Embedder
from atlas_core.ratelimit import RateLimiter
from atlas_core.vectorstore import VectorStore

log = logging.getLogger(__name__)

# The async worker's own Lambda timeout: a job silent this long made zero progress through a
# full worker lifetime (progress writes bump updated_at every step, and the crash-retry's
# mark_running bumps it again), so it is dead, not slow.
STALE_JOB_S = 900


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


class QueryResponse(BaseModel):
    brief: str
    papers: list[PaperOut]
    # Semantic-similarity edges among the cited papers only, for the citation graph. Same
    # source as the landscape map (stored edges, never citations); empty on older cache rows.
    links: list[GraphLink] = []
    trace: list[TraceStep]
    cost_usd: float
    cached: bool = False
    # A cap stopped the run: the brief is gathered evidence, not a synthesized answer.
    partial: bool = False


class QueryAccepted(BaseModel):
    job_id: str
    status: str


class QueryStatus(BaseModel):
    job_id: str
    status: str
    progress: list[dict[str, str]] = []
    result: QueryResponse | None = None
    error: str | None = None


class LandscapeRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)


class LandscapePaper(BaseModel):
    arxiv_id: str
    title: str
    primary_category: str
    published_month: str  # "YYYY-MM", enough for the activity timeline drill-down


class DirectionOut(BaseModel):
    name: str
    problem: str
    papers: list[LandscapePaper]  # most-central-first
    representative_ids: list[str]


class TimelinePointOut(BaseModel):
    month: str
    direction: str
    count: int


class ReadingStepOut(BaseModel):
    arxiv_id: str
    title: str
    reason: str


class LandscapeResponse(BaseModel):
    topic: str
    overview: str
    key_ideas: list[str]
    directions: list[DirectionOut]
    timeline: list[TimelinePointOut]
    reading_order: list[ReadingStepOut]
    open_problems: list[str]
    # Semantic-similarity edges between the landscape's own papers, for the topic map.
    links: list[GraphLink]
    trace: list[TraceStep]
    cost_usd: float
    cached: bool = False
    declined: bool = False


class LandscapeStatus(BaseModel):
    job_id: str
    status: str
    progress: list[dict[str, str]] = []
    result: LandscapeResponse | None = None
    error: str | None = None


class StatusResponse(BaseModel):
    status: str
    corpus_size: int


def _client_key(request: Request) -> str:
    # Mangum fills request.client from API Gateway's requestContext sourceIp, which the caller
    # cannot forge. X-Forwarded-For is caller-influenced (API Gateway appends the real IP to
    # whatever the client sent, so its first hop is attacker-chosen and would mint a fresh
    # bucket per request); it is only a fallback for when there is no client at all. The IP is
    # hashed so raw addresses never reach the bucket store or logs.
    if request.client is not None:
        ip = request.client.host
    else:
        forwarded = request.headers.get("x-forwarded-for")
        ip = forwarded.split(",")[0].strip() if forwarded else "unknown"
    return hashlib.sha256(ip.encode()).hexdigest()


class _JobProgressSink:
    def __init__(self, jobs: JobStore, job_id: str) -> None:
        self._jobs = jobs
        self._job_id = job_id
        self._steps: list[dict[str, str]] = []

    def step(self, record: StepRecord, *, model: str | None, version: str | None) -> None:
        self._steps.append({"step": record.step, "summary": record.summary})
        try:
            self._jobs.set_progress(self._job_id, list(self._steps))
        except Exception:
            # Progress is best-effort; a failed update must not sink the run.
            log.debug("progress update failed for job %s", self._job_id)


def _answer(
    question: str,
    *,
    client: BedrockClient,
    store: VectorStore,
    embedder: Embedder,
    cache: ResponseCache | None,
    budget: DailyBudgetGuard | None,
    sink: StepSink | None = None,
) -> QueryResponse:
    try:
        answer = ask(question, client=client, store=store, embedder=embedder, sink=sink)
    except AgentError as exc:
        # An aborted run still spent real money; charge it so the daily counter stays honest.
        # charge() swallows backend errors, so this can never mask the original failure.
        if budget is not None:
            budget.charge(exc.spent_usd)
        raise
    # Relationships among the cited papers only, from the same stored similarity edges the
    # landscape map uses; nothing outside the citation set is pulled in.
    cited_ids = {s.paper.arxiv_id for s in answer.papers}
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
        links=[
            GraphLink.model_validate(e, from_attributes=True) for e in inner_links(store, cited_ids)
        ],
        trace=[TraceStep.model_validate(r, from_attributes=True) for r in answer.trace],
        cost_usd=answer.cost_usd,
        partial=answer.partial,
    )
    if budget is not None:
        budget.charge(answer.cost_usd)
    # Never cache a partial: it would serve a capped run's leftovers to everyone asking that
    # question for the whole TTL, long after the budget recovered.
    if cache is not None and not response.partial:
        cache.put(embedder.embed([QUERY_PREFIX + question])[0], response.model_dump())
    return response


def _serialize_landscape(landscape: Landscape) -> LandscapeResponse:
    titles = {p.arxiv_id: p.title for d in landscape.directions for p in d.papers}
    return LandscapeResponse(
        topic=landscape.topic,
        overview=landscape.overview,
        key_ideas=landscape.key_ideas,
        directions=[
            DirectionOut(
                name=d.name,
                problem=d.problem,
                papers=[
                    LandscapePaper(
                        arxiv_id=p.arxiv_id,
                        title=p.title,
                        primary_category=p.primary_category,
                        published_month=p.published_at.strftime("%Y-%m"),
                    )
                    for p in d.papers
                ],
                representative_ids=d.representative_ids,
            )
            for d in landscape.directions
        ],
        timeline=[
            TimelinePointOut.model_validate(t, from_attributes=True) for t in landscape.timeline
        ],
        reading_order=[
            ReadingStepOut(arxiv_id=r.arxiv_id, title=titles.get(r.arxiv_id, ""), reason=r.reason)
            for r in landscape.reading_order
        ],
        open_problems=landscape.open_problems,
        links=[GraphLink.model_validate(e, from_attributes=True) for e in landscape.links],
        trace=[TraceStep.model_validate(r, from_attributes=True) for r in landscape.trace],
        cost_usd=landscape.cost_usd,
        declined=landscape.declined,
    )


def _landscape(
    topic: str,
    *,
    client: BedrockClient,
    store: VectorStore,
    embedder: Embedder,
    cache: ResponseCache | None,
    budget: DailyBudgetGuard | None,
    sink: StepSink | None = None,
) -> LandscapeResponse:
    try:
        landscape = map_topic(topic, client=client, store=store, embedder=embedder, sink=sink)
    except AgentError as exc:
        if budget is not None:
            budget.charge(exc.spent_usd)
        raise
    response = _serialize_landscape(landscape)
    if budget is not None:
        budget.charge(landscape.cost_usd)
    # A decline costs one planner call; caching it would pin "no landscape" for the whole TTL
    # even after the corpus or phrasing would map fine.
    if cache is not None and not response.declined:
        cache.put(
            embedder.embed([QUERY_PREFIX + topic])[0], response.model_dump(), kind="landscape"
        )
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
    sink = _JobProgressSink(jobs, job_id)
    try:
        if job.kind == "landscape":
            result = _landscape(
                job.question,
                client=client,
                store=store,
                embedder=embedder,
                cache=cache,
                budget=budget,
                sink=sink,
            ).model_dump()
        else:
            result = _answer(
                job.question,
                client=client,
                store=store,
                embedder=embedder,
                cache=cache,
                budget=budget,
                sink=sink,
            ).model_dump()
    except (AgentError, StructuredOutputError) as exc:
        jobs.fail(job_id, str(exc))
        return
    except Exception:
        log.exception("%s job %s failed", job.kind, job_id)
        jobs.fail(job_id, "internal error")
        return
    jobs.finish(job_id, result)


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
    app = FastAPI(title="ArXiv Atlas API", version="1.3.0")

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
            except (AgentError, StructuredOutputError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        # The real loop outruns API Gateway's 30s limit, so run it in a background invocation and
        # hand back a job id to poll.
        job_id = jobs.create(req.question)
        try:
            dispatch(job_id)
        except Exception as exc:
            # Mark the job failed rather than leave it stuck pending until TTL.
            log.exception("dispatch failed for job %s", job_id)
            jobs.fail(job_id, "could not start the query worker")
            raise HTTPException(status_code=503, detail="could not start the query worker") from exc
        response.status_code = 202
        return QueryAccepted(job_id=job_id, status="pending")

    def _fetch_job(job_id: str, kind: str) -> Job:
        # Kind is part of the identity: polling a landscape job on the query route (or vice
        # versa) would validate the wrong result shape, so it is simply not found here.
        if jobs is None:
            raise HTTPException(status_code=404, detail="async jobs are not enabled")
        job = jobs.get(job_id)
        if job is None or job.kind != kind:
            raise HTTPException(status_code=404, detail=f"job {job_id} not found")
        return job

    def _worker_died(job: Job) -> bool:
        # A worker that hard-crashed through its retry never writes a terminal status, which
        # would leave pollers on an eternal spinner until the row's TTL. The worker bumps
        # updated_at on every step, so silence longer than one whole worker lifetime (the
        # Lambda 900s ceiling) means it is dead. Read-only verdict: every poll computes the
        # same answer, and GETs never write.
        return (
            job.status in ("pending", "running")
            and job.updated_at is not None
            and time() - job.updated_at > STALE_JOB_S
        )

    @app.get("/api/query/{job_id}")
    def query_status(job_id: str) -> QueryStatus:
        job = _fetch_job(job_id, "query")
        if _worker_died(job):
            return QueryStatus(
                job_id=job.id,
                status="error",
                progress=job.progress,
                error="the query worker stopped responding; retry the question",
            )
        result = QueryResponse.model_validate(job.result) if job.result is not None else None
        return QueryStatus(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            result=result,
            error=job.error,
        )

    @app.post("/api/landscape")
    def landscape(req: LandscapeRequest, response: Response) -> LandscapeResponse | QueryAccepted:
        # Same order as the query route: rate-limit (middleware) -> cache -> budget -> agent.
        if cache is not None:
            hit = cache.get(embedder.embed([QUERY_PREFIX + req.topic])[0], kind="landscape")
            if hit is not None:
                return LandscapeResponse.model_validate(hit).model_copy(update={"cached": True})
        if budget is not None:
            try:
                budget.check()
            except DailyBudgetExceeded as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if jobs is None or dispatch is None:
            try:
                return _landscape(
                    req.topic,
                    client=client,
                    store=store,
                    embedder=embedder,
                    cache=cache,
                    budget=budget,
                )
            except (AgentError, StructuredOutputError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        job_id = jobs.create(req.topic, kind="landscape")
        try:
            dispatch(job_id)
        except Exception as exc:
            log.exception("dispatch failed for landscape job %s", job_id)
            jobs.fail(job_id, "could not start the landscape worker")
            raise HTTPException(
                status_code=503, detail="could not start the landscape worker"
            ) from exc
        response.status_code = 202
        return QueryAccepted(job_id=job_id, status="pending")

    @app.get("/api/landscape/{job_id}")
    def landscape_status(job_id: str) -> LandscapeStatus:
        job = _fetch_job(job_id, "landscape")
        if _worker_died(job):
            return LandscapeStatus(
                job_id=job.id,
                status="error",
                progress=job.progress,
                error="the landscape worker stopped responding; retry the topic",
            )
        result = LandscapeResponse.model_validate(job.result) if job.result is not None else None
        return LandscapeStatus(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            result=result,
            error=job.error,
        )

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

    # The versioned read-only surface the MCP package depends on (ADR 0004). Bedrock-free, so it
    # rides the same rate limiter but never the budget guard.
    register_v1(app, store=store, embedder=embedder)

    return app
