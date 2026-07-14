import hashlib
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from atlas_agents.ask import ask
from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import BudgetExceeded, IterationsExhausted
from atlas_agents.steps.synthesizer import UngroundedCitations
from atlas_core.budget import DailyBudgetExceeded, DailyBudgetGuard
from atlas_core.cache import ResponseCache
from atlas_core.embedding import QUERY_PREFIX, Embedder
from atlas_core.ratelimit import RateLimiter
from atlas_core.vectorstore import VectorStore


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


def create_app(
    *,
    store: VectorStore,
    embedder: Embedder,
    client: BedrockClient,
    limiter: RateLimiter | None = None,
    cache: ResponseCache | None = None,
    budget: DailyBudgetGuard | None = None,
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
    def query(req: QueryRequest) -> QueryResponse:
        # Order: rate-limit (middleware) -> cache -> budget -> agent. A cache hit is free to
        # serve, so it runs before the budget gate.
        embedding = None
        if cache is not None:
            embedding = embedder.embed([QUERY_PREFIX + req.question])[0]
            hit = cache.get(embedding)
            if hit is not None:
                return QueryResponse.model_validate(hit).model_copy(update={"cached": True})
        if budget is not None:
            try:
                budget.check()
            except DailyBudgetExceeded as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            answer = ask(req.question, client=client, store=store, embedder=embedder)
        except (BudgetExceeded, IterationsExhausted, UngroundedCitations) as exc:
            # The loop hitting its caps or failing to ground a brief, not bad input: 503, not 500.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
            # Upgrade path: surface ctx.spent_usd on the terminal exceptions and charge here too.
            budget.charge(answer.cost_usd)
        if cache is not None and embedding is not None:
            cache.put(embedding, response.model_dump())
        return response

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
