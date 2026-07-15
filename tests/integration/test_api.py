from dataclasses import replace
from time import time
from typing import Any

from anthropic.types import Message
from fastapi.testclient import TestClient

from atlas_agents.bedrock import SONNET
from atlas_api import create_app
from atlas_api.app import run_job
from atlas_api.jobs import Job
from atlas_core.budget import DailyBudgetGuard
from atlas_core.cache import ResponseCache
from atlas_core.models import Edge
from atlas_core.ratelimit import RateLimiter
from atlas_core.vectorstore import QdrantStore
from tests.conftest import FakeEmbedder, make_bedrock_client, make_message
from tests.unit.test_ask import IDS, check_json, extract_json, plan_json, rerank_json, seed
from tests.unit.test_budget import _FakeTable
from tests.unit.test_cache import _FakeStore
from tests.unit.test_ratelimit import MemoryBuckets


class MemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, question: str) -> str:
        job_id = f"job-{len(self._jobs)}"
        self._jobs[job_id] = Job(job_id, "pending", question, None, None)
        return job_id

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        self._jobs[job_id] = replace(self._jobs[job_id], status="running")

    def set_progress(self, job_id: str, progress: list[dict[str, str]]) -> None:
        self._jobs[job_id] = replace(self._jobs[job_id], progress=progress)

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        self._jobs[job_id] = replace(self._jobs[job_id], status="done", result=result)

    def fail(self, job_id: str, error: str) -> None:
        self._jobs[job_id] = replace(self._jobs[job_id], status="error", error=error)


def async_client(
    store: QdrantStore, messages: list[Message | Exception], jobs: MemoryJobStore
) -> TestClient:
    # dispatch runs the worker inline, so an enqueued job is already done when the POST returns.
    bedrock, _ = make_bedrock_client(messages)
    embedder = FakeEmbedder()

    def dispatch(job_id: str) -> None:
        run_job(
            job_id,
            jobs=jobs,
            client=bedrock,
            store=store,
            embedder=embedder,
            cache=None,
            budget=None,
        )

    app = create_app(store=store, embedder=embedder, client=bedrock, jobs=jobs, dispatch=dispatch)
    return TestClient(app)


def client_for(
    store: QdrantStore,
    messages: list[Message | Exception],
    *,
    limiter: RateLimiter | None = None,
    cache: ResponseCache | None = None,
    budget: DailyBudgetGuard | None = None,
) -> TestClient:
    bedrock, _ = make_bedrock_client(messages)
    return TestClient(
        create_app(
            store=store,
            embedder=FakeEmbedder(),
            client=bedrock,
            limiter=limiter,
            cache=cache,
            budget=budget,
        )
    )


def test_status_reports_corpus_size(memory_store: QdrantStore, fake_embedder: FakeEmbedder) -> None:
    seed(memory_store, fake_embedder)
    resp = client_for(memory_store, []).get("/api/status")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "corpus_size": 3}


def test_graph_returns_outgoing_neighborhood(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    memory_store.set_edges(
        IDS[0],
        [
            Edge(source=IDS[0], target=IDS[1], weight=0.8),
            Edge(source=IDS[0], target=IDS[2], weight=0.7),
        ],
    )
    body = client_for(memory_store, []).get(f"/api/graph/{IDS[0]}").json()

    assert body["center"] == IDS[0]
    assert {n["arxiv_id"] for n in body["nodes"]} == set(IDS)  # center + two neighbors
    assert {
        (link["source"], link["target"], round(link["weight"], 4)) for link in body["links"]
    } == {
        (IDS[0], IDS[1], 0.8),
        (IDS[0], IDS[2], 0.7),
    }


def test_graph_skips_links_to_pruned_neighbors(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    memory_store.set_edges(IDS[0], [Edge(source=IDS[0], target="9999.99999", weight=0.9)])
    body = client_for(memory_store, []).get(f"/api/graph/{IDS[0]}").json()
    assert [n["arxiv_id"] for n in body["nodes"]] == [IDS[0]]  # only the center resolves
    assert body["links"] == []


def test_graph_unknown_paper_404(memory_store: QdrantStore) -> None:
    assert client_for(memory_store, []).get("/api/graph/9999.99999").status_code == 404


def test_query_returns_brief_papers_and_trace(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    brief_md = f"Quantization leads [{IDS[0]}]."
    api = client_for(
        memory_store,
        [
            make_message(plan_json()),
            make_message(rerank_json({IDS[0]: 9, IDS[1]: 7, IDS[2]: 2})),
            make_message(extract_json([IDS[0], IDS[1]])),
            make_message(check_json(sufficient=True)),
            make_message(brief_md, model=SONNET),
        ],
    )

    resp = api.post("/api/query", json={"question": "kv cache?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["brief"] == brief_md
    assert {p["arxiv_id"] for p in body["papers"]} == {IDS[0], IDS[1]}
    assert body["trace"][0]["step"] == "planner"
    assert body["cost_usd"] > 0


def test_query_rejects_blank_question(memory_store: QdrantStore) -> None:
    assert client_for(memory_store, []).post("/api/query", json={"question": ""}).status_code == 422


def test_rate_limit_returns_429_with_retry_after(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    # burst=1: the first request spends the only token, the second is throttled. TestClient's
    # client address is constant, so every request shares one bucket.
    limiter = RateLimiter(MemoryBuckets(), rate_per_s=1 / 6, burst=1)
    api = client_for(memory_store, [], limiter=limiter)
    assert api.get("/api/status").status_code == 200
    resp = api.get("/api/status")
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) > 0


def test_rate_limit_ignores_spoofed_forwarded_header(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    # Behind API Gateway the first X-Forwarded-For hop is attacker-chosen, so rotating it must
    # not mint a fresh bucket per request; the unforgeable source address keys the bucket.
    limiter = RateLimiter(MemoryBuckets(), rate_per_s=1 / 6, burst=1)
    api = client_for(memory_store, [], limiter=limiter)
    assert api.get("/api/status", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert api.get("/api/status", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 429


def test_query_over_budget_returns_503(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    # A planner call charged past the per-query cap aborts the run; the route surfaces it as 503.
    api = client_for(memory_store, [make_message(plan_json(), output_tokens=5_000_000)])
    assert api.post("/api/query", json={"question": "kv cache?"}).status_code == 503


def test_query_cache_hit_skips_bedrock(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    brief_md = f"Cached brief [{IDS[0]}]."
    # Exactly one run's worth of scripted calls; a cache hit on the repeat spends none of them.
    api = client_for(
        memory_store,
        [
            make_message(plan_json()),
            make_message(rerank_json({IDS[0]: 9, IDS[1]: 7, IDS[2]: 2})),
            make_message(extract_json([IDS[0], IDS[1]])),
            make_message(check_json(sufficient=True)),
            make_message(brief_md, model=SONNET),
        ],
        cache=ResponseCache(_FakeStore()),
    )
    first = api.post("/api/query", json={"question": "kv cache?"})
    assert first.status_code == 200 and first.json()["cached"] is False

    second = api.post("/api/query", json={"question": "kv cache?"})
    assert second.status_code == 200
    body = second.json()
    assert body["cached"] is True
    assert body["brief"] == brief_md


def test_aborted_run_still_charges_daily_budget(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    # The planner call alone blows the per-query cap; that spend was real, so the daily
    # counter must see it even though the run aborted.
    table = _FakeTable()
    guard = DailyBudgetGuard(table, daily_cap_usd=50.0)
    api = client_for(
        memory_store, [make_message(plan_json(), output_tokens=5_000_000)], budget=guard
    )
    assert api.post("/api/query", json={"question": "kv cache?"}).status_code == 503
    assert float(table.spent) > 20  # ~5M output tokens of Haiku, charged despite the abort


def test_partial_answer_is_returned_but_never_cached(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    cache = ResponseCache(_FakeStore())
    # The check call blows the per-query cap after extraction, so the run returns gathered
    # evidence. Caching that would serve a capped run's leftovers for the whole TTL.
    api = client_for(
        memory_store,
        [
            make_message(plan_json()),
            make_message(rerank_json({IDS[0]: 9})),
            make_message(extract_json([IDS[0]])),
            make_message(check_json(sufficient=False), output_tokens=5_000_000),
        ],
        cache=cache,
    )
    body = api.post("/api/query", json={"question": "kv cache?"}).json()
    assert body["partial"] is True
    assert body["papers"][0]["arxiv_id"] == IDS[0]
    assert cache.get(fake_embedder.embed(["kv cache?"])[0]) is None  # nothing stored


def test_query_over_daily_budget_returns_503(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    guard = DailyBudgetGuard(_FakeTable(), daily_cap_usd=1.00)
    guard.charge(1.00)  # day already at cap, so the check denies before any Bedrock call
    api = client_for(memory_store, [], budget=guard)
    assert api.post("/api/query", json={"question": "kv cache?"}).status_code == 503


def test_async_query_enqueues_then_completes(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    brief_md = f"Async brief [{IDS[0]}]."
    api = async_client(
        memory_store,
        [
            make_message(plan_json()),
            make_message(rerank_json({IDS[0]: 9, IDS[1]: 7, IDS[2]: 2})),
            make_message(extract_json([IDS[0], IDS[1]])),
            make_message(check_json(sufficient=True)),
            make_message(brief_md, model=SONNET),
        ],
        MemoryJobStore(),
    )

    accepted = api.post("/api/query", json={"question": "kv cache?"})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]

    got = api.get(f"/api/query/{job_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["status"] == "done"
    assert body["result"]["brief"] == brief_md
    assert {p["arxiv_id"] for p in body["result"]["papers"]} == {IDS[0], IDS[1]}
    # progress accumulated per step as the worker ran (option C, surfaced via polling).
    steps = [s["step"] for s in body["progress"]]
    assert steps[0] == "planner" and steps[-1] == "synthesizer"


def test_query_dispatch_failure_marks_job_failed(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    jobs = MemoryJobStore()
    bedrock, _ = make_bedrock_client([])

    def dispatch(job_id: str) -> None:
        raise RuntimeError("invoke failed")

    api = TestClient(
        create_app(
            store=memory_store, embedder=fake_embedder, client=bedrock, jobs=jobs, dispatch=dispatch
        )
    )
    resp = api.post("/api/query", json={"question": "kv cache?"})
    assert resp.status_code == 503
    # the job was created then marked failed, not left stuck pending
    (job,) = jobs._jobs.values()
    assert job.status == "error"


def test_async_query_job_records_error(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    # A planner call charged past the per-query cap aborts the run inside the worker.
    api = async_client(
        memory_store, [make_message(plan_json(), output_tokens=5_000_000)], MemoryJobStore()
    )
    job_id = api.post("/api/query", json={"question": "kv cache?"}).json()["job_id"]
    body = api.get(f"/api/query/{job_id}").json()
    assert body["status"] == "error"
    assert body["result"] is None
    assert body["error"]


def test_query_status_reports_dead_worker_after_silence(memory_store: QdrantStore) -> None:
    jobs = MemoryJobStore()
    job_id = jobs.create("q")
    # A worker that hard-crashed through its retry never writes a terminal status; the route
    # reports it dead once silence exceeds a whole worker lifetime.
    jobs._jobs[job_id] = replace(jobs._jobs[job_id], status="running", updated_at=time() - 901)
    api = async_client(memory_store, [], jobs)
    body = api.get(f"/api/query/{job_id}").json()
    assert body["status"] == "error"
    assert "worker" in body["error"]


def test_query_status_recent_running_job_stays_running(memory_store: QdrantStore) -> None:
    jobs = MemoryJobStore()
    job_id = jobs.create("q")
    jobs._jobs[job_id] = replace(jobs._jobs[job_id], status="running", updated_at=time() - 30)
    api = async_client(memory_store, [], jobs)
    assert api.get(f"/api/query/{job_id}").json()["status"] == "running"


def test_query_status_unknown_job_404(memory_store: QdrantStore) -> None:
    api = async_client(memory_store, [], MemoryJobStore())
    assert api.get("/api/query/nope").status_code == 404


def test_query_status_404_when_async_disabled(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    # client_for wires no jobs store, so the app runs sync and the status route reports async off.
    assert client_for(memory_store, []).get("/api/query/whatever").status_code == 404
