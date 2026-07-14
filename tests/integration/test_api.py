from anthropic.types import Message
from fastapi.testclient import TestClient

from atlas_agents.bedrock import SONNET
from atlas_api import create_app
from atlas_core.models import Edge
from atlas_core.ratelimit import RateLimiter
from atlas_core.vectorstore import QdrantStore
from tests.conftest import FakeEmbedder, make_bedrock_client, make_message
from tests.unit.test_ask import IDS, check_json, extract_json, plan_json, rerank_json, seed
from tests.unit.test_ratelimit import MemoryBuckets


def client_for(
    store: QdrantStore,
    messages: list[Message | Exception],
    *,
    limiter: RateLimiter | None = None,
) -> TestClient:
    bedrock, _ = make_bedrock_client(messages)
    return TestClient(
        create_app(store=store, embedder=FakeEmbedder(), client=bedrock, limiter=limiter)
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
    # burst=1: the first request spends the only token, the second is throttled. TestClient
    # sends no X-Forwarded-For, so every request shares one bucket.
    limiter = RateLimiter(MemoryBuckets(), rate_per_s=1 / 6, burst=1)
    api = client_for(memory_store, [], limiter=limiter)
    assert api.get("/api/status").status_code == 200
    resp = api.get("/api/status")
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) > 0


def test_query_over_budget_returns_503(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    # A planner call charged past the per-query cap aborts the run; the route surfaces it as 503.
    api = client_for(memory_store, [make_message(plan_json(), output_tokens=5_000_000)])
    assert api.post("/api/query", json={"question": "kv cache?"}).status_code == 503
