from collections.abc import Callable

import httpx
import pytest

from atlas_mcp import server
from atlas_mcp.client import AtlasClient

Handler = Callable[[httpx.Request], httpx.Response]


def _wire(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    transport = httpx.MockTransport(handler)
    client = AtlasClient("https://atlas.test", http=httpx.Client(transport=transport))
    monkeypatch.setattr(server, "_client", client)


def test_search_passes_query_and_k(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(dict(req.url.params))
        return httpx.Response(200, json={"results": [{"arxiv_id": "2501.1"}], "note": None})

    _wire(monkeypatch, handler)
    out = server.search_papers("kv cache", k=3)
    assert seen == {"q": "kv cache", "k": "3"}
    assert out["results"][0]["arxiv_id"] == "2501.1"


def test_tool_errors_return_a_note_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, lambda req: httpx.Response(503))
    out = server.search_papers("anything")
    assert out["results"] == []
    assert out["note"]


def test_get_paper_missing_is_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, lambda req: httpx.Response(404, json={"detail": "x"}))
    out = server.get_paper("2501.99999")
    assert out["paper"] is None
    assert "not in the corpus" in out["note"]


def test_clusters_omits_k_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(dict(req.url.params))
        return httpx.Response(200, json={"clusters": []})

    _wire(monkeypatch, handler)
    server.get_topic_clusters("peft")
    assert "k" not in seen
    server.get_topic_clusters("peft", k=4)
    assert seen["k"] == "4"


def test_explore_forwards_the_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        return httpx.Response(200, json={"nodes": [], "links": [], "note": None})

    _wire(monkeypatch, handler)
    server.explore_from_paper("2501.12345")
    assert seen["path"] == "/api/v1/graph/2501.12345"
