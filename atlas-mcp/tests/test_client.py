import logging
from collections.abc import Callable

import httpx
import pytest

from atlas_mcp.client import (
    AtlasClient,
    AtlasError,
    AtlasRateLimited,
    AtlasUnavailable,
    NotFound,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> AtlasClient:
    transport = httpx.MockTransport(handler)
    return AtlasClient("https://atlas.test", http=httpx.Client(transport=transport))


def test_get_returns_json_on_200() -> None:
    client = _client(lambda req: httpx.Response(200, json={"ok": True}))
    assert client.get("/api/v1/search", {"q": "x"}) == {"ok": True}


def test_429_becomes_rate_limited_with_retry_after() -> None:
    client = _client(lambda req: httpx.Response(429, headers={"retry-after": "12"}))
    with pytest.raises(AtlasRateLimited) as exc:
        client.get("/api/v1/search")
    assert exc.value.retry_after == 12


def test_404_becomes_not_found() -> None:
    client = _client(lambda req: httpx.Response(404, json={"detail": "nope"}))
    with pytest.raises(NotFound):
        client.get("/api/v1/paper/x")


def test_5xx_never_leaks_internals() -> None:
    # The upstream body carries an internal message; the client must not pass it through.
    client = _client(lambda req: httpx.Response(502, text="traceback at internal-lambda-url"))
    with pytest.raises(AtlasUnavailable) as exc:
        client.get("/api/v1/search")
    assert "internal" not in str(exc.value).lower()


def test_retries_once_on_timeout_then_raises() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(AtlasUnavailable):
        _client(handler).get("/api/v1/search")
    assert calls["n"] == 2  # one retry, not an infinite loop


def test_retries_once_on_5xx_then_succeeds() -> None:
    # A cold start comes back as a gateway 503, not a client timeout. The first call must not be
    # the one the user sees fail.
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"results": []})

    assert _client(handler).get("/api/v1/search") == {"results": []}
    assert calls["n"] == 2


def test_gives_up_after_one_5xx_retry() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    with pytest.raises(AtlasUnavailable):
        _client(handler).get("/api/v1/search")
    assert calls["n"] == 2


def test_transport_error_is_not_leaked() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect to 10.0.0.1 failed", request=req)

    with pytest.raises(AtlasUnavailable) as exc:
        _client(handler).get("/api/v1/search")
    assert "10.0.0" not in str(exc.value)


def test_env_var_sets_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_API_URL", "https://staging.example/")
    assert AtlasClient().base_url == "https://staging.example"  # trailing slash trimmed


def test_explicit_base_url_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_API_URL", "https://staging.example")
    assert AtlasClient("https://prod.example").base_url == "https://prod.example"


def test_generic_4xx_message_is_not_specific() -> None:
    client = _client(lambda req: httpx.Response(422, json={"detail": [{"loc": ["q"]}]}))
    with pytest.raises(AtlasError):
        client.get("/api/v1/search")


def test_httpx_request_logging_is_silenced() -> None:
    # Importing the client must quiet httpx's INFO request log, where the query URL would leak.
    import atlas_mcp.client  # noqa: F401

    assert logging.getLogger("httpx").level >= logging.WARNING
