import threading
import time
from typing import Any

import pytest

import atlas_agents.tracing as tracing
from atlas_agents.bedrock import HAIKU
from atlas_agents.harness import StepRecord
from atlas_agents.tracing import _bounded_flush, _LangfuseSink, query_trace


class _FakeObs:
    def __init__(self) -> None:
        self.ended = False

    def end(self) -> None:
        self.ended = True


class _FakeRoot:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.obs = _FakeObs()

    def start_observation(self, **kwargs: Any) -> _FakeObs:
        self.calls.append(kwargs)
        return self.obs


def test_sink_emits_generation_with_version_and_usage() -> None:
    root = _FakeRoot()
    rec = StepRecord("planner", "2 subqueries", input_tokens=100, output_tokens=50, cost_usd=0.001)
    _LangfuseSink(root).step(rec, model=HAIKU, version="1.0.0")

    call = root.calls[0]
    assert call["name"] == "planner"
    assert call["version"] == "1.0.0"
    assert call["as_type"] == "generation"  # output tokens present
    assert call["usage_details"] == {"input": 100, "output": 50}
    assert root.obs.ended


def test_sink_emits_plain_span_for_modelless_step() -> None:
    root = _FakeRoot()
    _LangfuseSink(root).step(StepRecord("retriever", "12 candidates"), model=None, version=None)
    assert root.calls[0]["as_type"] == "span"
    assert root.calls[0]["usage_details"] is None


def test_sink_swallows_backend_errors() -> None:
    class _Boom:
        def start_observation(self, **kwargs: Any) -> None:
            raise RuntimeError("langfuse unreachable")

    # Must not raise into the loop.
    _LangfuseSink(_Boom()).step(StepRecord("planner", "x"), model=None, version=None)


def test_query_trace_without_credentials_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_client_ready", False)
    with query_trace("some question") as sink:
        sink.step(StepRecord("planner", "x"), model=None, version=None)  # no raise
        sink.set_output("brief")  # no raise


def test_bounded_flush_returns_within_timeout() -> None:
    release = threading.Event()
    flushed = threading.Event()

    class _SlowClient:
        def flush(self) -> None:
            release.wait(5)
            flushed.set()

    start = time.monotonic()
    _bounded_flush(_SlowClient())  # type: ignore[arg-type]
    elapsed = time.monotonic() - start

    assert elapsed < 1.0  # did not block on the slow flush
    assert not flushed.is_set()  # flush still running in the daemon
    release.set()  # let the daemon thread finish
