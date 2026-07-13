"""Langfuse tracing for agent runs: one trace per question, one span per loop step tagged
with token counts, cost, and the prompt registry version (plan section I).

Tracing is fire-and-forget. It never blocks a request and never fails one: without Langfuse
credentials it is a no-op, every SDK call is wrapped so an outage cannot raise into the loop,
and the end-of-run flush is bounded to `FLUSH_TIMEOUT_S` so a slow collector cannot stall the
response. Correctness of the answer never depends on the trace being delivered.
"""

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from langfuse import Langfuse

from atlas_agents.harness import StepRecord

log = logging.getLogger(__name__)

FLUSH_TIMEOUT_S = 0.3

_client: Langfuse | None = None
_client_ready = False


def _get_client() -> Langfuse | None:
    """Cache one client. Absent credentials means tracing is disabled, not an error."""
    global _client, _client_ready
    if not _client_ready:
        _client_ready = True
        if os.getenv("LANGFUSE_PUBLIC_KEY"):
            try:
                _client = Langfuse()
            except Exception as err:  # never let init failure touch the request path
                log.warning("langfuse init failed, tracing disabled: %s", err)
    return _client


class _LangfuseSink:
    def __init__(self, root: object) -> None:
        self._root = root

    def step(self, record: StepRecord, *, model: str | None, version: str | None) -> None:
        try:
            kind = "generation" if record.output_tokens else "span"
            usage = (
                {"input": record.input_tokens, "output": record.output_tokens}
                if record.input_tokens or record.output_tokens
                else None
            )
            obs = self._root.start_observation(  # type: ignore[attr-defined]
                name=record.step,
                as_type=kind,
                metadata={"summary": record.summary},
                version=version,
                model=model,
                usage_details=usage,
                cost_details={"total": record.cost_usd} if record.cost_usd else None,
            )
            obs.end()
        except Exception as err:
            log.debug("langfuse span emit failed: %s", err)

    def set_output(self, text: str) -> None:
        try:
            self._root.update(output=text)  # type: ignore[attr-defined]
        except Exception as err:
            log.debug("langfuse root update failed: %s", err)


class _NullSink:
    def step(self, record: StepRecord, *, model: str | None, version: str | None) -> None:
        pass

    def set_output(self, text: str) -> None:
        pass


def _bounded_flush(client: Langfuse) -> None:
    """Flush in a daemon thread and wait at most FLUSH_TIMEOUT_S. A slow or dead collector
    leaves the daemon running and returns; the request is never held on it."""
    thread = threading.Thread(target=client.flush, daemon=True)
    thread.start()
    thread.join(FLUSH_TIMEOUT_S)


@contextmanager
def query_trace(question: str) -> Iterator[_LangfuseSink | _NullSink]:
    client = _get_client()
    if client is None:
        yield _NullSink()
        return
    try:
        root = client.start_observation(name="ask", as_type="span", input={"question": question})
    except Exception as err:
        log.debug("langfuse trace start failed: %s", err)
        yield _NullSink()
        return
    sink = _LangfuseSink(root)
    try:
        yield sink
    finally:
        try:
            root.end()
        except Exception as err:
            log.debug("langfuse trace end failed: %s", err)
        _bounded_flush(client)
