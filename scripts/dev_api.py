"""Run the API locally against live Qdrant and Bedrock, for frontend work on routes the
deployed stack does not serve yet.

    uv run --with uvicorn python scripts/dev_api.py

Point the frontend at it with VITE_API_PROXY=http://localhost:8000 in frontend/.env.local.
Mirrors production's async job path: a run executes in a background thread, the POST returns
202 with a job id, and the frontend polls it for live per-step progress (expect 30-120s per
uncached run). An in-memory response cache makes a repeated question or topic return instantly,
same as the deployed DynamoDB cache. No budget guard or rate limit here; every uncached call
spends real Bedrock money. Needs .env (Qdrant + AWS credentials) and the exported model in
models/ (scripts/export_onnx.py).
"""

import threading
from dataclasses import replace
from pathlib import Path
from time import time
from typing import Any

import uvicorn
from qdrant_client import QdrantClient

from atlas_agents.bedrock import BedrockClient
from atlas_api import create_app
from atlas_api.app import run_job
from atlas_api.jobs import Job
from atlas_core.cache import CacheEntry, ResponseCache
from atlas_core.config import Settings, setup_logging
from atlas_core.embedding import Embedder, OnnxEmbedder
from atlas_core.vectorstore import QdrantStore


class _MemoryJobStore:
    """DynamoJobStore stand-in so local dev exercises the same enqueue -> poll -> per-step
    progress path as production, instead of blocking the request for the whole run."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, question: str, kind: str = "query") -> str:
        with self._lock:
            job_id = f"job-{len(self._jobs)}"
            self._jobs[job_id] = Job(
                job_id, "pending", question, None, None, updated_at=time(), kind=kind
            )
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        self._patch(job_id, status="running")

    def set_progress(self, job_id: str, progress: list[dict[str, str]]) -> None:
        self._patch(job_id, progress=progress)

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        self._patch(job_id, status="done", result=result)

    def fail(self, job_id: str, error: str) -> None:
        self._patch(job_id, status="error", error=error)

    def _patch(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                self._jobs[job_id] = replace(job, updated_at=time(), **changes)


class _MemoryCacheStore:
    """List-backed CacheStore; ResponseCache does the similarity match and expiry over it."""

    def __init__(self) -> None:
        self._entries: list[CacheEntry] = []
        self._lock = threading.Lock()

    def recent(self) -> list[CacheEntry]:
        with self._lock:
            return list(self._entries)

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entries.append(entry)


def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    models = Path(__file__).resolve().parent.parent / "models"
    store = QdrantStore(QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key))
    embedder: Embedder = OnnxEmbedder(models / "model_quantized.onnx", models / "tokenizer.json")
    client = BedrockClient()
    cache = ResponseCache(_MemoryCacheStore())
    jobs = _MemoryJobStore()

    def dispatch(job_id: str) -> None:
        # ponytail: one daemon thread per job. The shared embedder and Qdrant client are only
        # ever driven one run at a time in single-user local dev; swap for a real queue if you
        # start hammering it concurrently.
        threading.Thread(
            target=run_job,
            kwargs={
                "job_id": job_id,
                "jobs": jobs,
                "client": client,
                "store": store,
                "embedder": embedder,
                "cache": cache,
                "budget": None,
            },
            daemon=True,
        ).start()

    app = create_app(
        store=store, embedder=embedder, client=client, cache=cache, jobs=jobs, dispatch=dispatch
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
