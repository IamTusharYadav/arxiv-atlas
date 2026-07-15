from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Job:
    id: str
    status: str  # pending | running | done | error
    question: str
    result: dict[str, Any] | None
    error: str | None
    progress: list[dict[str, str]] = field(default_factory=list)
    # Epoch seconds of the last store write; the API uses it to spot dead workers. None on
    # stores that predate the field.
    updated_at: float | None = None


class JobStore(Protocol):
    def create(self, question: str) -> str: ...
    def get(self, job_id: str) -> Job | None: ...
    def mark_running(self, job_id: str) -> None: ...
    def set_progress(self, job_id: str, progress: list[dict[str, str]]) -> None: ...
    def finish(self, job_id: str, result: dict[str, Any]) -> None: ...
    def fail(self, job_id: str, error: str) -> None: ...
