from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Job:
    id: str
    status: str  # pending | running | done | error
    question: str
    result: dict[str, Any] | None
    error: str | None


class JobStore(Protocol):
    def create(self, question: str) -> str: ...
    def get(self, job_id: str) -> Job | None: ...
    def mark_running(self, job_id: str) -> None: ...
    def finish(self, job_id: str, result: dict[str, Any]) -> None: ...
    def fail(self, job_id: str, error: str) -> None: ...
