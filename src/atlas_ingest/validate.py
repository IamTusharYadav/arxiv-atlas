"""Validation gates between fetch and upsert. Per-paper gates quarantine bad records; the
run-level gate fails loudly when the reject ratio points to upstream schema drift rather than
a few bad papers."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from atlas_core.models import Paper

MIN_ABSTRACT_CHARS = 200
MIN_TITLE_CHARS = 10
ALLOWED_CATEGORIES = frozenset({"cs.AI", "cs.LG", "cs.CL"})
EARLIEST_PUBLICATION = datetime(1991, 1, 1, tzinfo=UTC)
MAX_REJECT_RATIO = 0.2
MIN_PAPERS_FOR_RATIO_CHECK = 20


@dataclass
class Rejection:
    arxiv_id: str
    reason: str


class RunQualityError(RuntimeError):
    pass


def rejection_reason(paper: Paper, now: datetime | None = None) -> str | None:
    now = now or datetime.now(UTC)
    if len(paper.title) < MIN_TITLE_CHARS:
        return f"title too short ({len(paper.title)} chars)"
    if len(paper.abstract) < MIN_ABSTRACT_CHARS:
        return f"abstract too short ({len(paper.abstract)} chars)"
    if not paper.authors:
        return "no authors"
    if not ALLOWED_CATEGORIES.intersection(paper.categories):
        return f"no allowed category in {paper.categories}"
    if paper.published_at < EARLIEST_PUBLICATION:
        return f"implausible publication date {paper.published_at.date()}"
    if paper.published_at > now + timedelta(days=2):
        return f"publication date in the future {paper.published_at.date()}"
    return None


def validate_batch(papers: list[Paper]) -> tuple[list[Paper], list[Rejection]]:
    accepted: list[Paper] = []
    rejected: list[Rejection] = []
    now = datetime.now(UTC)
    for paper in papers:
        reason = rejection_reason(paper, now)
        if reason is None:
            accepted.append(paper)
        else:
            rejected.append(Rejection(arxiv_id=paper.arxiv_id, reason=reason))
    return accepted, rejected


def check_run_quality(accepted: int, rejected: list[Rejection]) -> None:
    total = accepted + len(rejected)
    if total < MIN_PAPERS_FOR_RATIO_CHECK:
        return
    ratio = len(rejected) / total
    if ratio > MAX_REJECT_RATIO:
        sample = "; ".join(f"{r.arxiv_id}: {r.reason}" for r in rejected[:5])
        raise RunQualityError(
            f"{len(rejected)}/{total} papers rejected ({ratio:.0%}), likely upstream schema "
            f"drift. Sample: {sample}"
        )
