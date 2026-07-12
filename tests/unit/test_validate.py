from datetime import UTC, datetime

import pytest

from atlas_ingest.validate import (
    RunQualityError,
    check_run_quality,
    rejection_reason,
    validate_batch,
)
from tests.conftest import make_paper


def test_valid_paper_passes() -> None:
    assert rejection_reason(make_paper()) is None


def test_short_abstract_rejected() -> None:
    reason = rejection_reason(make_paper(abstract="Too short."))
    assert reason is not None and "abstract too short" in reason


def test_missing_authors_rejected() -> None:
    assert rejection_reason(make_paper(authors=[])) == "no authors"


def test_foreign_category_rejected() -> None:
    reason = rejection_reason(make_paper(categories=["q-bio.GN"], primary_category="q-bio.GN"))
    assert reason is not None and "no allowed category" in reason


def test_cross_listed_paper_accepted() -> None:
    paper = make_paper(categories=["cs.RO", "cs.AI"], primary_category="cs.RO")
    assert rejection_reason(paper) is None


def test_future_publication_rejected() -> None:
    paper = make_paper(
        published_at=datetime(2030, 1, 1, tzinfo=UTC),
        updated_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    reason = rejection_reason(paper)
    assert reason is not None and "future" in reason


def test_validate_batch_partitions() -> None:
    papers = [make_paper(arxiv_id="2607.00001"), make_paper(arxiv_id="2607.00002", abstract="x")]
    accepted, rejected = validate_batch(papers)
    assert [p.arxiv_id for p in accepted] == ["2607.00001"]
    assert [r.arxiv_id for r in rejected] == ["2607.00002"]


def test_run_quality_gate_trips_on_high_reject_ratio() -> None:
    _, rejected = validate_batch([make_paper(abstract="x") for _ in range(6)])
    with pytest.raises(RunQualityError, match="schema"):
        check_run_quality(accepted=14, rejected=rejected)


def test_run_quality_gate_ignores_tiny_batches() -> None:
    _, rejected = validate_batch([make_paper(abstract="x") for _ in range(5)])
    check_run_quality(accepted=5, rejected=rejected)
