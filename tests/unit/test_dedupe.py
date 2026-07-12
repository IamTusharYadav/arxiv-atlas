from datetime import UTC, datetime

from atlas_ingest.dedupe import dedupe
from tests.conftest import make_paper


def test_keeps_highest_version_per_id() -> None:
    papers = [
        make_paper(arxiv_id="2607.00001", version=1, title="A Study of KV Caches"),
        make_paper(arxiv_id="2607.00001", version=3, title="A Study of KV Caches (revised)"),
        make_paper(arxiv_id="2607.00001", version=2, title="A Study of KV Caches"),
    ]
    result = dedupe(papers)
    assert len(result) == 1
    assert result[0].version == 3


def test_drops_near_duplicate_titles_keeping_newest() -> None:
    older = make_paper(
        arxiv_id="2606.99999",
        title="Efficient Attention Mechanisms for Long Contexts",
        updated_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    newer = make_paper(
        arxiv_id="2607.00001",
        title="Efficient Attention Mechanisms for Long Contexts.",
        updated_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    result = dedupe([older, newer])
    assert [p.arxiv_id for p in result] == ["2607.00001"]


def test_distinct_titles_survive() -> None:
    papers = [
        make_paper(arxiv_id="2607.00001", title="Sparse Mixture-of-Experts Routing"),
        make_paper(arxiv_id="2607.00002", title="Speculative Decoding with Draft Trees"),
    ]
    assert len(dedupe(papers)) == 2
