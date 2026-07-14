import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from atlas_core.vectorstore import QdrantStore
from atlas_ingest.arxiv_client import ArxivClient
from atlas_ingest.pipeline import run_ingest
from tests.conftest import FIXTURES, FakeEmbedder

WINDOW = (datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 7, 10, tzinfo=UTC))


@pytest.fixture
def recorded_client() -> ArxivClient:
    # The live fixture claims 444205 total results; patch it to the page size so
    # pagination terminates after one page.
    content = (FIXTURES / "arxiv_page.xml").read_bytes()
    content = re.sub(rb"<opensearch:totalResults>\d+", b"<opensearch:totalResults>5", content)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=content))
    return ArxivClient(http=httpx.Client(transport=transport), delay_seconds=0)


def test_backfill_then_nightly_run(
    recorded_client: ArxivClient, memory_store: QdrantStore, tmp_path: Path
) -> None:
    adjacency = tmp_path / "adjacency.parquet"

    report = run_ingest(
        recorded_client, memory_store, FakeEmbedder(), window=WINDOW, adjacency_path=adjacency
    )
    assert report.fetched == 5
    assert report.rejected == 0
    assert report.upserted == 5
    assert report.corpus_size == 5
    assert adjacency.exists()

    # Nightly mode resumes from the corpus checkpoint and re-upserts idempotently.
    report2 = run_ingest(recorded_client, memory_store, FakeEmbedder())
    assert report2.corpus_size == 5

    latest = memory_store.latest_updated_at()
    assert latest == datetime(2026, 7, 9, 17, 59, 45, tzinfo=UTC)


def test_hitting_max_records_cap_fails_instead_of_truncating(
    recorded_client: ArxivClient, memory_store: QdrantStore
) -> None:
    with pytest.raises(RuntimeError, match="max-records cap"):
        run_ingest(recorded_client, memory_store, FakeEmbedder(), window=WINDOW, max_records=5)
    assert memory_store.count() == 0


def test_nightly_mode_requires_backfilled_corpus(
    recorded_client: ArxivClient, memory_store: QdrantStore
) -> None:
    with pytest.raises(RuntimeError, match="backfill"):
        run_ingest(recorded_client, memory_store, FakeEmbedder())
