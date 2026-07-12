import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from atlas_core.config import Settings, setup_logging
from atlas_core.embedding import Embedder, passage_text
from atlas_core.graph import link_paper, write_adjacency
from atlas_core.vectorstore import QdrantStore, VectorStore
from atlas_ingest.arxiv_client import ArxivClient
from atlas_ingest.dedupe import dedupe
from atlas_ingest.validate import check_run_quality, validate_batch

log = logging.getLogger(__name__)

# Re-fetch a little history each night; upserts are idempotent, and the overlap covers
# clock skew and papers announced late.
CHECKPOINT_OVERLAP = timedelta(hours=1)


@dataclass
class IngestReport:
    fetched: int
    rejected: int
    deduplicated: int
    upserted: int
    edges: int
    corpus_size: int


def run_ingest(
    client: ArxivClient,
    store: VectorStore,
    embedder: Embedder,
    window: tuple[datetime, datetime] | None = None,
    max_records: int = 5000,
    adjacency_path: Path | None = None,
) -> IngestReport:
    """Fetch, validate, dedupe, embed, upsert, link. Without `window`, resumes from the
    corpus itself: the max updated_at already stored is the checkpoint, so a separate
    checkpoint store cannot drift from reality."""
    store.ensure_collection()

    if window is not None:
        fetched = list(client.fetch_window(*window, max_records=max_records))
    else:
        since = store.latest_updated_at()
        if since is None:
            raise RuntimeError("corpus is empty; run a backfill window first (--from/--to)")
        fetched = list(client.fetch_since(since - CHECKPOINT_OVERLAP, max_records=max_records))
    log.info("fetched %d papers", len(fetched))

    accepted, rejections = validate_batch(fetched)
    for rejection in rejections:
        log.warning("rejected %s: %s", rejection.arxiv_id, rejection.reason)
    check_run_quality(len(accepted), rejections)

    papers = dedupe(accepted)
    vectors = embedder.embed([passage_text(p) for p in papers]) if papers else None

    edge_count = 0
    if vectors is not None:
        store.upsert(list(zip(papers, vectors.tolist(), strict=True)))
        # Link after all upserts so papers within the same batch can connect to each other.
        for paper, vector in zip(papers, vectors.tolist(), strict=True):
            edges = link_paper(store, paper.arxiv_id, vector)
            store.set_edges(paper.arxiv_id, edges)
            edge_count += len(edges)

    if adjacency_path is not None:
        total_edges = write_adjacency(store, adjacency_path)
        log.info("adjacency artifact written: %d edges -> %s", total_edges, adjacency_path)

    report = IngestReport(
        fetched=len(fetched),
        rejected=len(rejections),
        deduplicated=len(accepted) - len(papers),
        upserted=len(papers),
        edges=edge_count,
        corpus_size=store.count(),
    )
    log.info("ingest complete: %s", report)
    return report


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ArXiv Atlas ingestion pipeline")
    parser.add_argument(
        "--from",
        dest="from_date",
        type=_parse_date,
        default=None,
        help="backfill window start (YYYY-MM-DD); omit for nightly mode",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        type=_parse_date,
        default=None,
        help="backfill window end, defaults to now",
    )
    parser.add_argument(
        "--max-records", type=int, default=5000, help="hard cap per run as a runaway guard"
    )
    parser.add_argument(
        "--adjacency-out",
        type=Path,
        default=None,
        help="write the parquet adjacency artifact here after ingest",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    setup_logging(settings.log_level)

    window = None
    if args.from_date is not None:
        window = (args.from_date, args.to_date or datetime.now(UTC))

    from atlas_core.embedding import SentenceTransformerEmbedder

    run_ingest(
        client=ArxivClient(),
        store=QdrantStore.from_settings(settings),
        embedder=SentenceTransformerEmbedder(),
        window=window,
        max_records=args.max_records,
        adjacency_path=args.adjacency_out,
    )


if __name__ == "__main__":
    main()
