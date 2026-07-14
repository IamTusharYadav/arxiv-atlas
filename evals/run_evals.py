"""Eval runner CLI: answer each golden query with the live agent, judge it, and compare the
aggregate against the stored baseline.

    uv run python -m evals.run_evals --subset 15      # PR smoke gate
    uv run python -m evals.run_evals --full           # nightly
    uv run python -m evals.run_evals --full --update-baseline   # re-baseline after review

Exits non-zero when a gated dimension regresses, so CI can gate a merge on it. Needs live
Bedrock and a populated Qdrant (the `agents` and `ingest` extras); it is glue, not covered by
unit tests, which exercise the judge and comparison with fakes.
"""

import argparse
import logging
import sys

from atlas_agents.ask import ask
from atlas_agents.bedrock import BedrockClient
from atlas_core.config import Settings, setup_logging
from atlas_core.embedding import SentenceTransformerEmbedder
from atlas_core.vectorstore import QdrantStore
from evals.baseline import (
    aggregate,
    append_history,
    compare,
    load_baseline,
    save_baseline,
)
from evals.golden_set import load_golden
from evals.judge import Judgement, judge

log = logging.getLogger("evals.run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--subset", type=int, metavar="N", help="score only the first N queries")
    group.add_argument("--full", action="store_true", help="score the whole golden set")
    parser.add_argument("--samples", type=int, default=1, help="judge samples per answer (median)")
    parser.add_argument(
        "--update-baseline", action="store_true", help="overwrite the baseline with this run"
    )
    args = parser.parse_args(argv)

    setup_logging()
    settings = Settings.from_env()
    store = QdrantStore.from_settings(settings)
    embedder = SentenceTransformerEmbedder()
    client = BedrockClient()

    queries = load_golden()
    if args.subset:
        queries = queries[: args.subset]

    judgements: list[Judgement] = []
    for query in queries:
        answer = ask(query.question, client=client, store=store, embedder=embedder)
        result = judge(client, query, answer, samples=args.samples)
        judgements.append(result)
        log.info(
            "%s: rel=%d faith=%d cite=%d ($%.4f)",
            query.id,
            result.scores.relevance,
            result.scores.faithfulness,
            result.scores.citation_correctness,
            result.cost_usd,
        )

    current = aggregate(judgements)
    comparison = compare(current, load_baseline())
    append_history(current)

    log.info("aggregate over %d queries: %s", current.n, current)
    if comparison.baseline is not None:
        log.info("baseline: %s", comparison.baseline)
    if not comparison.passed:
        log.error("regressions on gated dimensions: %s", ", ".join(comparison.regressions))

    if args.update_baseline:
        save_baseline(current)
        log.info("baseline updated")

    return 0 if comparison.passed else 1


if __name__ == "__main__":
    sys.exit(main())
