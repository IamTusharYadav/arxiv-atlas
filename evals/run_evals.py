"""Usage:
uv run python -m evals.run_evals --subset 15      # PR smoke gate
uv run python -m evals.run_evals --full           # weekly scheduled run
uv run python -m evals.run_evals --full --update-baseline   # re-baseline after review
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
from evals.golden_set import load_golden, subset
from evals.judge import Judgement, judge

log = logging.getLogger("evals.run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--subset", type=int, metavar="N", help="score N queries, balanced across categories"
    )
    group.add_argument("--full", action="store_true", help="score the whole golden set")
    group.add_argument("--id", action="append", metavar="ID", help="score only these query ids")
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
        queries = subset(queries, args.subset)
    elif args.id:
        queries = [q for q in queries if q.id in set(args.id)]
        if not queries:
            parser.error(f"no golden query matches {args.id}")

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
        # A red gate that does not say why costs a whole rerun to diagnose.
        if min(result.scores.relevance, result.scores.faithfulness) <= 3:
            log.warning("%s judge rationale: %s", query.id, result.scores.rationale)

    current = aggregate(judgements)
    comparison = compare(current, load_baseline())
    if not args.id:  # a one-query debug run is not a data point in the trend
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
