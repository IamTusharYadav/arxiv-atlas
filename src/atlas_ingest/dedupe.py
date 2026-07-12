import logging
import re
from difflib import SequenceMatcher

from atlas_core.models import Paper

log = logging.getLogger(__name__)

NEAR_DUP_TITLE_RATIO = 0.92
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def _title_key(title: str) -> str:
    return _NON_ALNUM.sub("", title.lower()).strip()


def dedupe(papers: list[Paper]) -> list[Paper]:
    """Within-batch dedupe: keep the highest version per arxiv id, then drop near-duplicate
    titles under different ids (re-posts), keeping the most recently updated."""
    by_id: dict[str, Paper] = {}
    for paper in papers:
        current = by_id.get(paper.arxiv_id)
        if current is None or paper.version > current.version:
            by_id[paper.arxiv_id] = paper

    # O(n^2) title comparison; nightly batches are a few hundred papers.
    # Switch to trigram blocking if backfill windows ever make this slow.
    survivors = sorted(by_id.values(), key=lambda p: p.updated_at, reverse=True)
    kept: list[Paper] = []
    for paper in survivors:
        key = _title_key(paper.title)
        duplicate_of = next(
            (
                k
                for k in kept
                if SequenceMatcher(None, key, _title_key(k.title)).ratio() >= NEAR_DUP_TITLE_RATIO
            ),
            None,
        )
        if duplicate_of is None:
            kept.append(paper)
        else:
            log.info("dropping %s as near-duplicate of %s", paper.arxiv_id, duplicate_of.arxiv_id)
    return kept
