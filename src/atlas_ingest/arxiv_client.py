import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from xml.etree import ElementTree

import httpx

from atlas_core.features import collapse_whitespace
from atlas_core.models import Paper

log = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
CATEGORIES = ("cs.AI", "cs.LG", "cs.CL")
USER_AGENT = "arxiv-atlas (https://github.com/IamTusharYadav/arxiv-atlas)"

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivApiError(RuntimeError):
    pass


@dataclass
class FeedPage:
    papers: list[Paper]
    total_results: int


def split_versioned_id(value: str) -> tuple[str, int]:
    base, sep, tail = value.rpartition("v")
    if sep and base and tail.isdigit():
        return base, int(tail)
    return value, 1


def parse_feed(content: bytes) -> FeedPage:
    root = ElementTree.fromstring(content)
    total = int(root.findtext("opensearch:totalResults", "0", _NS))
    papers = [_parse_entry(entry) for entry in root.findall("atom:entry", _NS)]
    return FeedPage(papers=papers, total_results=total)


def _parse_entry(entry: ElementTree.Element) -> Paper:
    raw_id = entry.findtext("atom:id", "", _NS).rsplit("/abs/", 1)[-1]
    arxiv_id, version = split_versioned_id(raw_id)
    primary = entry.find("arxiv:primary_category", _NS)
    return Paper(
        arxiv_id=arxiv_id,
        version=version,
        title=collapse_whitespace(entry.findtext("atom:title", "", _NS)),
        abstract=collapse_whitespace(entry.findtext("atom:summary", "", _NS)),
        authors=[
            name
            for author in entry.findall("atom:author", _NS)
            if (name := author.findtext("atom:name", "", _NS))
        ],
        categories=[
            term for cat in entry.findall("atom:category", _NS) if (term := cat.get("term"))
        ],
        primary_category=primary.get("term", "") if primary is not None else "",
        published_at=datetime.fromisoformat(entry.findtext("atom:published", "", _NS)),
        updated_at=datetime.fromisoformat(entry.findtext("atom:updated", "", _NS)),
    )


class ArxivClient:
    """Paginated fetcher honoring arXiv etiquette: one request per `delay_seconds`,
    identifying User-Agent, retries with backoff on transient failures."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        delay_seconds: float = 3.0,
        page_size: int = 100,
        max_retries: int = 3,
    ) -> None:
        self._http = http or httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT})
        self._delay = delay_seconds
        self._page_size = page_size
        self._max_retries = max_retries

    def fetch_since(self, since: datetime, max_records: int = 5000) -> Iterator[Paper]:
        """Papers updated after `since`, newest first. The caller re-upserts overlap
        idempotently, so a generous `since` is safe."""
        query = " OR ".join(f"cat:{c}" for c in CATEGORIES)
        for paper in self._paginate(query, sort_by="lastUpdatedDate", max_records=max_records):
            if paper.updated_at <= since:
                return
            yield paper

    def fetch_window(
        self, start: datetime, end: datetime, max_records: int = 5000
    ) -> Iterator[Paper]:
        """Backfill window by submission date. Keep windows small enough that arXiv's
        deep-pagination flakiness does not bite (a month is fine)."""
        cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
        fmt = "%Y%m%d%H%M"
        query = f"({cats}) AND submittedDate:[{start.strftime(fmt)} TO {end.strftime(fmt)}]"
        yield from self._paginate(query, sort_by="submittedDate", max_records=max_records)

    def _paginate(self, query: str, sort_by: str, max_records: int) -> Iterator[Paper]:
        start = 0
        while start < max_records:
            page = self._get_page(query, sort_by, start)
            if not page.papers:
                return
            yield from page.papers
            start += len(page.papers)
            if start >= page.total_results:
                return
            time.sleep(self._delay)

    def _get_page(self, query: str, sort_by: str, start: int) -> FeedPage:
        params: dict[str, str | int] = {
            "search_query": query,
            "sortBy": sort_by,
            "sortOrder": "descending",
            "start": start,
            "max_results": self._page_size,
        }
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            if attempt:
                time.sleep(self._delay * 2**attempt)
            try:
                response = self._http.get(API_URL, params=params)
                response.raise_for_status()
                page = parse_feed(response.content)
            except (httpx.HTTPError, ElementTree.ParseError) as exc:
                last_error = exc
                log.warning("arxiv request failed (attempt %d): %s", attempt + 1, exc)
                continue
            if not page.papers and start < page.total_results:
                # arXiv intermittently returns empty pages mid-pagination; retry before
                # trusting it as the end of results.
                last_error = ArxivApiError(f"empty page at start={start} of {page.total_results}")
                log.warning("empty page at start=%d, retrying", start)
                continue
            return page
        raise ArxivApiError(f"arxiv api failed after {self._max_retries} attempts") from last_error
