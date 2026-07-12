from datetime import UTC, datetime

import httpx
import pytest

from atlas_ingest.arxiv_client import (
    ArxivApiError,
    ArxivClient,
    parse_feed,
    split_versioned_id,
)
from tests.conftest import FIXTURES

_ENTRY_TEMPLATE = """
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <title>Paper {arxiv_id}</title>
    <updated>{updated}</updated>
    <summary>An abstract.</summary>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <published>{updated}</published>
    <arxiv:primary_category term="cs.LG"/>
    <author><name>A. Author</name></author>
  </entry>
"""


def synthetic_feed(entries: list[tuple[str, str]], total: int) -> bytes:
    body = "".join(
        _ENTRY_TEMPLATE.format(arxiv_id=arxiv_id, updated=updated) for arxiv_id, updated in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>{body}</feed>"
    ).encode()


def client_with(handler: httpx.MockTransport) -> ArxivClient:
    return ArxivClient(http=httpx.Client(transport=handler), delay_seconds=0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2606.26428v2", ("2606.26428", 2)),
        ("2607.08768v1", ("2607.08768", 1)),
        ("cs/0112017v3", ("cs/0112017", 3)),
        ("2607.08768", ("2607.08768", 1)),
    ],
)
def test_split_versioned_id(raw: str, expected: tuple[str, int]) -> None:
    assert split_versioned_id(raw) == expected


def test_parse_recorded_fixture() -> None:
    page = parse_feed((FIXTURES / "arxiv_page.xml").read_bytes())
    assert page.total_results == 444205
    assert len(page.papers) == 5

    first = page.papers[0]
    assert first.arxiv_id == "2606.26428"
    assert first.version == 2
    assert first.title.startswith("Play2Perfect")
    assert first.primary_category == "cs.RO"
    assert "cs.AI" in first.categories
    assert len(first.authors) == 4
    assert first.published_at.tzinfo is not None
    assert first.updated_at == datetime(2026, 7, 9, 17, 59, 45, tzinfo=UTC)
    # Multi-line abstracts from the feed arrive whitespace-collapsed.
    assert "\n" not in page.papers[4].abstract


def test_fetch_since_stops_at_cutoff() -> None:
    feed = synthetic_feed(
        [
            ("2607.00003v1", "2026-07-03T00:00:00Z"),
            ("2607.00002v1", "2026-07-02T00:00:00Z"),
            ("2607.00001v1", "2026-07-01T00:00:00Z"),
        ],
        total=3,
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=feed))
    papers = list(client_with(transport).fetch_since(datetime(2026, 7, 1, 12, 0, tzinfo=UTC)))
    assert [p.arxiv_id for p in papers] == ["2607.00003", "2607.00002"]


def test_pagination_follows_start_offsets() -> None:
    pages = {
        0: synthetic_feed([("2607.00002v1", "2026-07-02T00:00:00Z")], total=2),
        1: synthetic_feed([("2607.00001v1", "2026-07-01T00:00:00Z")], total=2),
    }
    requested: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["start"])
        requested.append(start)
        return httpx.Response(200, content=pages[start])

    client = ArxivClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        delay_seconds=0,
        page_size=1,
    )
    papers = list(
        client.fetch_window(datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 7, 3, tzinfo=UTC))
    )
    assert [p.arxiv_id for p in papers] == ["2607.00002", "2607.00001"]
    assert requested == [0, 1]


def test_retries_server_errors_then_succeeds() -> None:
    feed = synthetic_feed([("2607.00001v1", "2026-07-01T00:00:00Z")], total=1)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=feed)

    transport = httpx.MockTransport(handler)
    papers = list(
        client_with(transport).fetch_window(
            datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 7, 2, tzinfo=UTC)
        )
    )
    assert len(papers) == 1
    assert calls["n"] == 3


def test_persistent_empty_pages_raise() -> None:
    # Claims 100 results but always returns an empty page: the known arXiv flakiness.
    feed = synthetic_feed([], total=100)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=feed))
    with pytest.raises(ArxivApiError, match="failed after"):
        list(
            client_with(transport).fetch_window(
                datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 7, 2, tzinfo=UTC)
            )
        )


def test_max_records_caps_runaway_fetch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["start"])
        entry_id = f"2607.{start:05d}v1"
        return httpx.Response(
            200, content=synthetic_feed([(entry_id, "2026-07-01T00:00:00Z")], total=10_000)
        )

    client = ArxivClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        delay_seconds=0,
        page_size=1,
    )
    papers = list(
        client.fetch_window(
            datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 7, 2, tzinfo=UTC), max_records=3
        )
    )
    assert len(papers) == 3
