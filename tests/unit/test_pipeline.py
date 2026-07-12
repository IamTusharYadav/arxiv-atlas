from datetime import UTC, datetime
from itertools import pairwise

from atlas_ingest.pipeline import month_windows


def _d(y: int, m: int, day: int) -> datetime:
    return datetime(y, m, day, tzinfo=UTC)


def test_month_windows_tile_without_gaps_or_overlap() -> None:
    start, end = _d(2025, 7, 12), _d(2026, 7, 12)
    windows = list(month_windows(start, end))

    assert len(windows) == 12
    assert windows[0][0] == start
    assert windows[-1][1] == end
    # Each window's end is the next window's start: contiguous, no overlap.
    for (_, a_end), (b_start, _) in pairwise(windows):
        assert a_end == b_start


def test_month_windows_clamps_short_months() -> None:
    # Jan 31 -> Feb 28 (no Feb 31); the tail window is whatever is left.
    windows = list(month_windows(_d(2025, 1, 31), _d(2025, 3, 15)))
    assert windows[0] == (_d(2025, 1, 31), _d(2025, 2, 28))
    assert windows[-1][1] == _d(2025, 3, 15)


def test_month_windows_empty_when_end_not_after_start() -> None:
    assert list(month_windows(_d(2025, 7, 12), _d(2025, 7, 12))) == []
