import pytest

from atlas_api.v1 import _lead, normalize_arxiv_id


@pytest.mark.parametrize(
    "raw",
    [
        "2301.12345",
        "arXiv:2301.12345",
        "arxiv:2301.12345",
        "2301.12345v3",
        "https://arxiv.org/abs/2301.12345",
        "https://arxiv.org/abs/2301.12345v2",
        "https://arxiv.org/pdf/2301.12345v1.pdf",
        "  2301.12345  ",
    ],
)
def test_normalize_recovers_the_canonical_id(raw: str) -> None:
    assert normalize_arxiv_id(raw) == "2301.12345"


def test_normalize_handles_four_digit_sequence() -> None:
    assert normalize_arxiv_id("2301.1234v9") == "2301.1234"


def test_normalize_leaves_an_unrecognized_id_stripped() -> None:
    # No new-style id to extract: return the trimmed input and let the store 404 it.
    assert normalize_arxiv_id("  not-a-paper  ") == "not-a-paper"


def test_lead_truncates_and_marks_long_abstracts() -> None:
    long = "word " * 200
    out = _lead(long)
    assert len(out) <= 401  # 400 chars plus the ellipsis
    assert out.endswith("…")


def test_lead_leaves_short_abstracts_whole() -> None:
    assert _lead("short abstract") == "short abstract"
