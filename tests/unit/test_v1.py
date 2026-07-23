import numpy as np
import pytest

from atlas_api.v1 import _lead, bridge_scores, normalize_arxiv_id


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


def test_bridge_scores_ranks_by_the_weaker_side() -> None:
    a = np.array([1, 0, 0], dtype=np.float32)  # sim_a reads a paper vector's first component
    b = np.array([0, 1, 0], dtype=np.float32)  # sim_b reads its second
    vectors = {
        "both": [0.7, 0.7, 0.0],  # strong on both
        "a_only": [0.95, 0.1, 0.0],  # strong on a, below floor on b
        "weakish": [0.5, 0.45, 0.0],  # clears both, but by less
    }
    out = bridge_scores(vectors, a, b, floor=0.35)

    assert [i for i, _, _ in out] == ["both", "weakish"]  # a_only filtered; ranked by weaker side
    assert all(sim_a >= 0.35 and sim_b >= 0.35 for _, sim_a, sim_b in out)


def test_bridge_scores_empty_when_nothing_clears_both() -> None:
    a = np.array([1, 0], dtype=np.float32)
    b = np.array([0, 1], dtype=np.float32)
    assert bridge_scores({"x": [0.9, 0.1]}, a, b, floor=0.35) == []
