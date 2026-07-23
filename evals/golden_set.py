from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import yaml

GOLDEN_DIR = Path(__file__).parent / "golden"


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    category: str
    question: str
    in_scope: bool
    expected_topics: tuple[str, ...] = ()
    rubric_notes: str = ""


def _parse(path: Path) -> GoldenQuery:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    query = GoldenQuery(
        id=str(data["id"]),
        category=str(data["category"]),
        question=str(data["question"]),
        in_scope=bool(data["in_scope"]),
        expected_topics=tuple(str(t) for t in data.get("expected_topics", [])),
        rubric_notes=str(data.get("rubric_notes", "")).strip(),
    )
    if query.id != path.stem:
        raise ValueError(f"golden query {path.name}: id {query.id!r} must match filename")
    return query


def load_golden(directory: Path = GOLDEN_DIR) -> list[GoldenQuery]:
    return [_parse(p) for p in sorted(directory.glob("*.yaml"))]


def subset(queries: list[GoldenQuery], n: int) -> list[GoldenQuery]:
    """Round-robin across categories, because filename order is alphabetical: a plain prefix
    would hand the nightly gate three adversarial cases and no technical survey at all."""
    by_category: dict[str, list[GoldenQuery]] = {}
    for query in queries:
        by_category.setdefault(query.category, []).append(query)
    interleaved = [
        query for group in zip_longest(*by_category.values()) for query in group if query
    ]
    return interleaved[:n]
