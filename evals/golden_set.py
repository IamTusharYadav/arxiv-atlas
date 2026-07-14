from dataclasses import dataclass
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
