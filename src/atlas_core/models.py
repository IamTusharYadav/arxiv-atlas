from datetime import datetime

from pydantic import BaseModel, field_validator


class Paper(BaseModel):
    """One arXiv paper at its latest known version. `arxiv_id` is the base id
    without the version suffix (e.g. "2507.01234", "cs/0112017")."""

    arxiv_id: str
    version: int = 1
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published_at: datetime
    updated_at: datetime

    @field_validator("published_at", "updated_at")
    @classmethod
    def _require_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("datetimes must be timezone-aware")
        return value


class Edge(BaseModel):
    """Semantic similarity edge between two papers, weight = cosine similarity."""

    source: str
    target: str
    weight: float
