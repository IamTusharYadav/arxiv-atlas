import hashlib
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import numpy as np
import numpy.typing as npt
import pytest
from anthropic.types import Message, TextBlock, Usage
from qdrant_client import QdrantClient

from atlas_agents.bedrock import BedrockClient
from atlas_core.embedding import CONTRACT
from atlas_core.models import Paper
from atlas_core.vectorstore import QdrantStore

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Deterministic per-text vectors; no model download. Distinct texts land near-orthogonal
    in 384 dims, so no accidental edges appear in tests."""

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        out = np.empty((len(texts), CONTRACT.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            vector = np.random.default_rng(seed).standard_normal(CONTRACT.dimension)
            out[i] = (vector / np.linalg.norm(vector)).astype(np.float32)
        return out


def make_paper(
    arxiv_id: str = "2607.00001",
    version: int = 1,
    title: str = "A Perfectly Plausible Paper About Language Models",
    updated_at: datetime | None = None,
    **overrides: object,
) -> Paper:
    fields: dict[str, object] = {
        "arxiv_id": arxiv_id,
        "version": version,
        "title": title,
        "abstract": "We study language models. " * 10,
        "authors": ["Ada Lovelace"],
        "categories": ["cs.LG"],
        "primary_category": "cs.LG",
        "published_at": datetime(2026, 7, 1, tzinfo=UTC),
        "updated_at": updated_at or datetime(2026, 7, 2, tzinfo=UTC),
    }
    fields.update(overrides)
    return Paper.model_validate(fields)


@pytest.fixture
def memory_store() -> QdrantStore:
    store = QdrantStore(QdrantClient(location=":memory:"))
    store.ensure_collection()
    return store


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_message(
    text: str,
    model: str = "anthropic.claude-haiku-4-5",
    input_tokens: int = 100,
    output_tokens: int = 50,
    stop_reason: str = "end_turn",
) -> Message:
    return Message.model_construct(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        content=[TextBlock(type="text", text=text)],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeMessages:
    """Scripted Bedrock outcomes per call; exceptions in the list are raised."""

    def __init__(self, outcomes: list[Message | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> Message:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_bedrock_client(
    outcomes: list[Message | Exception],
) -> tuple[BedrockClient, FakeMessages]:
    fake = FakeMessages(outcomes)
    inner = anthropic.AnthropicBedrockMantle(
        aws_region="us-east-1", aws_access_key="test", aws_secret_key="test"
    )
    inner.messages = fake  # type: ignore[assignment]
    return BedrockClient(client=inner), fake
