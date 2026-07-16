import json
from datetime import UTC, datetime

import numpy as np
import pytest

from atlas_agents.bedrock import SONNET
from atlas_agents.landscape import MIN_PAPERS, map_topic
from atlas_agents.steps.synthesizer import UngroundedCitations
from atlas_core.embedding import CONTRACT
from atlas_core.vectorstore import QdrantStore
from tests.conftest import FakeEmbedder, make_bedrock_client, make_message, make_paper
from tests.unit.test_ask import plan_json

# Three well-separated blobs of four papers each; k-means with k=3 recovers them exactly,
# so the scripted call count (one direction label per blob) is stable.
BLOBS = 3
PER_BLOB = 4


def landscape_seed(store: QdrantStore) -> list[list[str]]:
    rng = np.random.default_rng(42)
    ids: list[list[str]] = []
    items = []
    for b in range(BLOBS):
        base = np.zeros(CONTRACT.dimension, dtype=np.float32)
        base[b] = 1.0
        blob_ids = []
        for m in range(PER_BLOB):
            vector = base + rng.normal(0, 0.02, CONTRACT.dimension).astype(np.float32)
            vector /= np.linalg.norm(vector)
            # 25xx ids so they can never collide with the 2607.* ids test_ask's seed() uses;
            # integration tests mix both corpora in one store.
            arxiv_id = f"250{5 + b}.0000{m + 1}"
            paper = make_paper(
                arxiv_id=arxiv_id,
                title=f"Paper {arxiv_id}",
                published_at=datetime(2025, 5 + b, 3, tzinfo=UTC),
            )
            items.append((paper, vector.tolist()))
            blob_ids.append(arxiv_id)
        ids.append(blob_ids)
    store.upsert(items)
    return ids


def direction_json(name: str, representative_ids: list[str] | None = None) -> str:
    return json.dumps(
        {
            "name": name,
            "problem": f"the problem {name} attacks",
            "representative_ids": representative_ids or [],
        }
    )


def landscape_json(cite_id: str, read_id: str, bogus_read: str | None = None) -> str:
    reading = [{"arxiv_id": read_id, "reason": "start here"}]
    if bogus_read:
        reading.append({"arxiv_id": bogus_read, "reason": "not retrieved"})
    return json.dumps(
        {
            "overview": f"This area is active [{cite_id}].",
            "key_ideas": ["one defining idea"],
            "reading_order": reading,
            "open_problems": ["an unresolved challenge"],
        }
    )


def test_map_topic_builds_grounded_landscape(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    ids = landscape_seed(memory_store)
    first = ids[0][0]
    client, fake = make_bedrock_client(
        [
            make_message(plan_json()),
            make_message(direction_json("Direction A")),
            make_message(direction_json("Direction B")),
            make_message(direction_json("Direction C")),
            make_message(landscape_json(first, first, bogus_read="9999.99999"), model=SONNET),
        ]
    )

    landscape = map_topic("kv cache", client=client, store=memory_store, embedder=fake_embedder)

    assert not landscape.declined
    assert f"[{first}]" in landscape.overview
    assert len(landscape.directions) == BLOBS
    for direction in landscape.directions:
        assert len(direction.papers) == PER_BLOB
        # blob membership survives clustering: every member shares the blob prefix
        prefixes = {p.arxiv_id.split(".")[0] for p in direction.papers}
        assert len(prefixes) == 1
        # scripted labels returned no ids, so representatives fall back to central order
        assert direction.representative_ids == [p.arxiv_id for p in direction.papers[:3]]
    # the un-retrieved reading id was dropped, the real one kept with its reason
    assert [(r.arxiv_id, r.reason) for r in landscape.reading_order] == [(first, "start here")]
    # one blob per month, four papers each
    assert {(t.month, t.count) for t in landscape.timeline} == {
        ("2025-05", 4),
        ("2025-06", 4),
        ("2025-07", 4),
    }
    steps = [r.step for r in landscape.trace]
    assert steps[:2] == ["planner", "retriever"]
    assert steps.count("direction") == BLOBS
    assert steps[-1] == "landscape"
    assert landscape.cost_usd > 0
    assert fake.calls[-1]["model"] == SONNET


def test_map_topic_out_of_scope_declines(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    plan = json.dumps(
        {
            "in_scope": False,
            "subqueries": [],
            "stop_criterion": "",
            "scope_note": "Astrophysics lives outside this corpus.",
        }
    )
    client, fake = make_bedrock_client([make_message(plan)])

    landscape = map_topic("dark matter", client=client, store=memory_store, embedder=fake_embedder)

    assert landscape.declined
    assert "Astrophysics" in landscape.overview
    assert landscape.directions == []
    assert len(fake.calls) == 1  # planner only


def test_map_topic_declines_on_thin_retrieval(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    # Fewer matches than MIN_PAPERS: clustering noise, so the pipeline says so instead.
    papers = [make_paper(arxiv_id=f"2607.0000{i}") for i in range(1, MIN_PAPERS - 1)]
    vectors = fake_embedder.embed([p.arxiv_id for p in papers])
    memory_store.upsert([(p, v.tolist()) for p, v in zip(papers, vectors, strict=True)])
    client, fake = make_bedrock_client([make_message(plan_json())])

    landscape = map_topic("kv cache", client=client, store=memory_store, embedder=fake_embedder)

    assert landscape.declined
    assert "too few" in landscape.overview
    assert len(fake.calls) == 1


def test_map_topic_rejects_ungrounded_citations_after_repair(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    ids = landscape_seed(memory_store)
    bad = landscape_json("9999.99999", ids[0][0])
    client, _ = make_bedrock_client(
        [
            make_message(plan_json()),
            make_message(direction_json("A")),
            make_message(direction_json("B")),
            make_message(direction_json("C")),
            make_message(bad, model=SONNET),
            make_message(bad, model=SONNET),  # repair returns the same invented id
        ]
    )

    with pytest.raises(UngroundedCitations):
        map_topic("kv cache", client=client, store=memory_store, embedder=fake_embedder)
