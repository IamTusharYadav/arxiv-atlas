from dataclasses import FrozenInstanceError

import pytest

from atlas_agents.bedrock import HAIKU, SONNET
from atlas_agents.prompts import (
    CHECK,
    EXTRACTOR,
    LANDSCAPE,
    PLANNER,
    RERANKER,
    SYNTHESIZER,
    Prompt,
    get,
)

ALL = [PLANNER, RERANKER, EXTRACTOR, SYNTHESIZER, CHECK]


def test_every_card_binds_a_known_model_and_semver_version() -> None:
    for prompt in ALL:
        assert prompt.model in {HAIKU, SONNET}
        assert prompt.tag == f"{prompt.id}@{prompt.version}"
        assert prompt.version.count(".") == 2


def test_render_substitutes_placeholders() -> None:
    rendered = PLANNER.render(max_subqueries=4)
    assert "1 to 4" in rendered
    assert "{" not in rendered  # placeholder consumed


def test_static_card_render_is_untouched() -> None:
    # No params: the raw system text comes back without touching str.format, so a stray
    # brace in a card could never blow up a static step.
    assert RERANKER.render() == RERANKER.system


def test_synthesizer_runs_on_sonnet_the_rest_on_haiku() -> None:
    assert SYNTHESIZER.model == SONNET
    assert {PLANNER.model, RERANKER.model, EXTRACTOR.model, CHECK.model} == {HAIKU}


def test_prose_cards_carry_the_no_lineage_rail() -> None:
    # The corpus has similarity edges and no citation data, so a brief claiming one paper
    # builds on another is fabrication. It happened (adversarial-citation-counts scored
    # 1/1/1), which is why the rail is pinned rather than left to a future edit.
    for card in (SYNTHESIZER, LANDSCAPE):
        system = card.system.lower()
        assert "builds on" in system
        assert "citation" in system


def test_get_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        get("nonexistent")


def test_prompt_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        PLANNER.version = "9.9.9"  # type: ignore[misc]


def test_registry_isinstance() -> None:
    assert all(isinstance(p, Prompt) for p in ALL)
