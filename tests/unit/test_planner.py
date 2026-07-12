import json

import pytest

from atlas_agents.bedrock import HAIKU, StructuredOutputError
from atlas_agents.harness import RunContext
from atlas_agents.steps.planner import Plan, plan_query
from tests.conftest import make_bedrock_client, make_message


def plan_json(**overrides: object) -> str:
    plan = {
        "in_scope": True,
        "subqueries": ["kv cache quantization", "token eviction policies"],
        "stop_criterion": "at least one paper per method family",
        "scope_note": "",
    }
    plan.update(overrides)
    return json.dumps(plan)


def test_plan_query_returns_plan_and_charges_trace() -> None:
    client, fake = make_bedrock_client([make_message(plan_json())])
    ctx = RunContext()

    plan = plan_query(client, ctx, "What are current approaches to KV cache compression?")

    assert plan.in_scope
    assert plan.subqueries == ["kv cache quantization", "token eviction policies"]
    assert fake.calls[0]["model"] == HAIKU
    prompt = fake.calls[0]["messages"][0]["content"]  # type: ignore[index]
    assert "KV cache compression" in prompt
    assert ctx.trace[0].step == "planner"
    assert ctx.trace[0].summary == "2 subqueries"
    assert ctx.spent_usd > 0


def test_out_of_scope_plan() -> None:
    client, _ = make_bedrock_client(
        [
            make_message(
                plan_json(
                    in_scope=False,
                    subqueries=[],
                    stop_criterion="",
                    scope_note="Gene editing is outside this corpus; ML for biology is nearby.",
                )
            )
        ]
    )
    ctx = RunContext()
    plan = plan_query(client, ctx, "Latest advances in CRISPR?")
    assert not plan.in_scope
    assert plan.subqueries == []
    assert "outside this corpus" in plan.scope_note
    assert ctx.trace[0].summary == "out of scope"


def test_inconsistent_plan_triggers_repair() -> None:
    # First response claims in-scope but gives no subqueries; validator rejects it and
    # the repair round trip returns a consistent plan.
    client, fake = make_bedrock_client(
        [
            make_message(plan_json(subqueries=[])),
            make_message(plan_json(subqueries=["kv cache compression"])),
        ]
    )
    ctx = RunContext()
    plan = plan_query(client, ctx, "KV cache compression?")
    assert plan.subqueries == ["kv cache compression"]
    assert len(fake.calls) == 2
    # The trace charges the aggregate of both attempts.
    assert ctx.trace[0].input_tokens == 200


def test_unrepairable_plan_raises() -> None:
    client, _ = make_bedrock_client(
        [make_message(plan_json(subqueries=[])), make_message(plan_json(subqueries=[]))]
    )
    with pytest.raises(StructuredOutputError):
        plan_query(client, RunContext(), "KV cache compression?")


def test_plan_validator_rejects_out_of_scope_with_subqueries() -> None:
    with pytest.raises(ValueError, match="out-of-scope"):
        Plan(in_scope=False, subqueries=["stray"], stop_criterion="")
