from pydantic import BaseModel, model_validator

from atlas_agents.bedrock import BedrockClient
from atlas_agents.harness import RunContext
from atlas_agents.prompts import PLANNER

MAX_SUBQUERIES = 4


class Plan(BaseModel):
    in_scope: bool
    subqueries: list[str]
    stop_criterion: str
    scope_note: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "Plan":
        if self.in_scope and not (1 <= len(self.subqueries) <= MAX_SUBQUERIES):
            raise ValueError(f"in-scope plans need 1 to {MAX_SUBQUERIES} subqueries")
        if not self.in_scope and self.subqueries:
            raise ValueError("out-of-scope plans must not have subqueries")
        return self


def plan_query(client: BedrockClient, ctx: RunContext, question: str) -> Plan:
    plan, completion = client.complete_structured(
        model=PLANNER.model,
        system=PLANNER.render(max_subqueries=MAX_SUBQUERIES),
        prompt=f"<question>{question}</question>",
        output_type=Plan,
        max_tokens=500,
    )
    summary = f"{len(plan.subqueries)} subqueries" if plan.in_scope else "out of scope"
    ctx.record("planner", summary, completion, model=PLANNER.model, version=PLANNER.version)
    return plan
