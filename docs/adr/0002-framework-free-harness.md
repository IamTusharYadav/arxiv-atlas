# ADR 0002: framework-free agent harness

Date: 2026-07-12
Status: accepted

## Context

The query-time agent loop (plan, retrieve, rerank, extract, synthesize) could be built on
LangChain, LlamaIndex, or LangGraph. The loop itself is small: a state machine with a typed
tool registry, iteration and budget caps, and structured-output parsing with one repair retry.

## Decision

Build the harness by hand in `atlas_agents`, with no orchestration framework in the core path.

Reasons:

1. Control and debuggability. Every token that enters a prompt is visible in one place.
   Budget enforcement must run before each model call; wiring that through a framework's
   callback layers is harder than owning the loop.
2. Token efficiency. Frameworks add prompt scaffolding that is difficult to inspect or trim.
   With a hard per-query cost cap, prompt bytes are a budget line.
3. The loop is the portfolio piece. A readable, tested, hand-built harness demonstrates
   understanding below the framework layer, which is the point of the project.
4. Small surface. The requirements fit in a few hundred lines; a framework dependency is
   larger than the problem.

A LangGraph comparison module may be added later as a documented experiment, which is a
stronger signal than defaulting to it.

## Consequences

We own retries, structured-output repair, and tracing integration ourselves. That code must be
well tested (harness termination, budget math, and parser repair are all unit-tested). We forgo
framework ecosystem features (prebuilt tools, agent templates) that v1 does not need.
