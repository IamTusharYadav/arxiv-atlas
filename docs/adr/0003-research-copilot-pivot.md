# ADR 0003: research copilot pivot, landscape-first product

Date: 2026-07-16
Status: accepted

## Context

v0.1.0 shipped question-in, cited-brief-out. That is useful but undifferentiated: the primary
output is still "read these papers". The product direction shifts to understanding-first: a
visitor names a topic and gets a structured map of the research area (what it is, the major
directions, how activity is distributed, where to start reading, what remains open) before
any individual paper is opened. Papers become the drill-down, not the landing.

The pivot must survive contact with what the system actually has:

- The corpus is a rolling ~12-month window of cs.AI/cs.LG/cs.CL **abstracts**. There is no
  full text and, per ADR 0001, no citation data. Nightly ingestion keeps it current.
- Edges are embedding similarity, not intellectual lineage.
- The unit economics are fixed: fail-closed daily cap, per-run caps.

Several parts of the copilot vision cannot be built honestly from that data, and pretending
otherwise would fabricate scholarship:

- "Which papers cite/extend/critique X" requires citation data we do not have. Similarity
  plus publication order is a real signal ("earlier/later related work") but it is not
  citation, and the UI must not dress it up as one.
- "Foundational papers" of a field are mostly older than the window (nothing from 2017 is
  in a 12-month corpus). A landscape over this corpus maps **current activity**, and its
  reading order picks the best entry points among recent papers (surveys score well here),
  not the historical canon.
- Reading difficulty and reading-time estimates from an abstract alone are pseudo-precision.
  Cut entirely rather than shown with false confidence.

## Decision

Build the landscape as a new first-class pipeline beside `ask()`, grounded in the same
corpus with the same citation-validation discipline:

- `POST /api/landscape {topic}`: plan (reusing the planner's scope gate and subquery
  decomposition) -> wide retrieval -> k-means clustering of the stored embeddings (pure
  math, no model spend) -> one Haiku call per cluster to name the direction and its problem
  -> a computed per-direction monthly activity timeline (no model call) -> one Sonnet call
  synthesizing overview, key ideas, reading order and open problems, citing only retrieved
  ids, validated with the same invented-citation check and single repair as the brief path.
- Everything reuses the existing budget/trace harness (`RunContext`), the async job
  infrastructure (jobs gain a `kind`), the semantic response cache (entries gain a `kind`
  so a topic landscape and a question brief can never serve each other), and the daily
  budget guard. Per-landscape cap: $0.30 (roughly two rounds of Haiku labeling plus one
  Sonnet synthesis; measured cost expected well under it).
- The UI leads with the landscape ("map a topic"); the Q&A brief remains as the second mode
  ("ask a question"). Both share progress, trace, and the graph explorer.
- Framing is explicit in-product: the landscape describes the last ~12 months of activity,
  and relations shown are semantic, not citations.

Out of scope for this pivot until measured need (v2 candidates, same rule as ADR 0001):
citation-edge ingestion (Semantic Scholar / OpenAlex would supply it, at the price of a new
external dependency and rate limits), per-paper structured "paper intelligence" cards,
typed relation labels between specific papers, difficulty estimation, learning paths that
span beyond the corpus window.

## Consequences

The landscape is honest but bounded: it cannot narrate a decade of intellectual history,
and its "start here" is the best entry point among recent work, not Attention Is All You
Need. In exchange, every sentence stays grounded in indexed abstracts, the cost model
survives, and no citation graph dependency lands before its value is proven. ADR 0001's
fence stays for data (abstracts, three categories, semantic edges); this ADR widens the
product shape on top of that data, not the data itself.
