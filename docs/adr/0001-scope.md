# ADR 0001: v1 scope fence

Date: 2026-07-12
Status: accepted

## Context

The project must ship a working, publicly demoable system within roughly 8 weeks of part-time
work and then run unattended for 4+ months on about $130 of AWS credits. Every adjacent feature
(PDF parsing, citation graphs, more arXiv categories, user accounts) has a plausible argument
for inclusion and a track record of killing similar projects through scope creep.

## Decision

v1 is limited to:

- arXiv categories cs.AI, cs.LG, cs.CL only
- Abstracts and metadata only, no PDF or full-text parsing
- Semantic edges (embedding similarity), not citation edges
- Anonymous, rate-limited access with no accounts or auth
- One deployable API, one ingestion package, one shared core library

Explicitly out of v1 (candidates for v2, decided by measured need, not enthusiasm):

- Full-text or PDF ingestion ("deep-dive mode")
- Graph-traversal retrieval (until this ships, the README says "semantic research graph",
  never "GraphRAG")
- Additional arXiv categories
- Reranker fine-tuning

## Consequences

Some questions will be answerable only at abstract depth, and the graph will not reflect
intellectual lineage the way citations do. Both limitations are documented rather than hidden.
In exchange, the pipeline stays simple enough to run unattended, and the schedule survives
contact with TA duties.
