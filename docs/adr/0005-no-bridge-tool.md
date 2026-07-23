# ADR 0005: cutting find_bridge_papers after hand-validation

Date: 2026-07-23
Status: accepted
Supersedes part of: ADR 0004 (which reserved the decision for hand-validation)

## Context

ADR 0004 shipped a versioned read-only surface for the MCP package and left one tool
conditional: `find_bridge_papers`, which takes two topics and returns papers spanning both.
It was built (`GET /api/v1/bridge`, shipped in v1.3.0), and ADR 0004 said plainly that if it
did not convince on hand-validation it ships as four tools, with the honest note that the
scoring was not good enough.

The implementation: embed both topics, pull the top 300 papers per topic, keep any paper
clearing a 0.35 cosine floor on both sides, rank by the weaker of the two similarities. A
0.90 query-query cosine guard was meant to catch two topics that are really the same topic.

## Validation

Thirty topic pairs, each labelled before the run: BRIDGE (a real cross-topic literature
should exist in a 12-month cs.AI/cs.LG/cs.CL window), NONE (it should not), DUP (the same
topic phrased twice). Scored through the deployed route's exact path, same ONNX query
embedder against the live 100k corpus.

**Where it works.** Fourteen of fifteen BRIDGE pairs returned genuinely correct papers, most
of them excellent: RAG x knowledge-graph reasoning returned GraphRAG systems, diffusion x LLM
decoding found the diffusion-language-model literature, quantization x LLM inference,
multi-agent x LLM agents, VLM x autonomous driving, RL x robotic manipulation, GNN x molecular
property prediction, federated learning x differential privacy, speech recognition x LLMs, and
recommender systems x LLMs were all clean. One miss: adversarial robustness x image
classification returned five papers, none of them about adversarial robustness.

**Where it fabricates.** Eleven of twelve NONE pairs returned a full five results with no note
and no empty list: speech recognition x graph neural networks returned explainable
speaker-recognition papers, chain-of-thought x compiler optimization returned prompt-efficiency
papers, neural architecture search x hate speech detection returned AI-content-detector
benchmarks belonging to neither topic, gender bias in word embeddings x convolutional pruning
returned pure embedding papers at 0.842/0.779. The twelfth, topic modeling x quantum computing,
was a mislabel on our side: a real quantum-NLP literature exists in the corpus and the tool
found it.

So the false-positive rate on pairs that genuinely have no bridge is 11 of 11. Every one reads
as a confident answer built from real papers that connect nothing.

**Both guards are dead code, measured.** The 0.35 floor dropped 0 of 17,696 candidates across
all 30 pairs; bge-small puts everything in this corpus above 0.6. The 0.90 overlap guard fired
on two of three duplicate pairs and missed the abbreviation case: cos("large language models",
"LLMs") is 0.584, which is *lower* than six genuinely distinct pairs (quantization x LLM
inference at 0.792, multi-agent x LLM agents at 0.774, gender bias x pruning at 0.750). Query
similarity does not measure topic distinctness, so the guard cannot be repaired by moving it.

**No threshold separates the two cases.** Scoring each paper relative to its topic's own top
hit, the best score per pair:

```
genuine  .973 .963 .929 .929 .925 .925 .924 .908 .908 .907 .905 .873 .868 .867
junk     .912 .861 .850 .842 .841 .823 .818 .812 .811 .756 .732
```

The worst junk pair outranks seven of fourteen genuine ones. Absolute similarity is no better:
that junk pair's weaker side is 0.779, above genuine lows of 0.742, 0.748 and 0.752. Two
independent scales, both overlapping. A pool-intersection variant was also tried and is worse:
the two 300-paper pools overlap by 0 to 4 papers even for the pairs that produce excellent
bridges.

## Decision

Cut `find_bridge_papers`. Remove the tool, the `GET /api/v1/bridge` route, both thresholds
that filter nothing, and their tests. The MCP package ships four tools.

The reasoning is the honesty rail this system is built on. Every other tool's limitation is
cheap to state truthfully: `explore_from_paper` says "related earlier work, never lineage" and
that is simply what the data is. The equivalent note here would be "no bridge exists between
these topics", and the measurement above says we cannot detect that condition. A tool that
answers confidently in exactly the case where it should decline is the same defect class as
the citation-lineage failure that scored 1/1/1 in the eval baseline, except that one was
fixable at the planner's scope gate and this one is geometry, not prompting.


## Consequences

The MCP surface is four tools, all verified against the live corpus, none of which can
fabricate a relationship the corpus does not contain. `/api/v1` loses a route before anyone
depends on it, which is the cheapest possible moment.

What would make a bridge tool viable later, in rough order of cost: a reranker pass over the
candidates (a model reading both abstracts and judging whether the paper really spans the
topics, which reintroduces LLM spend and breaks the budget-free promise of ADR 0004); an
embedding whose similarity scale is not compressed into 0.6-0.9 for an entire field; or real
citation data, where a bridge is a structural fact rather than a geometric guess. None is
justified by measured need today.
