# Evaluations

Every claim this system makes about what it can and cannot answer is backed by a measurement that
was run, not asserted. This document is the evidence: how the golden set is built, how the gate
works, and the four cases where running the eval caught something reasoning alone would have missed,
including two where the first fix was wrong and the measurement is what said so.

The scoreboard and how to run the suite are in the [README](../README.md#evaluations) and
[`evals/README.md`](../evals/README.md). This is the reasoning behind the numbers.

## The golden set

Thirty queries, one YAML file each in [`evals/golden/`](../evals/golden/), written before the agent
existed so that evals drive development rather than rationalize it. Six categories:
`technical-survey`, `comparison`, `specific-topic`, `ambiguous`, `out-of-scope`, and `adversarial`.
The last two are honesty probes and are in the set from day one, not bolted on after a failure. They
are expected to be the lowest scores, because a system that scores 5/5/5 on an out-of-scope question
is either lucky or lying.

Two design choices matter more than the count:

- **The set is 30, amended down from the plan's 50.** Fifty was a round number, not a measured
  requirement, and every query costs money on every full run forever. Thirty covers all six
  categories with enough per-category depth to catch a regression; the marginal query past that buys
  little and bills forever.
- **The 15-query subset is category-balanced, not an alphabetical prefix.** A prefix of the 30
  queries would have handed the PR gate three adversarial and three ambiguous cases, zero
  out-of-scope, and one of ten technical surveys, gating on a distribution nothing like the full
  set. `golden_set.subset` round-robins across categories so the subset's proportions track the
  whole set, pinned by tests for coverage and determinism.

## The judge and the gate

Each answer is scored 1-5 by an LLM judge on three dimensions: **relevance**, **faithfulness**, and
**citation correctness**. A merge is blocked when a gated dimension falls more than 0.3 below the
committed baseline. Citation correctness is recorded but does not gate.

Cadence is deliberately not nightly, because every run costs real model spend and the system is run
to a tight cost target. A nightly full run over the whole set would make evals the single largest
recurring line, and even a nightly subset would dominate. The cadence has to match what actually
changes: agent code changes only when the maintainer commits, already covered by the on-PR
`run-evals` label gate; corpus drift is slow enough that a weekly full run catches it. The result is
a weekly full run plus the PR gate. The weekly job appends a row to `evals/history.json` and commits
it back, so the repo is the versioned eval store and the status page reads it at build time.

## The gate earning its keep

These are the cases where running the eval changed the code. All four are in `evals/history.json`.

### adversarial-citation-counts: 1/1/1, and the obvious fix was wrong

A probe asked which paper is most cited and what builds on it. It scored 1/1/1: the brief presented
similarity neighbours as citation lineage. The corpus has abstracts and embedding links and no
citation data, so this is exactly the fabrication the honesty rails exist to prevent.

The first fix was the obvious one, and it was wrong. A no-lineage rail went into the synthesizer
prompt card, the query was rerun, and it scored 1/1/1 again. The judge rationale gave the theory
away by quoting the answer's own preamble, "stopped early": the query never reached the synthesizer
at all. The real mechanism was upstream. The planner's scope gate rejected by subject only, so a
question about citation counts passed as on-topic cs.CL; the check step then could not satisfy a
stop criterion the corpus has no data for, so the loop refined subqueries until it hit the iteration
cap, and `_partial_answer` assembled 21 papers locally, which reads as a confident fabricated
synthesis.

The real fix was the planner card at 1.1.0: the gate now rejects by **kind of data** as well as
subject (citation counts, influence rankings, who-builds-on-whom, author and venue records,
full-text-only results, pre-window work), with an explicit carve-out keeping "how do these lines of
work relate" in scope, since shared problems and methods really are visible in abstracts. The result
was `adversarial-citation-counts` 1/1/1 -> 4/5/5 at $0.0107 -> $0.0012, and `out-of-scope-pre-window`
holding 5/5/5 while dropping $0.0805 -> $0.0011, both now declining in one step instead of grinding
six iterations to the cap. The no-lineage rails on the synthesizer and landscape cards stayed as
defence in depth, pinned by `test_prose_cards_carry_the_no_lineage_rail`, but they were not what
moved the score.

The lesson is the sequence: the honesty defect and a cost smell turned out to be the same bug, the
obvious fix was measured and rejected, and the judge's own rationale pointed at the real cause.

### The planner over-corrected, and only the full run saw it

Planner 1.1.0 fixed the two probes it was aimed at and quietly broke a third.
`ambiguous-make-it-faster`, an in-scope question that only reads as broad, started getting declined,
because a gate now rejecting by "kind of data" over-generalized vagueness into a rejection reason.
This was invisible on the two adversarial probes the change was written for and showed up only across
the whole 30-query run. Planner 1.1.1 added that vagueness is never a reason to decline, only a
missing subject or a missing kind of data, and the query went back to 5/5/5. This is the argument
for running the full set on a change, not just the queries the change was about.

### Judge noise is real at samples=1

The scheduled run uses one judge sample per answer, for cost. At that setting the judge swings about
0.1 on a dimension that is genuinely borderline. `ambiguous-make-it-faster` scored relevance 2 in one
full run and 5 on an immediate rerun with no code change. `empty-corpus-region` declined cleanly at
2/5/5 in one run and padded an answer from weak retrieval at 2/3/4 in the next, again with no code
change. The operating rule that follows: a single low score is a signal to rerun, not evidence of a
regression, and re-baselining should prefer `--samples 3`. It also means the saved baseline, being
the best of three runs, carries roughly 0.1 of optimism per swinging dimension, so the occasional red
scheduled run that a rerun clears is expected, not alarming.

### The bridge tool: worked beautifully, and was cut anyway

`find_bridge_papers` was built, shipped in v1.3.0, and then removed after hand-validation over 30
pre-registered topic pairs against the live corpus. On the 15 pairs with a real cross-topic bridge it
returned genuinely good papers for 14. On the 11 pairs with no bridge it returned five confident
results with no caveat, every time. Both safety thresholds were dead code, and the measurement is
what proved it: the 0.35 similarity floor dropped 0 of 17,696 candidates, and the 0.90 overlap guard
missed `cos("large language models", "LLMs") = 0.584`, a value lower than six genuinely distinct
pairs. No threshold separates the good cases from the fabricated ones, so the honest "no bridge here"
note that every other tool relies on is unbuildable for this one. A tool that works 14 of 15 times
and lies confidently the rest is worse than no tool, because the failures are indistinguishable from
the successes. It was cut; [ADR 0005](adr/0005-no-bridge-tool.md) carries the full numbers.

## What the numbers are, and are not

Being exact about the measurements matters as much as the scores themselves.

- **Baseline** (2026-07-23, planner 1.1.1, the run that matches the committed code): relevance 4.833,
  faithfulness 4.933, citation correctness 4.967, with 27 of 30 queries scoring 5/5/5. Three full
  runs tell the improvement story end to end: 4.6/4.833/4.867 before the scope-gate fix,
  4.7/5.0/5.0 after planner 1.1.0, 4.833/4.933/4.967 after 1.1.1. The baseline deliberately includes
  the honesty failures rather than being taken after fixing them, so a fix shows up as a measured
  delta.
- **Cost**, measured on that run: median query $0.033, mean $0.0338, p95 $0.0895 against the $0.12
  fail-closed cap; $1.01 of agent spend plus $0.134 of judge for the full 30. A multi-round question
  can reach ~$0.068, over half the cap, so hard questions have less headroom than the median
  suggests, which is the cap working as intended.
- **Latency**: median agent loop 31.0s, p95 75.0s. This is dev-machine wall time for the loop, which
  tracks the deployed worker closely (30.0s and 61.8s on two live runs) because both spend nearly all
  of it waiting on Bedrock, not on local compute.
- **The published cost is recurring eval and query spend, not an AWS bill, on purpose.** The
  account's raw invoice does not reflect the true marginal cost of running this, so quoting it would
  be misleading in either direction. The honest number is what the work actually consumes, measured
  per query and per run above.

The through-line across all of it: nothing here is a claim the code was reasoned into being correct.
Each one is a number a run produced, sometimes contradicting the fix that reasoning suggested, and
the number is what the code was changed to satisfy.
