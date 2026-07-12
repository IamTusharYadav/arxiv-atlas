# Evaluation golden set

One YAML file per golden query. The eval runner (Phase B) executes each question against the
agent, and an LLM judge scores the answer using the rubric plus the per-query notes below.
Queries are written before the agent exists so that evals drive development.

## Schema

```yaml
id: kv-cache-compression        # matches the filename, stable forever
category: technical-survey     # technical-survey | comparison | specific-topic |
                               # ambiguous | out-of-scope | adversarial
question: the user question, verbatim
in_scope: true                 # false for queries the agent should decline or redirect
expected_topics:               # keywords a good answer will engage with; hints for the
  - kv cache                   # judge, not an exhaustive checklist
rubric_notes: >
  What a strong answer looks like and the failure modes to penalize.
```

## Conventions

- Ids are kebab-case and never reused or renamed; eval history keys on them.
- The corpus is cs.AI, cs.LG, cs.CL abstracts only. Rubric notes must not demand knowledge
  that cannot come from abstracts.
- The corpus moves nightly, so queries must not pin specific arxiv ids; they describe topics.
- Adversarial and out-of-scope cases are part of the set from day one and grow whenever a
  real failure is observed in production logs.
