# ArXiv Atlas

[![CI](https://github.com/IamTusharYadav/arxiv-atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/IamTusharYadav/arxiv-atlas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*A living map of AI research, drawn nightly by agents.*

ArXiv Atlas is an autonomous research intelligence system. Every night, an ingestion pipeline
pulls new papers from arXiv (cs.AI, cs.LG, cs.CL), embeds them locally, and links them into a
semantic graph. Ask a research question and a hand-built agent loop running on AWS Bedrock
returns a cited brief plus an interactive graph of the surrounding literature, with the full
agent trace and token cost attached to every answer.

Atlas is equally an exercise in operating an LLM system: a golden evaluation set gates every
merge, a fail-closed budget guard caps daily spend, and a public status page tracks eval
scores, corpus size, and month-to-date cost.

## Status

Early development. The execution plan lives in
[`arxiv-atlas-execution-plan.md`](arxiv-atlas-execution-plan.md); current progress is tracked
in [`docs/implementation-status.md`](docs/implementation-status.md).

## Why not citation graphs?

Tools like Connected Papers build citation graphs over a static corpus. A paper published this
week has zero citations and is invisible exactly when it matters most. Atlas links papers by
embedding similarity instead, so brand-new work is connected from day one. The tradeoff:
semantic proximity is not intellectual lineage. See [ADR 0001](docs/adr/0001-scope.md) for the
scope this implies.

## Development

Requires [uv](https://docs.astral.sh/uv/). It provisions Python 3.12 automatically.

```sh
uv sync
uv run pre-commit install

# checks (or `make check` where make is available)
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

## Architecture decisions

Recorded in [`docs/adr/`](docs/adr/). Start with
[0002: framework-free agent harness](docs/adr/0002-framework-free-harness.md).

## License

[MIT](LICENSE)
