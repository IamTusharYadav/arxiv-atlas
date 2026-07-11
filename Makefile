# On Windows without make, run the uv commands directly.
.PHONY: install lint format typecheck test check

install:
	uv sync

lint:
	uv run ruff format --check .
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test
