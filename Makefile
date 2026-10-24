.PHONY: install lint format typecheck test check env

install:
	uv sync --extra dev
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src

test:
	uv run pytest

check: lint typecheck test

env:
	uv run python scripts/check_env.py
