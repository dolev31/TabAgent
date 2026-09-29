.PHONY: setup hooks lint format test guard check

setup:
	uv sync --group dev
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -m "not slow and not gpu and not network"

guard:
	uv run python scripts/privacy_guard.py
	uv run python scripts/privacy_guard.py --all-commits

check: lint test guard
