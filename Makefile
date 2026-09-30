.PHONY: setup hooks lint format test check docs data reproduce

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

check: lint test

docs:
	uv run --group docs mkdocs build --strict

data:
	uv run tabagent datasets --all --download

reproduce:
	uv run --extra embed --group repro python experiments/reproduce.py --download
