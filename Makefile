.PHONY: setup hooks lint format test check docs gate pr-ready data reproduce

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

# The contribution bar on this branch's changes (CONTRIBUTING.md#the-bar). BASE is the branch you will merge into.
BASE ?= origin/main
gate:
	uv run pytest -m "not slow and not gpu and not network" --cov=src/tabagent --cov-report=xml -q
	uv run python scripts/pr_gate.py --base $(BASE) --coverage coverage.xml

# Everything CI checks on a pull request, in one command: run it before you push.
pr-ready: lint docs gate

data:
	uv run tabagent datasets --all --download

reproduce:
	uv run --extra embed --group repro python experiments/reproduce.py --download
