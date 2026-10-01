# Contributing to TabAgent

Thank you for helping. Bug reports, new log formats, decision kinds and benchmark loaders are all welcome.
Loaders for the public OAK and BPO benchmarks are the most wanted addition.

## Set up

```bash
uv sync --group dev        # the locked environment, with pytest, ruff and pre-commit
uv run pre-commit install  # ruff, file hygiene and gitleaks on every commit
```

## Before you open a pull request

```bash
make check   # ruff lint and format check, then the CPU test suite (no network, no GPU)
make docs    # only if you touched docs/ or mkdocs.yml: the strict documentation build
```

Tests that need the network, a GPU or a long run carry the `network`, `gpu` and `slow` markers and are left out
of `make test`. Add a test next to the code you change. A change to a metric, a fold or a default that moves a
paper number must also update `experiments/paper_cells.yaml` and say why in the pull request.

## Common contributions

| You want to | Start from |
|---|---|
| read a new agent log format | `src/tabagent/adapters/`: implement the adapter protocol described in `adapters/__init__.py`, register it in `detect.py`, and add a fabricated log to `tests/adapters/fake_logs.py` |
| add a public benchmark | `src/tabagent/datasets/`: pin the upstream revision and each file's SHA-256 in a source manifest under `datasets/manifests/`, add a frozen card under `src/tabagent/cards/`, and list the license in `NOTICE` |
| add a classifier head | `TabHead(estimator=...)` takes a `seed -> classifier` factory (`fit`, `predict_proba`, `classes_`); `experiments/figures/head_ablation.py` compares heads on a fixed card |
| report a result that differs from the paper | open an issue with the command, the environment (`uv pip freeze`) and the output |

## Data

Never commit benchmark data, agent trajectories, model weights or API keys. Datasets are downloaded at run time
from their pinned upstream releases, and test fixtures are small, fabricated and marked as such.

## License

By contributing you agree that your contribution is licensed under the Apache License 2.0.
