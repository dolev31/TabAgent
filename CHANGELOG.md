# Changelog

Changes that affect users, newest first. The GitHub release notes list every pull request.

## Unreleased

### Serving

- `tabagent serve` runs the extractor in warm sandbox workers by default (`--sandbox warm`): persistent
  `python -I -S` interpreters with the one-shot sandbox's isolation, so the sandbox adds about 0.1 ms to a
  decision instead of an interpreter start-up. `--sandbox subprocess` keeps one interpreter per decision.
- The service has readiness (`/ready`), versioning (`/version`), Prometheus metrics (`/metrics`), feature drift
  monitoring (`/drift`), batch decisions (`/decide_batch`), structured JSON logs without request content, and
  a request timeout that defers instead of failing. See the production runbook in `docs/deployment.md`.
- `TabAgent.fit` records the training distribution of the head's inputs (`model/feature_stats.json`), which
  drift monitoring compares live decisions with.

### Upgrading

- **Older extractors may be refused at serve time.** A sandbox chosen by name now re-applies the
  security-relevant static checks of the compile step to `extract.py` before running it. The checks are the
  allowed standard-library imports, the forbidden builtins, and no interpreter internals or frame
  introspection. A head saved by an earlier version whose `extract.py` fails them, for example by importing
  `time`, `os` or `signal`, now answers `/ready` with 503 and defers every decision to the LLM. Onboard it
  again, or review `extract.py` and serve it with `--allow-unchecked-extractor`
  (`TabAgentHead.load(..., allow_unchecked_extractor=True)`). The opt-out keeps the extractor in the sandbox
  and logs a warning.
- **Status codes.** A failure of the head other than a `RequestError` is a 500 (a plain `ValueError` from a
  custom head used to be a 400). A `k` below 1 is a 400. A decision past its deadline lists every candidate id
  in input order, with empty `scores`.
- **Drift needs refitted heads.** Heads fitted before this version have no training statistics, and their
  `/drift` reports `unavailable`.
