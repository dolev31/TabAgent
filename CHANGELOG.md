# Changelog

Changes that affect users, newest first. The GitHub release notes list every pull request.

## Unreleased

### Agent logs

- Every adapter reads files through one reader (`tabagent.adapters.logfiles`), and reports what it skipped
  or repaired in its `warnings` (the onboarding report, the dry run, `tabagent scan`):
  - **encodings:** UTF-8 with or without a BOM, UTF-16 and UTF-32; a file that is not UTF-8 is read in the
    locale's encoding or cp1252, with a warning naming it, and only bytes nothing reads become U+FFFD, with
    a count;
  - **damaged JSON:** trailing commas, bad JSONL lines (counted, with their place), JSON Lines in a `.json`
    file and a pretty-printed document in a `.jsonl` file;
  - **other damage:** values nested deeper than 200 levels, broken surrogate pairs;
  - **files:** linked folders, any-case suffixes, unique run ids.
- `tabagent check-logs LOGS` (`tabagent.check_logs`) diagnoses a log folder before an onboarding, with no
  LLM call. It reports the files, the runs and outcomes and, for each decision: candidates, labels missing
  from the candidates (with near-miss names), train/serve prefix parity and the post-decision data check.
  It exits with status 1 when a problem would stop an onboarding.
- From 256 MB of logs, runs are read again from their files when needed instead of being held in memory.
  Every read checks a fingerprint of exactly the bytes the run came from, so a touched file or an appended
  JSON Lines file is fine, and a changed run stops with `LogChangedError`. `lazy=` on `load` and
  `onboard`, and `--no-lazy` on the CLI, choose. Loading a 1.6 GB folder of agent logs peaks at 0.8 GB instead of
  2.7 GB. The decision points of one run share their steps, so memory grows linearly with run length:
  1,500 turns took 598 MB and now take 20 MB.
- `redact=` keeps lazy runs lazy (`Trajectory.with_raw`).
- Adapter fixes:
  - CUGA labels read `call_api` as Python: keywords in any order, names assigned one literal resolved,
    calls in comments and strings no longer counted.
  - Chat runs with tool calls as a string or a single object, Responses `custom_tool_call` and `mcp_call`,
    and a conversation written one message per line are read.
  - A list of runs is no longer lost to one run without messages.
  - A run's `tools` string is never opened as a file path.
  - Tool descriptions no longer depend on file order.
  - A `{tool: argument schema}` catalog is read.
  - Malformed steps, prompts and τ²-bench records no longer crash the adapters.

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
- **Outcome values.** These are read as success, in any case: `true`, a positive finite number or numeric
  string, and the words `yes`, `y`, `pass`, `passed`, `success`, `succeeded`, `successful`, `solved`,
  `resolved` and `ok`.
  - The words `false`, `no`, `n`, `fail`, `failed`, `failure`, `error`, `errored`, `unsolved`,
    `unresolved`, `incorrect`, `wrong`, `timeout`, `none` and `null` mean failure.
  - Any other value means failure and is reported: another word, an empty string, `NaN`, an infinity, an
    object or a list. Chat logs used to count an object or a list as a success. Pass `success=` to read
    such values.
  - A CUGA `score` is read the same way, so `"true"` and `"PASS"` now mean success. A missing or unreadable
    score is a reported failure.
- **`Trajectory`** is still a frozen dataclass: `dataclasses.fields`, `asdict` and `replace` work. `raw`
  may now be read lazily from the run's file, each access returning a fresh copy. `replace` on a lazy
  trajectory returns an in-memory one; use `Trajectory.with_raw` to keep it lazy. The repr no longer prints
  the log.
- **The CUGA and generic adapters read `.jsonl` files and lists of runs** in a `.json` file. They also follow
  linked folders.
- **Folders with several log formats.** Detection reads the format most files hold; the generic adapter
  used to read whichever came first. Runs of other formats are skipped with a warning that names them. An
  adapter given explicitly does the same, and never raises. `on_other_formats="error"` (CLI
  `--on-other-formats error`) raises `MixedFormatsError` instead. Chat messages without an assistant turn
  (prompt templates, few-shot files) are not runs.
- **`adapter.load(path)` raises `ValueError`** when it reads files but finds no run in them, saying what it
  skipped. It used to return an empty list.
- **Prefixes of one run share their step objects.** Treat `DecisionPoint.prefix` as read-only, and copy it
  before changing it. `success=` and `labels_fn` functions get a copy of the run.
