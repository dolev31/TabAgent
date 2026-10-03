"""Command line interface.

    tabagent scan LOGS [--price-in X --price-out Y] [--format text|markdown|json]
                                                  rank the LLM calls in agent logs that TabAgent can replace
    tabagent components LOGS                      list the decision components found in agent logs
    tabagent onboard LOGS --decision NAME --out DIR [--dry-run] [--redact MODULE:FUNCTION]
    tabagent serve DIR|hf://OWNER/REPO [--port 8077] [--trust-extractor | --sandbox warm|subprocess|docker]
                   [--workers N] [--request-timeout S] [--redact MODULE:FUNCTION] [--log-content]
    tabagent verify-head DIR|hf://OWNER/REPO [--sandbox docker] [--require-goldens]
                                                  re-run a saved head's extractor tests (exit 1 on failure)
    tabagent datasets                             list the public benchmark datasets
    tabagent datasets NAME [--download]           describe one dataset, or build it
    tabagent datasets --all --download            fetch and build every public dataset

A user error (a missing file, an unknown dataset, a license to accept, a missing extra, ...) prints one
message and exits with status 2. ``-v`` logs progress; ``-vv`` also shows tracebacks.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

from tabagent.tabschema.llm import BudgetExceeded, MissingCredentials

#: Exceptions that describe a problem with the user's input or environment, not a bug.
USER_ERRORS = (KeyError, OSError, ValueError, ImportError, BudgetExceeded, MissingCredentials)

#: Library messages name Python arguments; on the command line the flags are what the user can change.
_HINTS = (
    (re.compile(r"[Cc]all with download=True"), "Rerun with --download"),
    (re.compile(r"Pass accept_license=True"), "Rerun with --accept-license"),
    (re.compile(r"Pass prices=\{.*?\} \(CLI: (--price-in and --price-out)\)"), r"Pass \1"),
    (re.compile(r"with decision=\.\.\."), "with --decision NAME"),
    (re.compile(r"see tabagent\.datasets\.available\(\)"), "see `tabagent datasets`"),
)


class _Formatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show an option's default only when it has one worth showing."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default in (None, False, argparse.SUPPRESS):
            return action.help
        return super()._get_help_string(action)


_FORMATTER = _Formatter


def _message(err: BaseException) -> str:
    msg = err.args[0] if isinstance(err, KeyError) and err.args else str(err)
    msg = str(msg) or type(err).__name__
    for pattern, repl in _HINTS:
        msg = pattern.sub(repl, msg)
    return msg


def _configure_logging(verbose: int) -> None:
    log = logging.getLogger("tabagent")
    log.setLevel(logging.WARNING if not verbose else logging.INFO if verbose == 1 else logging.DEBUG)
    if not any(getattr(h, "tabagent_cli", False) for h in log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        handler.tabagent_cli = True  # type: ignore[attr-defined]
        log.addHandler(handler)


# --------------------------------------------------------------------------- commands
def _prices(args) -> tuple[float, float] | None:
    if (args.price_in is None) != (args.price_out is None):
        raise ValueError("--price-in and --price-out go together")
    return None if args.price_in is None else (args.price_in, args.price_out)


def _scan(args) -> int:
    from tabagent.scanner import scan

    if args.top < 1:
        raise ValueError("--top must be at least 1")
    report = scan(args.logs, prices=_prices(args), top=args.top)
    if args.format == "json":
        print(report.to_json())
    else:
        print(report.to_markdown() if args.format == "markdown" else report.summary(), end="")
    return 0


def _components(args) -> int:
    from tabagent.adapters import detect_adapter

    adapter = detect_adapter(args.logs)
    trajs = [t for t in adapter.load(args.logs) if t.success]
    for name, n in sorted(adapter.components(trajs).items(), key=lambda kv: -kv[1]):
        print(f"{n:6d}  {name}")
    return 0


def _redact(ref: str | None):
    """The redaction function an import path ``module:function`` names, checked before any work."""
    if ref is None:
        return None
    from tabagent.api import import_object

    try:
        fn = import_object(ref)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"--redact {ref}: cannot import it ({e})") from e
    if not callable(fn):
        raise ValueError(f"--redact {ref}: not a function")
    return fn


def _pairs(items: list[str] | None, flag: str) -> dict[str, str]:
    """``["KEY=VALUE", ...]`` from a repeatable flag -> ``{KEY: VALUE}``."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"{flag} takes KEY=VALUE, not {item!r}")
        out[key.strip()] = value.strip()
    return out


def _model_prices(items: list[str] | None) -> dict[str, tuple[float, float]]:
    """``--price MODEL=IN,OUT`` (USD per 1M input and output tokens), repeatable."""
    out = {}
    for model, value in _pairs(items, "--price").items():
        try:
            pin, pout = (float(x) for x in value.split(","))
        except ValueError:
            raise ValueError(f"--price {model}=IN,OUT takes two numbers, not {value!r}") from None
        out[model] = (pin, pout)
    return out


def _adapter_for(logs: str, catalog: str | None):
    """The adapter for ``logs``, given the tool catalog file when there is one (``tools=`` for chat-message
    and τ²-bench logs, ``catalog=`` for CUGA)."""
    from tabagent.adapters import detect_adapter

    if not catalog:
        return detect_adapter(logs)
    try:
        return detect_adapter(logs, tools=catalog)
    except TypeError:
        return detect_adapter(logs, catalog=catalog)


def _onboard(args) -> int:
    import tabagent
    from tabagent.tabschema import TabSchemaConfig
    from tabagent.tabschema.llm import DEFAULT_MODEL, auto_llm

    price = _prices(args)
    redact = _redact(args.redact)
    adapter = _adapter_for(args.logs, args.catalog)
    llm = None if args.dry_run else auto_llm(args.model)
    model = llm.model if llm is not None else (args.model or DEFAULT_MODEL)
    prices = {**({} if price is None else {model: price}), **_model_prices(args.price)} or None
    settings = {
        "model": args.model,
        "n_analyze": args.n_analyze,
        "max_cost_usd": args.max_cost,
        "prices": prices,
        "stage_models": _pairs(args.stage_model, "--stage-model") or None,
        "fallback_model": args.fallback_model,
    }
    config = TabSchemaConfig(
        **{k: v for k, v in settings.items() if v is not None},
        audit=not args.no_audit,
        verify=not args.no_verify_extractor,
        verify_adapter=not args.no_verify_adapter,
        prune_harmful=args.prune_harmful,
    )
    result = tabagent.onboard(
        args.logs,
        args.decision,
        adapter=adapter,
        llm=llm,
        kind=args.kind,
        config=config,
        dry_run=args.dry_run,
        redact=redact,
        search=args.search,
        llm_cost_per_call=args.llm_cost_per_call,
        trust_extractor=args.trust_extractor,
        run_dir=args.run_dir,
        gate_coverage=args.gate_coverage,
    )
    print(result if args.dry_run else result.report)
    if not args.dry_run and args.out:
        # head.json records the adapter (so live prefixes are built as in training) and the redaction
        # function's import path (so serving can be told to apply it too)
        result.adapter = result.adapter or adapter
        result.redact = result.redact or args.redact
        result.save(args.out)
        print(f"saved to {args.out}")
    return 0


def _serve(args) -> int:
    from tabagent.serve import serve

    _redact(args.redact)
    if args.trust_extractor and args.sandbox == "warm":
        args.sandbox = None  # the default sandbox; the extractor runs in-process
    serve(
        args.path,
        args.host,
        args.port,
        args.trust_extractor,
        sandbox=args.sandbox,
        redact=args.redact,
        allow_unpinned=args.allow_unpinned,
        allow_unchecked_extractor=args.allow_unchecked_extractor,
        workers=args.workers,
        request_timeout=args.request_timeout or None,
        log_format=args.log_format,
        log_content=args.log_content,
        drift=not args.no_drift,
        drift_window=args.drift_window,
        threads=args.threads,
        graceful_timeout=args.graceful_timeout,
    )
    return 0


def _verify_head(args) -> int:
    from tabagent.tabschema.verify import verify_head

    report = verify_head(args.path, sandbox=args.sandbox, require_goldens=args.require_goldens)
    print(report, end="")
    return 0 if report.passed else 1


def _short_license(text: str) -> str:
    return re.split(r"[,(]", text, maxsplit=1)[0].strip()


def _list_datasets() -> int:
    from tabagent import datasets

    rows = [("name", "kind", "license", "access")]
    for name in datasets.names():
        ds = datasets.info(name)
        rows.append((name, datasets.card(name).spec.kind, _short_license(ds.source.license), ds.access))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())
    print(
        "\nDetails: tabagent datasets NAME. Build: tabagent datasets NAME --download (or --all --download)."
    )
    return 0


def _describe_dataset(name: str) -> int:
    from tabagent import datasets

    ds = datasets.info(name)
    src = ds.source
    fields = [
        ("decision", ds.decision),
        ("kind", datasets.card(name).spec.kind),
        ("paper", ds.paper),
        ("description", ds.description),
        ("benchmark", f"{src.benchmark} ({src.citation})"),
        ("source", f"{src.url} @ {src.revision}"),
        ("license", src.license),
        ("access", ds.access),
        ("card", ds.card),
        ("folds", ds.folds or "none"),
        ("notes", src.notes),
    ]
    print(name)
    for key, value in fields:
        if value:
            print(f"  {key + ':':<13}{value}")
    flags = " --download" + (" --accept-license" if ds.requires_license_acceptance else "")
    print(f"\nBuild it with: tabagent datasets {name}{flags}")
    return 0


def _build_dataset(name: str, args) -> str:
    from tabagent import datasets

    table = datasets.load(name, download=args.download, accept_license=args.accept_license)
    return f"{name}: {len(table)} rows, {len(table.groups)} tasks"


def _skip_reason(name: str, accept_license: bool) -> str | None:
    from tabagent import datasets

    ds = datasets.info(name)
    if ds.requires_license_acceptance and not accept_license:
        return f"needs --accept-license to confirm the upstream terms ({_short_license(ds.source.license)})"
    if ds.local_root_env and not os.environ.get(ds.local_root_env):
        return f"needs a local copy: set {ds.local_root_env} (see tabagent datasets {name})"
    return None


def _all_datasets(args) -> int:
    from tabagent import datasets

    failed = 0
    for name in datasets.names():
        reason = _skip_reason(name, args.accept_license)
        if reason is not None:
            print(f"skipped {name}: {reason}")
            continue
        try:
            print(f"built {_build_dataset(name, args)}")
        except USER_ERRORS as e:
            failed += 1
            print(f"failed {name}: {_message(e).splitlines()[0]}")
    return 1 if failed else 0


def _datasets(args) -> int:
    if args.all:
        if args.name:
            raise ValueError("give a dataset NAME or --all, not both")
        return _all_datasets(args)
    if not args.name:
        if args.download:
            raise ValueError("--download needs a dataset NAME or --all")
        return _list_datasets()
    if not args.download:
        return _describe_dataset(args.name)
    print(_build_dataset(args.name, args))
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """The ``tabagent`` argument parser."""
    from tabagent import __version__

    ap = argparse.ArgumentParser(
        prog="tabagent",
        description="TabAgent: replace an LLM agent's closed-set decisions with cheap tabular classifiers.",
        formatter_class=_FORMATTER,
    )
    ap.add_argument("--version", action="version", version=f"tabagent {__version__}")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="log progress (-vv: also tracebacks)")
    verbose = argparse.ArgumentParser(add_help=False)
    verbose.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="log progress (-vv: also tracebacks)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    p = sub.add_parser(
        "scan",
        help="find the LLM calls in agent logs that TabAgent can replace",
        description=(
            "Rank every component of the agent that calls an LLM: its calls per run and, where the logs "
            "record usage, its share of the agent's LLM tokens, latency and cost; whether its outputs come "
            "from a closed set of candidates (the decisions TabAgent replaces), with the evidence; how many "
            "successful runs contain it; and the onboard command for the top candidates. Reads the logs "
            "only: no LLM is called."
        ),
        formatter_class=_FORMATTER,
        parents=[verbose],
    )
    p.add_argument("logs", help="a log file or a directory of logs")
    p.add_argument(
        "--price-in", type=float, help="USD per 1M input tokens, to price every call from its recorded tokens"
    )
    p.add_argument("--price-out", type=float, help="USD per 1M output tokens (with --price-in)")
    p.add_argument("--format", default="text", choices=["text", "markdown", "json"], help="output format")
    p.add_argument("--top", type=int, default=3, help="closed-set candidates to detail, with their next step")
    p.set_defaults(fn=_scan)

    p = sub.add_parser(
        "components",
        help="list the decision components in agent logs",
        description="List the decision components found in the successful runs of agent logs, by count.",
        formatter_class=_FORMATTER,
        parents=[verbose],
    )
    p.add_argument("logs", help="a log file or a directory of logs")
    p.set_defaults(fn=_components)

    p = sub.add_parser(
        "onboard",
        help="discover features, train and evaluate a head from agent logs",
        description=(
            "Discover a feature card with TabSchema, train and evaluate a head, and save it. The LLM comes "
            "from environment keys: OpenAI, Azure OpenAI or any OpenAI-compatible server (OPENAI_API_KEY, "
            "OPENAI_BASE_URL), or Anthropic, Gemini and others through LiteLLM (tabagent[llm])."
        ),
        formatter_class=_FORMATTER,
        parents=[verbose],
    )
    p.add_argument("logs", help="a log file or a directory of logs")
    p.add_argument("--decision", help="the component to replace (see `tabagent components`)")
    p.add_argument(
        "--model", help="LLM for TabSchema (default: GPT-5.4, or the provider found in the environment)"
    )
    p.add_argument(
        "--kind", default="ranking", choices=["ranking", "multilabel", "single"], help="the decision pattern"
    )
    p.add_argument("--out", help="directory for the saved head")
    p.add_argument("--n-analyze", type=int, default=20, help="decision points the LLM analyzes")
    p.add_argument(
        "--max-cost", type=float, help="stop if TabSchema's LLM cost exceeds this (USD); needs prices"
    )
    p.add_argument("--price-in", type=float, help="the model's price, USD per 1M input tokens")
    p.add_argument("--price-out", type=float, help="the model's price, USD per 1M output tokens")
    p.add_argument(
        "--stage-model",
        action="append",
        metavar="STAGE=MODEL",
        help="a model for one TabSchema stage (analyzer, extractor, validator, aggregator, judges, "
        "meta_judge, auditor); repeatable",
    )
    p.add_argument(
        "--fallback-model",
        help="asked once, for any stage, when the provider's content filter stops an answer",
    )
    p.add_argument(
        "--price",
        action="append",
        metavar="MODEL=IN,OUT",
        help="a model's price, USD per 1M input and output tokens (for --stage-model and --fallback-model "
        "models); repeatable",
    )
    p.add_argument(
        "--catalog",
        help="a tool catalog file (OpenAI or Anthropic tool schemas, MCP tools, or {name: description}): the "
        "candidates' names, descriptions and argument hints, when the logs do not carry them",
    )
    p.add_argument("--search", default="fast", choices=["fast", "full"], help="training recipe search grid")
    p.add_argument(
        "--llm-cost-per-call", type=float, help="cost of the replaced LLM call (USD), for the report"
    )
    p.add_argument(
        "--trust-extractor",
        action="store_true",
        help="run the LLM-written extractor in-process instead of in the sandbox (review it first)",
    )
    p.add_argument("--run-dir", help="TabSchema run directory (default under $TABAGENT_CACHE/runs)")
    p.add_argument(
        "--gate-coverage", type=float, help="calibrate the confidence gate to keep this share of decisions"
    )
    p.add_argument(
        "--redact",
        metavar="MODULE:FUNCTION",
        help="a function applied to every trajectory before any LLM call (an import path, e.g. "
        "mypkg.privacy:scrub); recorded in head.json, and applied when serving only if passed again",
    )
    p.add_argument(
        "--no-audit",
        action="store_true",
        help="skip the OutputAuditor, the LLM check of the extracted values (saves its LLM calls)",
    )
    p.add_argument(
        "--no-verify-extractor",
        action="store_true",
        help="skip the compile-time checks of the extracted values (determinism, outcome and future "
        "invariance, robustness)",
    )
    p.add_argument(
        "--no-verify-adapter",
        action="store_true",
        help="skip only the check that the log adapter's prefixes hold no post-decision data (it runs "
        "before any LLM call and stops a leaking adapter); for an adapter whose prefixes hold the decision "
        "step's own input, declare decision_input_keys instead",
    )
    p.add_argument(
        "--prune-harmful",
        action="store_true",
        help="drop the features whose removal clearly helps in grouped CV on the onboarding rows (by "
        "default the usefulness estimate is a report only)",
    )
    p.add_argument("--dry-run", action="store_true", help="estimate the work and cost, call no LLM")
    p.set_defaults(fn=_onboard)

    p = sub.add_parser(
        "serve",
        help="serve a saved head over HTTP",
        description='Serve a head saved by onboard over HTTP (needs pip install "tabagent[serve]").',
        formatter_class=_FORMATTER,
        parents=[verbose],
    )
    p.add_argument("path", help="a head saved by onboard: its directory, or hf://OWNER/REPO[@REVISION]")
    p.add_argument("--host", default="127.0.0.1", help="interface to listen on")
    p.add_argument("--port", type=int, default=8077, help="port to listen on")
    p.add_argument(
        "--trust-extractor", action="store_true", help="run the extractor in-process (reviewed code only)"
    )
    p.add_argument(
        "--sandbox",
        default="warm",
        metavar="{warm,subprocess,docker,docker:IMAGE}",
        help="where the untrusted extractor runs: persistent resource-limited worker processes (warm), a "
        "fresh resource-limited subprocess per decision, or Docker with no network and a read-only "
        "filesystem",
    )
    p.add_argument(
        "--workers", type=int, help="warm extractor workers, i.e. extractions at once (default min(4, CPUs))"
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="seconds the extractor gets per decision; a later answer is deferred to the LLM (0: no limit)",
    )
    p.add_argument("--threads", type=int, default=8, help="decisions computed at once")
    p.add_argument("--log-format", default="json", choices=["json", "text"], help="log record format")
    p.add_argument(
        "--log-content",
        action="store_true",
        help="also log each request's prefix, run, candidates, answer and error message (off: the agent's "
        "data stays out of the logs)",
    )
    p.add_argument("--no-drift", action="store_true", help="do not monitor feature drift")
    p.add_argument("--drift-window", type=int, default=500, help="recent decisions compared for drift")
    p.add_argument(
        "--graceful-timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="on SIGTERM or Ctrl+C, how long in-flight decisions may finish before the workers stop",
    )
    p.add_argument(
        "--redact",
        metavar="MODULE:FUNCTION",
        help="the redaction function the head was onboarded with, applied to every run before its prefix",
    )
    p.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="serve a head whose card pins no hash for extract.py (saved by an earlier version); review "
        "extract.py first, and save the head again to pin it",
    )
    p.add_argument(
        "--allow-unchecked-extractor",
        action="store_true",
        help="run, still in the sandbox, an extract.py that fails the extraction rules (a head saved by an "
        "earlier version that imports time, say); review it first: the sandbox is not a security boundary",
    )
    p.set_defaults(fn=_serve)

    p = sub.add_parser(
        "verify-head",
        help="re-run a saved head's extractor tests, for CI and deployment gates",
        description=(
            "Check a head saved by onboard: extract.py matches the card's hash, and re-running it in the "
            "sandbox on the golden prefixes of extractor_tests.json reproduces every recorded feature, "
            "twice, with outcome fields making no difference. Exits with status 1 when a check fails."
        ),
        formatter_class=_FORMATTER,
        parents=[verbose],
    )
    p.add_argument("path", help="a head saved by onboard: its directory, or hf://OWNER/REPO[@REVISION]")
    p.add_argument(
        "--sandbox",
        default="subprocess",
        metavar="{subprocess,docker,docker:IMAGE}",
        help="where the extractor runs (docker for a head you do not trust, e.g. from the Hub)",
    )
    p.add_argument(
        "--require-goldens",
        action="store_true",
        help="fail when the head has no golden prefixes (by default that passes with a warning: a head "
        "onboarded with goldens=0 carries no trajectory text)",
    )
    p.set_defaults(fn=_verify_head)

    p = sub.add_parser(
        "datasets",
        help="list, describe or build the public benchmark datasets",
        description=(
            "Without arguments, list the datasets. With NAME, describe it; add --download to build it. "
            "--all --download fetches and builds every dataset that needs no license opt-in or local copy."
        ),
        formatter_class=_FORMATTER,
        parents=[verbose],
    )
    p.add_argument("name", nargs="?", help="a dataset name, for example skillsbench/skill-select")
    p.add_argument("--all", action="store_true", help="every registered dataset")
    p.add_argument("--download", action="store_true", help="fetch missing source files and build the table")
    p.add_argument(
        "--accept-license", action="store_true", help="confirm the upstream terms of datasets that ask for it"
    )
    p.set_defaults(fn=_datasets)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns the exit status (0 success, 1 partial failure, 2 user error)."""
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.fn(args)
    except USER_ERRORS as e:
        if args.verbose > 1:
            raise
        print(f"tabagent {args.cmd}: error: {_message(e)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("tabagent: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
