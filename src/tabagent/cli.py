"""Command line interface.

    tabagent components LOGS                      list the decision components found in agent logs
    tabagent onboard LOGS --decision NAME --out DIR [--dry-run]
    tabagent serve DIR [--port 8077] [--trust-extractor]
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
def _components(args) -> int:
    from tabagent.adapters import detect_adapter

    adapter = detect_adapter(args.logs)
    trajs = [t for t in adapter.load(args.logs) if t.success]
    for name, n in sorted(adapter.components(trajs).items(), key=lambda kv: -kv[1]):
        print(f"{n:6d}  {name}")
    return 0


def _onboard(args) -> int:
    import tabagent
    from tabagent.tabschema.llm import DEFAULT_MODEL, auto_llm

    if (args.price_in is None) != (args.price_out is None):
        raise ValueError("--price-in and --price-out go together")
    llm = None if args.dry_run else auto_llm(args.model)
    model = llm.model if llm is not None else (args.model or DEFAULT_MODEL)
    prices = None if args.price_in is None else {model: (args.price_in, args.price_out)}
    result = tabagent.onboard(
        args.logs,
        args.decision,
        llm=llm,
        model=args.model,
        kind=args.kind,
        out=args.out,
        n_analyze=args.n_analyze,
        max_cost_usd=args.max_cost,
        prices=prices,
        dry_run=args.dry_run,
        search=args.search,
        llm_cost_per_call=args.llm_cost_per_call,
        trust_extractor=args.trust_extractor,
        run_dir=args.run_dir,
        gate_coverage=args.gate_coverage,
    )
    print(result if args.dry_run else result.report)
    if not args.dry_run and args.out:
        print(f"saved to {args.out}")
    return 0


def _serve(args) -> int:
    from tabagent.serve import serve

    serve(args.path, args.host, args.port, args.trust_extractor)
    return 0


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
            "from environment keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...)."
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
    p.add_argument("--n-analyze", type=int, default=10, help="decision points the LLM analyzes")
    p.add_argument(
        "--max-cost", type=float, help="stop if TabSchema's LLM cost exceeds this (USD); needs prices"
    )
    p.add_argument("--price-in", type=float, help="the model's price, USD per 1M input tokens")
    p.add_argument("--price-out", type=float, help="the model's price, USD per 1M output tokens")
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
    p.add_argument("--dry-run", action="store_true", help="estimate the work and cost, call no LLM")
    p.set_defaults(fn=_onboard)

    p = sub.add_parser(
        "serve",
        help="serve a saved head over HTTP",
        description='Serve a head saved by onboard over HTTP (needs pip install -e ".[serve]").',
        formatter_class=_FORMATTER,
        parents=[verbose],
    )
    p.add_argument("path", help="the directory of a head saved by onboard")
    p.add_argument("--host", default="127.0.0.1", help="interface to listen on")
    p.add_argument("--port", type=int, default=8077, help="port to listen on")
    p.add_argument(
        "--trust-extractor", action="store_true", help="run the extractor in-process (reviewed code only)"
    )
    p.set_defaults(fn=_serve)

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
