"""Command line interface.

tabagent components LOGS                      list the decision components found in agent logs
tabagent onboard LOGS --decision NAME --out DIR [--dry-run]
tabagent serve DIR [--port 8077] [--trust-extractor]
tabagent datasets [NAME --download]           public benchmark tables (see tabagent.datasets)
"""

from __future__ import annotations

import argparse
import sys


def _components(args) -> int:
    from tabagent.adapters import detect_adapter

    adapter = detect_adapter(args.logs)
    trajs = [t for t in adapter.load(args.logs) if t.success]
    for name, n in sorted(adapter.components(trajs).items(), key=lambda kv: -kv[1]):
        print(f"{n:6d}  {name}")
    return 0


def _onboard(args) -> int:
    import tabagent

    result = tabagent.onboard(
        args.logs,
        args.decision,
        model=args.model,
        kind=args.kind,
        out=args.out,
        n_analyze=args.n_analyze,
        max_cost_usd=args.max_cost,
        dry_run=args.dry_run,
        search=args.search,
        llm_cost_per_call=args.llm_cost_per_call,
    )
    print(result if args.dry_run else result.report)
    if not args.dry_run and args.out:
        print(f"saved to {args.out}")
    return 0


def _serve(args) -> int:
    from tabagent.serve import serve

    serve(args.path, args.host, args.port, args.trust_extractor)
    return 0


def _datasets(args) -> int:
    from tabagent import datasets

    if not args.name:
        print(datasets.available().to_string())
        return 0
    table = datasets.load(args.name, download=args.download, accept_license=args.accept_license)
    print(f"{args.name}: {len(table)} rows, {len(table.groups)} tasks")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tabagent", description="TabAgent: cheap classifiers for agent decisions"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("components", help="list the decision components in agent logs")
    p.add_argument("logs")
    p.set_defaults(fn=_components)

    p = sub.add_parser("onboard", help="discover features, train and evaluate a head from agent logs")
    p.add_argument("logs")
    p.add_argument("--decision", help="the component to replace (see `tabagent components`)")
    p.add_argument("--model", help="LLM for TabSchema (default: from environment keys)")
    p.add_argument("--kind", default="ranking", choices=["ranking", "multilabel", "single"])
    p.add_argument("--out", help="directory for the saved head")
    p.add_argument("--n-analyze", type=int, default=10)
    p.add_argument("--max-cost", type=float, default=100.0, help="stop if the LLM cost exceeds this (USD)")
    p.add_argument("--search", default="fast", choices=["fast", "full"])
    p.add_argument("--llm-cost-per-call", type=float, help="cost of the replaced LLM call, for the report")
    p.add_argument("--dry-run", action="store_true", help="estimate the work and cost, call no LLM")
    p.set_defaults(fn=_onboard)

    p = sub.add_parser("serve", help="serve a saved head over HTTP")
    p.add_argument("path")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument(
        "--trust-extractor", action="store_true", help="run the extractor in-process (reviewed code)"
    )
    p.set_defaults(fn=_serve)

    p = sub.add_parser("datasets", help="list or build public benchmark tables")
    p.add_argument("name", nargs="?")
    p.add_argument("--download", action="store_true")
    p.add_argument("--accept-license", action="store_true")
    p.set_defaults(fn=_datasets)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
