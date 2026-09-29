"""Onboard a decision component from your own agent logs: provide an LLM key, TabAgent does the rest.

    export OPENAI_API_KEY=...          # or OPENAI_BASE_URL for any OpenAI-compatible server
    python examples/03_onboard_agent_logs.py ./my_agent_logs --decision ToolShortlist

Supported formats are detected automatically (OpenAI / Anthropic Messages API / LangChain tool-call logs,
τ²-bench results, CUGA trajectories, generic JSON with a `steps` list). Run `tabagent components <logs>` to
list the decisions you can replace. The dry run reports what TabSchema would read and send before any call.
"""

import argparse

import tabagent

ap = argparse.ArgumentParser()
ap.add_argument("logs")
ap.add_argument("--decision", default=None)
ap.add_argument("--out", default="my_head")
ap.add_argument("--max-cost", type=float, default=50.0)
args = ap.parse_args()

print(tabagent.onboard(args.logs, args.decision, dry_run=True))
if input("Proceed with LLM-driven discovery? [y/N] ").strip().lower() == "y":
    head = tabagent.onboard(
        args.logs, args.decision, out=args.out, max_cost_usd=args.max_cost, gate_coverage=0.9
    )
    print(head.report)
    print(f"saved to {args.out}; review {args.out}/extract.py, then: tabagent serve {args.out}")
