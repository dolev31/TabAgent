"""Onboard a decision component from your own agent logs: provide an LLM key, TabAgent does the rest.

    export OPENAI_API_KEY=...          # plus OPENAI_BASE_URL for Azure OpenAI or an OpenAI-compatible server
    python examples/03_onboard_agent_logs.py ./my_agent_logs --decision ToolShortlist
    python examples/03_onboard_agent_logs.py ./my_agent_logs --max-cost 50 --price-in IN --price-out OUT

Supported formats are detected automatically (OpenAI / Anthropic Messages API / LangChain tool-call logs,
τ²-bench results, CUGA trajectories, generic JSON with a `steps` list). Run `tabagent scan <logs>` to rank
the LLM calls worth replacing, and `tabagent components <logs>` to list the decisions you can replace. The
dry run reports what TabSchema would read and send before any call.
A cost budget (`--max-cost`) needs the model's prices, IN and OUT, in USD per 1M input and output tokens.
"""

import argparse

import tabagent
from tabagent.adapters import detect_adapter

ap = argparse.ArgumentParser()
ap.add_argument("logs")
ap.add_argument("--decision", default=None)
ap.add_argument("--out", default="my_head")
ap.add_argument("--model", default="gpt-5.4", help="the LLM that runs TabSchema")
ap.add_argument("--max-cost", type=float, default=None, help="stop discovery above this LLM cost (USD)")
ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens of --model")
ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens of --model")
args = ap.parse_args()
prices = None if args.price_in is None else {args.model: (args.price_in, args.price_out or 0.0)}
adapter = detect_adapter(args.logs)  # the log format; the head keeps it to build live prefixes as in training
kw = {"adapter": adapter, "model": args.model, "prices": prices}

print(tabagent.onboard(args.logs, args.decision, dry_run=True, **kw))
if input("Proceed with LLM-driven discovery? [y/N] ").strip().lower() == "y":
    head = tabagent.onboard(args.logs, args.decision, max_cost_usd=args.max_cost, gate_coverage=0.9, **kw)
    head.adapter = head.adapter or adapter  # recorded in head.json: decide(run=...) and POST {"run": ...}
    head.save(args.out)
    print(head.report)
    print(f"saved to {args.out}; review {args.out}/extract.py, then: tabagent serve {args.out}")
