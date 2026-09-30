"""Deploy a saved head: a live decision in-process, the HTTP service, or a tool filter in front of your LLM.

    python examples/04_deploy.py my_head                    # a head saved by tabagent.onboard (03)
    python examples/04_deploy.py /tmp/tabagent_ranker_xyz   # a TabAgent artifact (the path 02 prints)

The two kinds take different inputs (docs/deployment.md). An onboarded head carries the compiled TabSchema
extractor, so it decides from the raw trajectory prefix; it also backs the HTTP service and the tool filter.
A TabAgent artifact has no extractor, so it decides on rows that already carry the columns it was trained on.
"""

import sys

import pandas as pd

import tabagent
from tabagent import DecisionTable, TabAgent
from tabagent.integrations.openai_tools import ToolFilter

path = sys.argv[1] if len(sys.argv) > 1 else "my_head"
TOOLS = ["search_flights", "book_flight", "get_weather", "send_email"]
REQUEST = "find me a flight to rome and book it"

head = tabagent.load(path, trust_extractor=True)  # trust only an extractor you have reviewed

if isinstance(head, TabAgent):
    # A TabAgent artifact, e.g. the one 02_bring_your_own_table.py saves: one row per candidate with the
    # columns it was trained on, and the decision and candidate ids in the columns its spec names.
    rows = pd.DataFrame(
        {
            "request": REQUEST,
            "tool": TOOLS,
            "tool_description": [
                "search flights between two airports",
                "book a seat on a flight",
                "weather forecast for a city",
                "send an email message",
            ],
        }
    )
    spec = head.card_.spec
    rows = rows.assign(**{spec.group: "new-task", spec.candidate: rows["tool"]})
    print(head.decide(DecisionTable(rows, spec, has_labels=False), k=2))
else:
    # 1. In-process: the prefix is the trajectory so far, in the same format the head was trained on.
    prefix = {"steps": [{"name": "user", "content": REQUEST, "tool_calls": []}]}
    print(head.decide(prefix, TOOLS, k=2))

    # 2. Over HTTP (pip install -e ".[serve]"):  tabagent serve my_head --port 8077
    #    POST /decide {"prefix": {...}, "candidates": [...], "k": 9}

    # 3. In front of a tool-calling LLM: send the model only the tools the head shortlists.
    tool_filter = ToolFilter(head, k=9)
    messages = [{"role": "user", "content": REQUEST}]
    tools = [{"type": "function", "function": {"name": n, "description": n.replace("_", " ")}} for n in TOOLS]
    print([t["function"]["name"] for t in tool_filter(messages, tools)])
    # client.chat.completions.create(model=..., messages=messages, tools=tool_filter(messages, tools))
