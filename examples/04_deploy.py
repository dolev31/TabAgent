"""Deploy a head: a live decision in-process, the HTTP service, or a tool filter in front of your LLM.

Assumes a head saved by `tabagent.onboard(..., out="my_head")` (see 03_onboard_agent_logs.py).
"""

import tabagent
from tabagent.integrations.openai_tools import ToolFilter

head = tabagent.load("my_head", trust_extractor=True)  # only for an extractor you have reviewed

# 1. In-process: the prefix is the trajectory so far, in the same format the head was trained on.
prefix = {"steps": [{"name": "user", "content": "find me a flight to rome and book it", "tool_calls": []}]}
print(head.decide(prefix, ["search_flights", "book_flight", "get_weather", "send_email"], k=2))

# 2. Over HTTP (pip install "tabagent[serve]"):  tabagent serve my_head --port 8077
#    POST /decide {"prefix": {...}, "candidates": [...], "k": 9}

# 3. In front of a tool-calling LLM: send the model only the tools the head shortlists.
tool_filter = ToolFilter(head, k=9)
messages = [{"role": "user", "content": "find me a flight to rome and book it"}]
tools = [
    {"type": "function", "function": {"name": n, "description": n.replace("_", " ")}}
    for n in ["search_flights", "book_flight", "get_weather", "send_email"]
]
print([t["function"]["name"] for t in tool_filter(messages, tools)])
# client.chat.completions.create(model=..., messages=messages, tools=tool_filter(messages, tools))
