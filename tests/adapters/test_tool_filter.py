"""The OpenAI tools integration: onboard on chat logs, then filter a live tool list."""

from fake_logs import openai_tools, write_openai_logs
from test_adapter_discovery import responder

import tabagent
from tabagent.adapters import OpenAIMessagesAdapter
from tabagent.integrations.openai_tools import ToolFilter
from tabagent.tabschema import TabSchemaConfig
from tabagent.tabschema.llm import ScriptedLLM


def test_tool_filter_end_to_end(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_openai_logs(logs, n=40)
    cfg = TabSchemaConfig(n_analyze=2, workers=2, pair_embeddings=False)
    tabagent.onboard(
        logs,
        adapter=OpenAIMessagesAdapter(),
        llm=ScriptedLLM(responder),
        config=cfg,
        run_dir=tmp_path / "run",
        n_splits=3,
        iterations=40,
        out=tmp_path / "head",
    )
    tools = openai_tools()
    messages = [
        {"role": "system", "content": "You are a helpful travel and email assistant."},
        {"role": "user", "content": "find a flight to rome and book it"},
    ]
    f = ToolFilter(tmp_path / "head", k=2, always=["send_email"])
    kept = f(messages, tools)
    names = [t["function"]["name"] for t in kept]
    assert len(names) <= 3 and "send_email" in names and f.last["choice"]
    assert all(t in tools for t in kept)
    f.head.gate_threshold = float("inf")  # everything deferred -> the full list goes to the LLM
    assert len(f(messages, tools)) == len(tools)
