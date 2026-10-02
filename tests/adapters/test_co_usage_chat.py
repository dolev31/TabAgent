"""Step-level chat logs feed the fitted co-usage feature: NextToolCall prefixes record the tools called so far
(``steps[].tool_calls``), and an extractor feature listing them is the co-usage context."""

import json

import pandas as pd
from fake_logs import write_openai_logs

from tabagent import TabAgent, TabHead
from tabagent.adapters import NEXT_TOOL_CALL, OpenAIMessagesAdapter
from tabagent.table import DecisionSpec
from tabagent.tabschema import Feature, FeatureCard
from tabagent.tabschema.compile import ExtractorRunner
from tabagent.tabschema.orchestrator import CANDIDATE, DECISION, GROUP, LABEL, rows_to_table


def test_step_level_chat_logs_feed_co_usage_through_the_extractor(tmp_path):
    """NextToolCall prefixes record the tools called so far (steps[].tool_calls); an extractor feature
    listing them is the co-usage context."""
    code = (
        "calls = [t for s in trajectory['steps'] if s['name'] == 'assistant' for t in s['tool_calls']]\n"
        "features = {'tools_called_so_far': ', '.join(calls)}\n"
    )
    logs = write_openai_logs(tmp_path / "logs", n=24, fail_every=0)
    adapter = OpenAIMessagesAdapter(granularity="step")
    dps = [dp for t in adapter.load(logs) for dp in adapter.decision_points(t, NEXT_TOOL_CALL)]
    spec = DecisionSpec("ranking", GROUP, LABEL, CANDIDATE, decision=(DECISION,))
    card = FeatureCard(
        "chat_co_usage",
        spec,
        [
            Feature("tools_called_so_far", "state", "text", "extractor"),
            Feature("candidate_name", "schema", "cat"),
            Feature("co_usage", "dependency", "num", "fitted:co_usage", {"context": "tools_called_so_far"}),
        ],
    )
    feats = ExtractorRunner(code, trust_extractor=True).run([dp.prefix for dp in dps])
    table = rows_to_table(card, dps, feats, adapter)
    fitted = TabAgent(card, head=TabHead("ranking", iterations=30)).fit(table)
    co = fitted.card_._fitted["co_usage"]
    live = pd.DataFrame(
        {CANDIDATE: ["book_flight", "search_flights"], "tools_called_so_far": "search_flights"}
    )
    book, search = co.transform(live, spec)
    assert book > 0.6 > search  # in the fabricated runs, booking follows a flight search
    assert json.dumps(co.state()[0])  # JSON-able state for the pickle-free artifact
