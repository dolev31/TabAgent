"""`tabagent.onboard`: agent logs + an LLM -> an evaluated, saved, reloadable, servable decision head."""

import inspect
import warnings

import pandas as pd
import pytest
from test_discover import TOOLS, adapter, responder, write_logs

import tabagent
from tabagent import DecisionSpec, DecisionTable, Feature, FeatureCard, TabAgent, TabHead
from tabagent.api import DryRun, TabAgentHead
from tabagent.tabschema import TabSchemaConfig
from tabagent.tabschema.compile import ExtractorRunner
from tabagent.tabschema.llm import BudgetExceeded, ScriptedLLM


@pytest.fixture
def logs(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    write_logs(d)
    return d


def test_dry_run_calls_no_llm(logs):
    dry = tabagent.onboard(logs, "ShortlisterAgent", adapter=adapter(), dry_run=True, n_analyze=5)
    assert isinstance(dry, DryRun)
    assert dry.n_successful == 44 and dry.n_decision_points == 44 and dry.n_analyzed == 5
    assert dry.est_input_tokens > 0 and dry.est_cost_usd is None and "ShortlisterAgent" in dry.components
    assert "no price for gpt-5.4" in str(dry)
    priced = tabagent.onboard(
        logs, "ShortlisterAgent", adapter=adapter(), dry_run=True, prices={"gpt-5.4": (1.0, 8.0)}
    )
    assert priced.est_cost_usd > 0


def test_decision_component_must_be_chosen_when_ambiguous(logs):
    with pytest.raises(ValueError, match="decision="):
        tabagent.onboard(logs, adapter=adapter(), dry_run=True)


def test_onboard_end_to_end(logs, tmp_path):
    llm = ScriptedLLM(responder, model="scripted")
    cfg = TabSchemaConfig(n_analyze=3, workers=2, pair_embeddings=False)
    head = tabagent.onboard(
        logs,
        "ShortlisterAgent",
        adapter=adapter(),
        llm=llm,
        config=cfg,
        run_dir=tmp_path / "run",
        n_splits=4,
        llm_cost_per_call=0.05,
        out=tmp_path / "head",
        iterations=40,
    )
    assert isinstance(head, TabAgentHead) and head.trust_extractor is False  # sandboxed by default
    assert "TabAgent" in head.report.results.index and any("recipe" in n for n in head.report.notes)
    cost = next(n for n in head.report.notes if "LLM cost" in n)
    assert "cost unknown (no price for scripted" in cost and "input" in cost and "$0.00" not in cost
    assert {"user_intent", "plan_note"} <= set(head.card.feature_names)

    prefix = {
        "intent": "find a flight to rome and book it",
        "steps": [
            {"name": "TaskAnalyzerAgent", "data": "apps needed for: find a flight to rome and book it"},
            {"name": "PlanControllerAgent", "data": "plan: search then book"},
        ],
    }
    out = head.decide(prefix, list(TOOLS), k=2)
    assert set(out["candidates"]) == {"search_flights", "book_flight"} and out["confidence"] >= 0
    assert out["deferred"] is False

    again = tabagent.load(tmp_path / "head", trust_extractor=True)
    assert again.decide(prefix, list(TOOLS), k=2)["candidates"] == out["candidates"]
    (tmp_path / "head" / "extract.py").write_text("features = {}\n")
    with pytest.raises(ValueError, match="hash"):
        tabagent.load(tmp_path / "head")


def test_gate_calibration(logs, tmp_path):
    llm = ScriptedLLM(responder, model="scripted")
    cfg = TabSchemaConfig(n_analyze=2, workers=2, pair_embeddings=False)
    head = tabagent.onboard(
        logs,
        "ShortlisterAgent",
        adapter=adapter(),
        llm=llm,
        config=cfg,
        run_dir=tmp_path / "run",
        n_splits=3,
        gate_coverage=0.8,
        iterations=40,
    )
    assert head.gate_threshold is not None
    assert head.report.cv is not None


def test_extractor_is_sandboxed_unless_trusted():
    assert inspect.signature(tabagent.onboard).parameters["trust_extractor"].default is False
    assert ExtractorRunner("features = {}").trust_extractor is False


def test_budget_without_a_price_fails_before_any_llm_call(logs, tmp_path):
    llm = ScriptedLLM(responder, model="scripted")
    with pytest.raises(ValueError, match="no price is known for model 'scripted'"):
        tabagent.onboard(
            logs, "ShortlisterAgent", adapter=adapter(), llm=llm, max_cost_usd=5.0, run_dir=tmp_path / "r"
        )
    assert llm.calls == []


def test_budget_with_a_price_is_enforced(logs, tmp_path):
    llm = ScriptedLLM(responder, model="scripted")
    with pytest.raises(BudgetExceeded):
        tabagent.onboard(
            logs,
            "ShortlisterAgent",
            adapter=adapter(),
            llm=llm,
            config=TabSchemaConfig(workers=1, max_cost_usd=1e-9, prices={"scripted": (1.0, 1.0)}),
            run_dir=tmp_path / "r",
        )
    assert len(llm.calls) == 1  # the first priced call trips the budget


def test_conflicting_arguments_are_rejected(logs):
    with pytest.raises(ValueError, match="config"):
        tabagent.onboard(logs, "ShortlisterAgent", adapter=adapter(), config=TabSchemaConfig(), n_analyze=3)
    with pytest.raises(ValueError, match="single"):
        tabagent.onboard(logs, "ShortlisterAgent", adapter=adapter(), kind="single", gate_coverage=0.9)


def test_in_process_extractor_failures_are_counted_and_warned():
    runner = ExtractorRunner("features = {'n': len(trajectory['steps'])}", trust_extractor=True)
    with pytest.warns(RuntimeWarning, match="failed on 1 of 2"):
        out = runner.run([{"steps": [1, 2]}, {"no_steps": True}])
    assert out == [{"n": 2}, None] and (runner.n_runs, runner.n_failed) == (2, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert runner.run([{"steps": []}]) == [{"n": 0}]  # no failure, no warning


def _single_class_head():
    spec = DecisionSpec("single", "task", "label", decision=("decision_id",))
    texts = ["book a flight to rome", "cancel my order", "play a jazz song", "book a flight to oslo"] * 8
    df = pd.DataFrame(
        {
            "task": [f"t{i}" for i in range(len(texts))],
            "decision_id": [f"t{i}#0" for i in range(len(texts))],
            "intent": texts,
            "label": [t.split()[0] for t in texts],
        }
    )
    card = FeatureCard("ops", spec, [Feature("intent", "state", "text", "extractor")])
    agent = TabAgent(card, head=TabHead("single", iterations=30)).fit(DecisionTable(df, spec))
    code = "features = {'intent': trajectory['intent']}\n"
    return TabAgentHead(agent, "Classifier", code, trust_extractor=True)


def test_single_class_head_decides_with_class_probabilities():
    head = _single_class_head()
    out = head.decide({"intent": "please book a flight to lima"})
    assert out["choice"] == "book" and set(out["scores"]) == {"book", "cancel", "play"}
    assert out["confidence"] == pytest.approx(max(out["scores"].values())) and out["deferred"] is False
    assert head.decide({"intent": "play a song"}, defer_below=1.01)["deferred"] is True
    with pytest.raises(ValueError, match="no candidates"):
        head.decide({"intent": "x"}, ["book", "play"])
