"""`tabagent.onboard`: agent logs + an LLM -> an evaluated, saved, reloadable, servable decision head."""

import pytest
from test_discover import TOOLS, adapter, responder, write_logs

import tabagent
from tabagent.api import DryRun, TabAgentHead
from tabagent.tabschema import TabSchemaConfig
from tabagent.tabschema.llm import ScriptedLLM


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
    assert isinstance(head, TabAgentHead)
    assert "TabAgent" in head.report.results.index and any("recipe" in n for n in head.report.notes)
    assert {"user_intent", "plan_note"} <= set(head.card.feature_names)

    prefix = {
        "intent": "find a flight to rome and book it",
        "steps": [
            {"name": "TaskAnalyzerAgent", "data": "apps needed for: find a flight to rome and book it"},
            {"name": "PlanControllerAgent", "data": "plan: search then book"},
        ],
    }
    out = head.decide(prefix, list(TOOLS), k=2)
    assert set(out["choice"]) == {"search_flights", "book_flight"} and out["confidence"] >= 0

    again = tabagent.load(tmp_path / "head", trust_extractor=True)
    assert again.decide(prefix, list(TOOLS), k=2)["choice"] == out["choice"]
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
