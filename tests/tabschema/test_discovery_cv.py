"""`tabagent.cross_validate_discovery`: TabSchema discovery inside every training fold (paper Sec. 4.3),
with a scripted LLM (no network)."""

import pytest
from test_discover import adapter, responder, write_logs

import tabagent
from tabagent.evaluation import DiscoveryCVResult
from tabagent.tabschema import TabSchemaConfig
from tabagent.tabschema.llm import BudgetExceeded, ScriptedLLM
from tabagent.tabschema.orchestrator import TabSchema


@pytest.fixture
def logs(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    write_logs(d)
    return d


def _run(logs, tmp_path, llm, **kw):
    # usefulness=False: the usefulness estimate runs in every fold; one test below keeps it
    cfg = kw.pop("config", TabSchemaConfig(n_analyze=2, workers=2, pair_embeddings=False, usefulness=False))
    return tabagent.cross_validate_discovery(
        logs,
        "ShortlisterAgent",
        adapter=adapter(),
        llm=llm,
        config=cfg,
        n_splits=3,
        search=None,
        iterations=40,
        run_dir=tmp_path / "infold",
        **kw,
    )


def test_discovery_runs_on_the_training_fold_only(logs, tmp_path, monkeypatch):
    seen = []
    discover = TabSchema.discover

    def spy(self, trajectories, target, kind="ranking", name=None):
        seen.append({str(t.group) for t in trajectories})
        return discover(self, trajectories, target, kind, name)

    monkeypatch.setattr(TabSchema, "discover", spy)
    # a fold estimates usefulness only to prune (a report nobody reads per fold is skipped)
    cfg = TabSchemaConfig(n_analyze=2, workers=2, pair_embeddings=False, prune_harmful=True)
    res = _run(logs, tmp_path, ScriptedLLM(responder), seeds=(0, 1), config=cfg)

    assert isinstance(res, DiscoveryCVResult) and len(seen) == 3  # once per fold, reused by the second seed
    groups = set(map(str, res.table.groups))
    for f in range(3):
        held_out = {g for g, ff in res.folds.items() if ff == f}
        trained_on = groups - held_out
        assert trained_on in [s & groups for s in seen]  # failed runs of training tasks may come along
        assert not any(s & held_out for s in seen if s & groups == trained_on)
    assert set(res.discoveries) == {0, 1, 2} and res.seeds == [0, 1]
    assert {"user_intent", "plan_note"} <= set(res.discoveries[0].card.feature_names)
    assert res.summary(["p@r"]).loc["p@r", "value"] > 0.9
    assert res.cost_usd is None  # the scripted model has no price
    for d in res.discoveries.values():  # the usefulness estimate saw the fold's training decisions only
        assert d.report.usefulness["decisions"] == d.report.n_decision_points < len(res.table.groups)


def test_held_out_rows_use_their_fold_extractor_and_reruns_resume(logs, tmp_path):
    llm = ScriptedLLM(responder)
    report_only = TabSchemaConfig(n_analyze=2, workers=2, pair_embeddings=False)  # usefulness on, no pruning
    first = _run(logs, tmp_path, llm, config=report_only)
    assert all(d.report.usefulness == {} for d in first.discoveries.values())  # not paid again per fold
    n_calls = len(llm.calls)
    again = _run(logs, tmp_path, llm, config=report_only)
    assert len(llm.calls) == n_calls  # every fold resumes from its run directory
    assert (first.oof[0] == again.oof[0]).all()
    assert len({d.report.run_dir for d in first.discoveries.values()}) == 3


def test_the_budget_covers_all_folds(logs, tmp_path):
    prices = {"scripted": (1.0, 1.0)}
    priced = TabSchemaConfig(n_analyze=2, workers=1, pair_embeddings=False, prices=prices, usefulness=False)
    res = _run(logs, tmp_path / "a", ScriptedLLM(responder, model="scripted"), config=priced)
    per_fold = [d.report.cost["total_usd"] for d in res.discoveries.values()]
    assert res.cost_usd == pytest.approx(sum(per_fold)) and min(per_fold) > 0

    capped = TabSchemaConfig(
        n_analyze=2,
        workers=1,
        pair_embeddings=False,
        prices=prices,
        max_cost_usd=1.5 * max(per_fold),
        usefulness=False,
    )
    llm = ScriptedLLM(responder, model="scripted")
    with pytest.raises(BudgetExceeded):
        _run(logs, tmp_path / "b", llm, config=capped)
    assert len(llm.calls) > 0  # the first fold ran; the second one ran out of budget


def test_recipe_selection_runs_on_each_training_fold(logs, tmp_path):
    cfg = TabSchemaConfig(n_analyze=2, workers=2, pair_embeddings=False, usefulness=False)
    res = tabagent.cross_validate_discovery(
        logs,
        "ShortlisterAgent",
        adapter=adapter(),
        llm=ScriptedLLM(responder),
        config=cfg,
        n_splits=3,
        iterations=30,
        run_dir=tmp_path / "infold",
    )
    assert set(res.recipes) == {0, 1, 2} and all("synth=" in r for r in res.recipes.values())


def test_fold_discoveries_run_concurrently_and_match_sequential_runs(logs, tmp_path, monkeypatch):
    import threading
    import time

    active, peak = [0], [0]
    lock = threading.Lock()
    discover = TabSchema.discover

    def slow(self, *a, **kw):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        try:
            return discover(self, *a, **kw)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(TabSchema, "discover", slow)
    together = _run(logs, tmp_path / "together", ScriptedLLM(responder), fold_workers=3)
    assert peak[0] == 3  # the three folds' discoveries overlapped
    peak[0] = 0
    one_by_one = _run(logs, tmp_path / "sequential", ScriptedLLM(responder), fold_workers=1)
    assert peak[0] == 1
    assert (together.oof[0] == one_by_one.oof[0]).all()
