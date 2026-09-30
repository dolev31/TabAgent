"""experiments/reproduce.py and paper_cells.yaml: the cells, the Mind2Web element protocol, the report.

Everything runs on fabricated tables (the toy tool-shortlisting problem of tests/conftest.py and
Mind2Web-shaped rows); nothing is downloaded.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import reproduce as R
from experiments.reproduce import PaperElementSynth, as_read_from_csv, bm25_scores
from tabagent import TabAgent, TabHead, datasets
from tabagent.evaluation.cv import assign_folds
from tabagent.table import DecisionTable

from conftest import make_card, make_table

CARD = datasets.card("mind2web/element")


def element_table(tasks: dict[str, str]) -> DecisionTable:
    """One step per task, three candidates, the first one demonstrated; ``tasks`` maps task -> domain."""
    rows = []
    for g, dom in tasks.items():
        for c in range(3):
            rows.append(
                {
                    "task_uid": g,
                    "annotation_id": g,
                    "instruction": f"Buy item {g}. Pay online.",
                    "website": "site",
                    "domain": dom,
                    "subdomain": "sub",
                    "step_index": 0,
                    "history_action_reprs": "",
                    "backend_node_id": f"{g}-{c}",
                    "elem_tag": "button",
                    "elem_attr_text": f"item {g}" if c == 0 else f"other {c}",
                    "elem_type": "",
                    "elem_is_clickable": 1,
                    "elem_x": float(c),
                    "elem_y": 0.0,
                    "elem_w": 10.0,
                    "elem_h": 5.0,
                    "is_target": int(c == 0),
                    "eval_operation": "CLICK",
                    "eval_op_value": "",
                    "eval_original_op": "CLICK",
                }
            )
    return DecisionTable(pd.DataFrame(rows), CARD.spec)


EVALUATED = {"e1": "Shopping", "e2": "Shopping", "e3": "Travel", "e4": "Shopping"}
POOL = {"e1": "Shopping", "e2": "Shopping", "e3": "Travel", "p1": "Shopping", "p2": "Travel"}


def _augment(synth, train_groups):
    table = element_table(EVALUATED).subset(train_groups)
    card = CARD.clone()
    card.fit(table)
    X = card.transform(table)
    return synth.augment(X, table.labels, table, card)


def test_pool_synth_keeps_the_domain_and_drops_the_test_fold():
    synth = PaperElementSynth(element_table(POOL), "Shopping", EVALUATED, k=2)
    X_syn, y_syn = _augment(synth, ["e1", "e3"])  # test fold: e2, e4
    groups = synth.rows_[2]
    assert set(groups) == {"e1", "e2", "p1"}  # the Shopping rows of the pool, before the fold filter
    kept = [g for g in groups if g not in {"e2", "e4"}]
    assert len(X_syn) == len(kept) == 12  # e1 and p1 (outside the evaluated tasks): 3 rows x k=2 each
    assert set(X_syn["domain"]) == {"Shopping"} and X_syn["instruction"].str.startswith("Task: ").sum() == 6
    assert y_syn.tolist() == [1, 1, 0, 0, 0, 0] * 2  # labels copied, row-major


def test_pool_is_paraphrased_once_and_shared_by_fold_clones():
    synth = PaperElementSynth(element_table(POOL), "Shopping", EVALUATED)
    agent = TabAgent(CARD, head=TabHead("ranking", iterations=5), synth=synth)
    assert agent.clone(1).synth is synth and copy.deepcopy(synth) is synth
    _augment(synth, ["e1", "e3"])
    first = synth.rows_
    X_syn, _ = _augment(synth, ["e2", "e4"])  # another fold reuses the rows, with its own test fold removed
    assert synth.rows_ is first and len(X_syn) == 12  # e2, p1


def test_pool_synth_refuses_a_card_with_fitted_features():
    table = make_table(n_tasks=4)
    card = make_card()
    card.fit(table)
    with pytest.raises(ValueError, match="stateless"):
        PaperElementSynth(table, "x", table.groups).augment(card.transform(table), table.labels, table, card)


def test_as_read_from_csv_empties_missing_value_literals():
    t = element_table({"e1": "Shopping"})
    df = t.df.copy()
    df.loc[0, "elem_attr_text"], df.loc[1, "elem_attr_text"] = "None", "null value"
    out = as_read_from_csv(t.with_df(df)).df
    assert out.loc[0, "elem_attr_text"] == "" and out.loc[1, "elem_attr_text"] == "null value"
    assert out["elem_x"].tolist() == df["elem_x"].tolist()


def test_cell_bm25_uses_its_columns_and_cuts_documents():
    t = element_table({"e1": "Shopping"})
    df = t.df.copy()
    df.loc[0, "elem_attr_text"] = "x " * 150 + "Pay"  # the matching term lies beyond 200 characters
    t = t.with_df(df)
    spec = {"query": ["instruction"], "doc": ["elem_attr_text"]}
    full = bm25_scores({"bm25": spec}, t, CARD)
    cut = bm25_scores({"bm25": {**spec, "doc_chars": 200}}, t, CARD)
    assert full[0] > 0 and cut[0] == 0


def test_cell_bm25_cuts_queries():
    t = element_table({"e1": "Shopping"})
    df = t.df.copy()
    df["instruction"] = "x " * 250 + "item"  # the only matching query term lies beyond 400 characters
    t = t.with_df(df)
    spec = {"query": ["instruction"], "doc": ["elem_attr_text"]}
    assert bm25_scores({"bm25": spec}, t, CARD)[0] > 0
    assert bm25_scores({"bm25": {**spec, "query_chars": 400}}, t, CARD)[0] == 0


# --------------------------------------------------------------------------- paper_cells.yaml
SPEC = yaml.safe_load((R.HERE / "paper_cells.yaml").read_text())
CELLS = {c["id"]: c for c in SPEC["cells"]}


def test_cells_carry_the_current_table_1_values():
    tau = {c: CELLS[f"tau2-{c}"]["paper"] for c in ("airline", "retail", "telecom", "combined")}
    assert [tau[c]["tabagent"] for c in tau] == [72, 75, 85, 77]
    assert [tau[c]["synth"] for c in tau] == [71, 75, 85, 78]
    assert [tau[c]["synth_recall@9"] for c in tau] == [97, 96, 100, 97]  # Table 1, +synth head
    assert [tau[c]["recall@9"] for c in tau] == [98, 96, 100, 97]  # Table 11, real-only head
    assert [tau[c]["dsr_ft"] for c in tau] == [44, 42, 28, 42]
    skill = CELLS["skill-selection"]
    assert skill["paper"] == {
        "tabagent": 80,
        "recall@9": 91,
        "chance": 1,
        "bm25": 65,
        "dsr_zs": 51,
        "dsr_ft": 74,
        "fewshot": 33,
        "ft_slm": 52,
    }
    assert "bm25" not in skill  # BM25 over the card's own text
    element = CELLS["element-selection"]
    assert element["recall_at"] == 5 and element["paper"]["synth_recall@5"] == {
        "Entertainment": 82,
        "Shopping": 84,
        "Travel": 83,
    }
    assert element["paper"]["dsr_zs"] == {"Entertainment": 11, "Shopping": 13, "Travel": 12}
    assert (
        CELLS["operation-selection"]["paper"]["dsr_zs"] == 28
        and CELLS["operation-selection"]["paper"]["dsr_ft"] == 84
    )
    assert (
        CELLS["model-routing"]["paper"]["tabagent"] == 71 and CELLS["skill-gating"]["paper"]["tabagent"] == 48
    )
    transfer = SPEC["transfer"]["paper"]
    diag = np.mean([transfer[a][a] for a in transfer])
    off = np.mean([transfer[a][b] for a in transfer for b in transfer if a != b])
    assert round(diag, 1) == 76.6 and round(off, 1) == 73.5  # Table 2's own- and cross-agent means


def test_unknown_cell_ids_are_rejected_and_listed(capsys):
    with pytest.raises(SystemExit):
        R.main(["--cells", "tau2_airline"])
    err = capsys.readouterr().err
    assert "tau2_airline" in err and "tau2-airline" in err
    with pytest.raises(SystemExit):
        R.main(["--help"])
    assert "element-selection" in capsys.readouterr().out


# --------------------------------------------------------------------------- cells on toy data
def test_chance_is_the_expected_metric_of_a_random_ranking():
    t = make_table(n_tasks=6)
    k = t.df.groupby(t.decision_ids.to_numpy()).size()  # 8 candidates per decision
    relevant = t.df.groupby(t.decision_ids.to_numpy())["used"].sum()
    assert R.chance(t, "p@r") == pytest.approx(round(100 * (relevant / k).mean(), 1))
    assert R.chance(t, "recall@3") == pytest.approx(round(100 * 3 / 8, 1))


def test_ranking_cell_reports_both_recalls_and_the_nested_synth_head():
    t = make_table(n_tasks=24)
    card = make_card("hashing")
    folds = assign_folds(t.groups, 3)
    cell = {"metric": "p@r", "synth": True}
    out = R.ranking_cell(cell, t, card, folds, (0,), "paper")
    assert set(out) >= {"tabagent", "recall@9", "chance", "bm25", "synth", "synth_recall@9"}
    assert 0 <= out["synth"]["value"] <= 100 and 0 <= out["synth_recall@9"] <= 100


def test_attach_baselines_reads_the_default_recipe_and_domain_slices(tmp_path):
    folder = tmp_path / "baselines"
    folder.mkdir()
    summary = {"p@r": {"value": 0.4}}
    (folder / "tau2__airline__dsr-ft.json").write_text(json.dumps({"summary": summary}))
    other = {"summary": {"p@r": {"value": 0.9}}}
    (folder / "tau2__airline__dsr-ft__hard-negatives.json").write_text(json.dumps(other))
    (folder / "tau2__airline__fewshot__Llama.dryrun.json").write_text(json.dumps(other))
    result = {"card": {"tabagent": 1.0}}
    R.attach_baselines({"dataset": "tau2/airline", "metric": "p@r"}, result, tmp_path)
    assert result["card"]["dsr_ft"] == 40.0 and "fewshot" not in result["card"]
    slices = {"slices": {"Shopping": {"recall@3": 0.18}, "Travel": {"recall@3": 0.23}}}
    (folder / "mind2web__element-150__dsr-zs.json").write_text(json.dumps(slices))
    result = {"card": {}}
    cell = {"dataset": "mind2web/element-150", "metric": "recall@3", "per_domain": True}
    R.attach_baselines(cell, result, tmp_path)
    assert result["card"]["dsr_zs"] == {"Shopping": 18.0, "Travel": 23.0}


def test_markdown_lists_paper_and_released_quantities():
    cells = [{"id": "c", "paper": {"tabagent": 80, "synth": {"A": 70}}}]
    results = {
        "c": {
            "a|b": {
                "tabagent": {"value": 79.6, "ci": [74.0, 85.4]},
                "synth": {"A": 71.0, "overall": 70.0},
                "recall@5": {"A": 73.7},
                "seconds": 3.0,
            }
        }
    }
    md = R.markdown(cells, results)
    assert "| c | a\\|b | tabagent | 80 | 79.6 [74.0, 85.4] | -0.4 |" in md
    assert "| c | a\\|b | synth (A) | 70 | 71.0 | +1.0 |" in md
    assert "| c | a\\|b | recall@5 (A) |  | 73.7 |  |" in md  # released only
    assert "seconds" not in md
