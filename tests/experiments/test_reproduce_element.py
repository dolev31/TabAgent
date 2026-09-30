"""The paper's Mind2Web element protocol in experiments/reproduce.py, on fabricated Mind2Web-shaped rows."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reproduce import (  # noqa: E402
    PaperElementSynth,
    as_read_from_csv,
    bm25_scores,
    evaluator_recall,
    random_tie_recall,
)
from tabagent import TabAgent, TabHead, datasets  # noqa: E402
from tabagent.table import DecisionTable  # noqa: E402

from conftest import make_card, make_table  # noqa: E402

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


def test_evaluator_recall_per_domain():
    t = element_table({"e1": "Shopping", "e3": "Travel"})
    scores = np.array([0.9, 0.5, 0.1, 0.1, 0.9, 0.5])  # e1: target first; e3: target last
    assert evaluator_recall(t, scores, 1) == {"Shopping": 1.0, "Travel": 0.0, "overall": 0.5}
    assert evaluator_recall(t, scores, 3)["overall"] == 1.0


def test_cell_bm25_uses_its_columns_and_cuts_documents():
    t = element_table({"e1": "Shopping"})
    df = t.df.copy()
    df.loc[0, "elem_attr_text"] = "x " * 150 + "Pay"  # the matching term lies beyond 200 characters
    t = t.with_df(df)
    spec = {"query": ["instruction"], "doc": ["elem_attr_text"]}
    full = bm25_scores({"bm25": spec}, t, CARD)
    cut = bm25_scores({"bm25": {**spec, "doc_chars": 200}}, t, CARD)
    assert full[0] > 0 and cut[0] == 0


def test_random_tie_recall_shares_tied_slots():
    t = element_table({"e1": "Shopping", "e3": "Travel"})
    # e1: all three tied (target among them); e3: target strictly first.
    scores = np.array([0.0, 0.0, 0.0, 0.9, 0.1, 0.1])
    out = random_tie_recall(t, scores, 1)
    assert out["Shopping"] == pytest.approx(1 / 3) and out["Travel"] == 1.0
    assert random_tie_recall(t, scores, 2)["Shopping"] == pytest.approx(2 / 3)
