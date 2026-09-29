"""Mind2Web builders on fabricated tasks in the upstream train-file format (no real Mind2Web data)."""

import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tabagent import datasets
from tabagent.datasets import mind2web as m2w
from tabagent.evaluation.cv import CVResult, stable_uid
from tabagent.table import DecisionTable


def _cand(node, tag="button", **attrs):
    attrs = {"backend_node_id": str(node), "bounding_box_rect": "10,20,30,40", **attrs}
    return {
        "attributes": json.dumps(attrs),
        "backend_node_id": str(node),
        "is_original_target": False,
        "is_top_level_target": False,
        "tag": tag,
    }


def _action(op, value, pos, neg):
    return {
        "action_uid": "a",
        "raw_html": "<html>big page</html>",
        "cleaned_html": "<html>page</html>",
        "operation": {"op": op, "original_op": op, "value": value},
        "pos_candidates": pos,
        "neg_candidates": neg,
    }


def _tasks():
    search = _cand(1, "input", type="text", placeholder="Search  cities", is_clickable=True)
    go = _cand(2, "button", **{"class": "btn go", "aria_label": "Go"})
    negs = [_cand(100 + j, "div", **{"class": f"c{j}"}) for j in range(70)]
    trip = {
        "annotation_id": "ann-trip",
        "confirmed_task": "Find a hotel in Oslo",
        "website": "hotels",
        "domain": "Travel",
        "subdomain": "Hotel",
        "action_reprs": ["[textbox] Search -> TYPE: Oslo", "[button] Go -> CLICK", "[div] -> CLICK"],
        "actions": [
            _action("TYPE", "Oslo", [search], negs),
            _action("CLICK", "", [go], negs[:3]),
            _action("CLICK", "", [], negs[:5]),  # no positive candidate: dropped
        ],
    }
    many = [_cand(200 + j, "a") for j in range(30)]
    shop = {
        "annotation_id": "ann-shop",
        "confirmed_task": "Buy socks",
        "website": "store",
        "domain": "Shopping",
        "subdomain": "Clothing",
        "action_reprs": ["[a] Socks -> CLICK"],
        "actions": [_action("CLICK", "", many, negs)],
    }
    return [trip, shop]


@pytest.fixture
def originals(tmp_path, monkeypatch):
    """The eleven train files in the cache layout; the fabricated tasks go into the first one."""
    monkeypatch.delenv("TABAGENT_SOURCE_MIRROR", raising=False)
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    files = m2w.SOURCE.local_dir()
    rels = m2w.SOURCE.files("data/train/")
    assert len(rels) == 11
    for i, rel in enumerate(rels):
        p = files / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_tasks() if i == 0 else []))
    return files


def test_derive_drops_html_and_keeps_everything_else(originals):
    path = m2w.actions_path(m2w.SOURCE)
    assert path == m2w.SOURCE.local_dir() / m2w.ACTIONS
    with gzip.open(path, "rt") as f:
        tasks = [json.loads(line) for line in f]
    assert [t["annotation_id"] for t in tasks] == ["ann-trip", "ann-shop"]
    assert "raw_html" not in tasks[0]["actions"][0] and "cleaned_html" not in tasks[0]["actions"][0]
    expected = _tasks()[0]
    for a in expected["actions"]:
        del a["raw_html"], a["cleaned_html"]
    assert tasks[0] == expected
    assert m2w.actions_path(m2w.SOURCE) == path  # reused, not derived again


def test_operation_rows(originals):
    df = m2w.build_operation(m2w.SOURCE)
    assert len(df) == 3  # the step without a positive candidate is dropped
    trip = df[df["annotation_id"] == "ann-trip"]
    assert trip["op"].tolist() == ["TYPE", "CLICK"]
    assert trip["step_index"].tolist() == [0, 1]
    assert trip["history_action_reprs"].tolist() == ["", "[textbox] Search -> TYPE: Oslo"]
    assert trip["target_tag"].tolist() == ["input", "button"]
    assert trip["target_type"].tolist() == ["text", ""]
    assert (
        df["task_uid"] == df["annotation_id"].map(lambda a: stable_uid("task", "mind2web_train", a, ""))
    ).all()
    DecisionTable(df, datasets.card("mind2web/operation").spec)


def test_element_rows_and_slates(originals):
    df = m2w.build_element(m2w.SOURCE)
    sizes = df.groupby(["annotation_id", "step_index"]).size().to_dict()
    assert sizes == {("ann-trip", 0): 50, ("ann-trip", 1): 4, ("ann-shop", 0): 50}
    pos = df.groupby(["annotation_id", "step_index"])["is_target"].sum().to_dict()
    assert pos == {("ann-trip", 0): 1, ("ann-trip", 1): 1, ("ann-shop", 0): 25}  # positives capped at 25
    first = df[(df["annotation_id"] == "ann-trip") & (df["is_target"] == 1)].iloc[0]
    assert first["elem_attr_text"] == "Search cities text"  # attribute order, whitespace collapsed
    assert (first["elem_is_clickable"], first["elem_x"], first["elem_w"], first["elem_h"]) == (1, 10, 30, 40)
    assert first["eval_operation"] == "TYPE" and first["eval_op_value"] == "Oslo"
    go = df[(df["backend_node_id"] == "2")].iloc[0]
    assert (
        go["elem_attr_text"] == "btn go Go" and go["history_action_reprs"] == "[textbox] Search -> TYPE: Oslo"
    )


def test_negative_sampling_is_seeded_per_step():
    pos, neg = [_cand(1)], [_cand(100 + j) for j in range(70)]
    keep_pos, keep_neg = m2w.slate("ann", 3, pos, neg)
    assert len(keep_pos) + len(keep_neg) == m2w.SLATE
    assert m2w.slate("ann", 3, pos, neg)[1] == keep_neg
    assert m2w.slate("ann", 4, pos, neg)[1] != keep_neg
    h = int(hashlib.md5(b"ann:3").hexdigest()[:8], 16)  # the paper runs' per-step seed
    idx = np.random.default_rng(h ^ 20260720).permutation(70)[:49]
    assert keep_neg == [neg[j] for j in idx]
    many_pos, some_neg = m2w.slate("ann", 3, [_cand(j) for j in range(30)], neg)
    assert many_pos == [_cand(j) for j in range(25)] and len(some_neg) == 25


def test_bundled_cards_compute_the_derived_features(originals):
    df = m2w.build_element(m2w.SOURCE)
    card = datasets.card("mind2web/element")
    table = DecisionTable(df, card.spec)
    X = card.fit(table).transform(table.drop_labels())
    assert (X["n_prior_actions"] == df["step_index"]).all()
    assert np.allclose(X["elem_area"], df["elem_w"] * df["elem_h"])
    assert not set(X.columns) & set(card.spec.outcome_cols)


def test_element_subset_keeps_only_listed_tasks(originals, monkeypatch):
    monkeypatch.setattr(m2w, "_fold_groups", lambda name: {m2w.task_uid("ann-shop")})
    df = m2w.build_element(m2w.SOURCE, tasks="any")
    assert set(df["annotation_id"]) == {"ann-shop"}


def test_mirror_provides_the_reduced_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("TABAGENT_SOURCE_MIRROR", str(tmp_path / "mirror"))
    reduced = tmp_path / "mirror" / m2w.SOURCE.id / m2w.SOURCE.release_id / "files" / m2w.ACTIONS
    reduced.parent.mkdir(parents=True)
    with gzip.open(reduced, "wt") as f:
        for t in _tasks():
            f.write(json.dumps({**t, "_source_shard": 0}) + "\n")
    assert m2w.actions_path(m2w.SOURCE) == reduced
    assert len(m2w.build_operation(m2w.SOURCE)) == 3
    with pytest.raises(FileNotFoundError, match="download=True"):
        monkeypatch.delenv("TABAGENT_SOURCE_MIRROR")
        m2w.actions_path(m2w.SOURCE)


def test_registry_manifest_and_folds():
    names = {"mind2web/operation", "mind2web/element", "mind2web/element-150"}
    assert names <= set(datasets.available().index)
    man = m2w.SOURCE.manifest
    assert len(man) == 11 and all(m2w.REVISION in e["url"] and len(e["sha256"]) == 64 for e in man.values())
    full, sub = datasets.folds("mind2web/element"), datasets.folds("mind2web/element-150")
    assert len(full) == 1009 and set(full.values()) == set(range(5))
    assert datasets.folds("mind2web/operation") == full
    assert len(sub) == 150 and all(full[g] == f for g, f in sub.items())


def test_per_domain_slices_one_result():
    rows = []
    for g, dom in [("t1", "Travel"), ("t2", "Shopping")]:
        for step in (0, 1):
            rows += [{"task_uid": g, "step_index": step, "backend_node_id": c, "domain": dom} for c in "ab"]
    df = pd.DataFrame(rows)
    df["is_target"] = (df["backend_node_id"] == "a").astype(int)
    spec = datasets.card("mind2web/element").spec
    table = DecisionTable(df.assign(eval_operation="", eval_op_value="", eval_original_op=""), spec)
    good = np.where(df["backend_node_id"] == "a", 1.0, 0.0)
    scores = np.where(df["domain"] == "Travel", good, 1 - good)
    res = CVResult(table, pd.DataFrame({0: scores}), folds={"t1": 0, "t2": 1})
    out = m2w.per_domain(res, "recall@1")
    assert out.to_dict() == {"Shopping": 0.0, "Travel": 1.0, "overall": 0.5}
