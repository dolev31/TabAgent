"""Mind2Web: operation selection and GUI element selection (paper Sec. 5, Table 1; App. B.6).

Source: the public train split of Mind2Web (Deng et al., 2023; 1,009 tasks, CC-BY-4.0) on the Hugging Face
Hub at a pinned revision. The password-protected test splits are never used. The eleven original
``data/train/train_*.json`` files (5.9 GB, mostly page HTML) are downloaded once, verified against the
manifest, and reduced to one JSON line per task without the HTML (``ACTIONS``, about 100 MB); both tables
are built from that file. Parsing one original file needs about 2 GB of memory. A source mirror may
provide the reduced file directly; it is used as is.

Labels are the human demonstrations.

``mind2web/operation``
    One row per action step; the class ``op`` is CLICK, TYPE or SELECT, and the step's target element
    (tag and type) is part of the context. Macro-F1 (Table 1: 92).
``mind2web/element``
    One row per (action step, candidate element); ``is_target`` marks the demonstrated element. Each step
    keeps its positive candidates (at most 25) and a seeded sample of negatives, 50 candidates in all.
    Steps without a positive candidate are dropped (413 of 7,775). A decision is one step,
    ``(task_uid, step_index)``, and Recall@k is computed per step. Rows follow the paper evaluator's fixed
    task split (:func:`evaluator_split`: 706 train, 151 val, 152 test tasks), train tasks first.
``mind2web/element-150``
    A seeded sample of 150 whole tasks (1,019 steps) of ``mind2web/element``, on which the paper's element
    numbers were computed.

The paper reports element selection per web domain (Entertainment, Shopping, Travel) as slices of one
model trained on all domains, not as separate models: see :func:`per_domain`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from tabagent.datasets._core import DatasetInfo, Source
from tabagent.defaults import DEFAULT_SEED
from tabagent.evaluation.cv import stable_uid

REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"
SOURCE = Source(
    id="mind2web_train",
    benchmark="Mind2Web",
    url="https://huggingface.co/datasets/osunlp/Mind2Web",
    revision=REVISION,
    release_id=f"hf-{REVISION}",
    license="CC-BY-4.0",
    redistribution="redistributable",
    citation="deng2023mind2web",
    notes="Public train split only (1,009 tasks); the password-protected test splits are not used.",
)

ACTIONS = "mind2web_train_action_references.jsonl.gz"  # the train split without page HTML
HTML_FIELDS = frozenset({"raw_html", "cleaned_html", "html", "screenshot"})

SEED = DEFAULT_SEED  # seeds the negative sample of every step
SLATE, POS_CAP = 50, 25
ATTR_TEXT = (
    "class",
    "id",
    "name",
    "role",
    "aria_label",
    "alt",
    "title",
    "placeholder",
    "value",
    "label",
    "aria_description",
    "type",
)


# --------------------------------------------------------------------------- source files
def _strip_html(value):
    if isinstance(value, dict):
        return {k: _strip_html(v) for k, v in value.items() if k not in HTML_FIELDS}
    if isinstance(value, list):
        return [_strip_html(v) for v in value]
    return value


def derive_actions(originals: Iterable[Path], out: Path) -> Path:
    """Reduce the original train files (JSON arrays of tasks) to gzipped JSON lines without page HTML."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for path in originals:
            with open(path, encoding="utf-8") as src:
                tasks = json.load(src)
            for task in tasks:
                f.write(json.dumps(_strip_html(task), sort_keys=True, ensure_ascii=False) + "\n")
            del tasks
    tmp.replace(out)
    return out


def actions_path(source: Source = SOURCE, download: bool = False) -> Path:
    """The reduced action file: from the mirror or cache, else derived from the verified originals."""
    try:
        return source.path(ACTIONS)
    except FileNotFoundError:
        pass
    # train_0, train_1, train_10, train_2, ...: the task order of the paper runs
    originals = [source.path(rel, download) for rel in source.files("data/train/")]
    if not originals:
        raise FileNotFoundError(f"{source.id}: the manifest lists no train files")
    return derive_actions(originals, source.local_dir() / ACTIONS)


def iter_tasks(path: Path) -> Iterator[dict]:
    """The tasks of a reduced action file, one dict per line."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------- parsing
def task_uid(annotation_id: str) -> str:
    """The task id used by the paper's fold files."""
    return stable_uid("task", SOURCE.id, annotation_id, "")


def _rng(annotation_id: str, step: int) -> np.random.Generator:
    h = int(hashlib.md5(f"{annotation_id}:{step}".encode(), usedforsecurity=False).hexdigest()[:8], 16)
    return np.random.default_rng(h ^ SEED)


def _attrs(cand: dict) -> dict:
    try:
        a = json.loads(cand.get("attributes", "{}"))
    except (TypeError, ValueError):
        a = {}
    return a if isinstance(a, dict) else {}


def _box(a: dict) -> tuple[float, float, float, float]:
    try:
        x, y, w, h = (float(v) for v in a.get("bounding_box_rect", "0,0,0,0").split(",")[:4])
    except (AttributeError, TypeError, ValueError):
        x = y = w = h = 0.0
    return x, y, w, h


def element_fields(cand: dict) -> dict:
    """What a candidate element exposes: tag, attribute text, type, clickability and bounding box.

    The element's inner text is not part of the public release; the attribute text joins the
    identifying attributes (``ATTR_TEXT``), whitespace-collapsed and cut to 500 characters.
    """
    a = _attrs(cand)
    x, y, w, h = _box(a)
    text = " ".join(str(a[k]) for k in ATTR_TEXT if a.get(k))
    return {
        "backend_node_id": cand.get("backend_node_id"),
        "elem_tag": cand.get("tag"),
        "elem_attr_text": " ".join(text.split())[:500],
        "elem_type": a.get("type", ""),
        "elem_is_clickable": int(bool(a.get("is_clickable"))),
        "elem_x": x,
        "elem_y": y,
        "elem_w": w,
        "elem_h": h,
    }


def slate(annotation_id: str, step: int, pos: list, neg: list) -> tuple[list, list]:
    """One step's candidates: up to ``POS_CAP`` positives plus seeded negatives, ``SLATE`` in all."""
    keep_pos = pos[:POS_CAP]
    need = SLATE - len(keep_pos)
    if need <= 0 or not neg:
        return keep_pos, []
    idx = _rng(annotation_id, step).permutation(len(neg))[:need]
    return keep_pos, [neg[j] for j in idx]


def steps(tasks: Iterable[dict]) -> Iterator[tuple[dict, list, list, dict]]:
    """Usable action steps: ``(context, positives, negatives, operation)``, in source order.

    The context holds only what precedes the action: the task, the web domain and the representations of
    the earlier actions (the current action's representation names its target element).
    """
    for t in tasks:
        aid = t["annotation_id"]
        reprs = t.get("action_reprs") or []
        for i, act in enumerate(t.get("actions") or []):
            pos = act.get("pos_candidates") or []
            if not pos:
                continue
            ctx = {
                "task_uid": task_uid(aid),
                "annotation_id": aid,
                "instruction": t.get("confirmed_task", ""),
                "website": t.get("website", ""),
                "domain": t.get("domain", ""),
                "subdomain": t.get("subdomain", ""),
                "step_index": i,
                "history_action_reprs": " | ".join(reprs[:i]),
            }
            keep_pos, keep_neg = slate(aid, i, pos, act.get("neg_candidates") or [])
            yield ctx, keep_pos, keep_neg, act.get("operation") or {}


def element_rows(tasks: Iterable[dict]) -> pd.DataFrame:
    """One row per (usable step, candidate element), with the step's operation as evaluation columns."""
    rows = []
    for ctx, pos, neg, op in steps(tasks):
        ev = {
            "eval_operation": op.get("op"),
            "eval_op_value": op.get("value"),
            "eval_original_op": op.get("original_op"),
        }
        rows += [{**ctx, **element_fields(c), "is_target": 1, **ev} for c in pos]
        rows += [{**ctx, **element_fields(c), "is_target": 0, **ev} for c in neg]
    return pd.DataFrame(rows)


def operation_rows(tasks: Iterable[dict]) -> pd.DataFrame:
    """One row per usable step: its context, the target element's tag and type, and the operation class."""
    rows = []
    for ctx, pos, _, op in steps(tasks):
        target = pos[0]
        rows.append(
            {
                **ctx,
                "target_tag": target.get("tag"),
                "target_type": _attrs(target).get("type", ""),
                "op": op.get("op"),
                "eval_original_op": op.get("original_op"),
                "eval_op_value": op.get("value"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- builders
def _fold_groups(name: str) -> set[str]:
    res = resources.files("tabagent.datasets") / "folds" / f"{name}.json"
    return set(json.loads(res.read_text())["by_group"])


def build_operation(source: Source, download: bool = False) -> pd.DataFrame:
    """Build ``mind2web/operation`` from the pinned release."""
    return operation_rows(iter_tasks(actions_path(source, download)))


def evaluator_split() -> dict[str, str]:
    """The paper evaluator's fixed split of the 1,009 tasks: 706 train, 151 val, 152 test (seed 20260720).

    The evaluator read the element table split by split, and the paper's element +synth column paraphrases
    the train tasks (``experiments/reproduce.py``, ``element_synth: paper``).
    """
    res = resources.files("tabagent.datasets") / "folds" / "mind2web_split.json"
    return json.loads(res.read_text())["by_group"]


_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def in_evaluator_order(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of train tasks first, then val, then test (tasks outside the split last), otherwise unchanged."""
    split = df["task_uid"].map(evaluator_split()).map(_SPLIT_ORDER).fillna(len(_SPLIT_ORDER))
    return df.iloc[np.argsort(split.to_numpy(), kind="stable")].reset_index(drop=True)


def build_element(source: Source, download: bool = False, tasks: str | None = None) -> pd.DataFrame:
    """``tasks`` names a bundled fold file; only its tasks are kept (e.g. the paper's 150-task sample).

    Rows are listed train tasks first, then val, then test, as the paper's evaluator read them. CatBoost
    depends on row order, and the paper's element numbers were computed in this order.
    """
    items = iter_tasks(actions_path(source, download))
    if tasks:
        keep = _fold_groups(tasks)
        items = (t for t in items if task_uid(t["annotation_id"]) in keep)
    df = element_rows(items)
    return df if df.empty else in_evaluator_order(df)


def box_area(df: pd.DataFrame, w: str = "elem_w", h: str = "elem_h") -> pd.Series:
    """Bounding-box area (a stateless card feature)."""
    return (pd.to_numeric(df[w], errors="coerce") * pd.to_numeric(df[h], errors="coerce")).astype(float)


def per_domain(result, metric: str = "recall@3", **kw) -> pd.Series:
    """Mean per-step ``metric`` of a ``CVResult`` by web domain, plus ``overall`` (the paper's element rows).

    Each step is scored once, by the model whose held-out fold contains it, so the domain rows are slices
    of one model trained on all domains.
    """
    per_step = result.per_decision(metric, **kw)
    table = result.table
    domain = pd.Series(table.df["domain"].to_numpy(), index=table.decision_ids.to_numpy())
    domain = domain[~domain.index.duplicated()].reindex(per_step.index)
    out = per_step.groupby(domain.to_numpy()).mean()
    out.loc["overall"] = per_step.mean()
    return out.rename(metric)


_ELEMENT_PAPER = "Table 1 (GUI element selection: R@3 per web domain, R@5 in the R@k column); App. B.6"
DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        name="mind2web/operation",
        source=SOURCE,
        build=build_operation,
        card="mind2web_operation",
        folds="mind2web_groupkfold5",
        decision="operation selection (single-class: CLICK, TYPE or SELECT)",
        paper="Table 1 (operation selection, macro-F1); App. B.6",
        description="Operation type (CLICK / TYPE / SELECT) of each demonstrated action step.",
    ),
    DatasetInfo(
        name="mind2web/element",
        source=SOURCE,
        build=build_element,
        card="mind2web_element",
        folds="mind2web_groupkfold5",
        decision="GUI element selection (ranking over 50 candidate elements per step)",
        paper=_ELEMENT_PAPER,
        description="Target element among 50 candidate elements of each action step (all 1,009 tasks).",
        version=2,  # rows in the paper evaluator's split order
    ),
    DatasetInfo(
        name="mind2web/element-150",
        source=SOURCE,
        build=build_element,
        card="mind2web_element",
        folds="mind2web_element_150",
        decision="GUI element selection (ranking over 50 candidate elements per step)",
        paper=_ELEMENT_PAPER,
        description="The seeded 150-task sample of mind2web/element behind the paper's element numbers.",
        args={"tasks": "mind2web_element_150"},
        version=2,
    ),
]
