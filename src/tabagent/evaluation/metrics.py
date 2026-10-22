"""Decision-quality metrics as the paper defines them (Sec. 4.3; App. L).

Shortlisting (Sec. 4.3): for a decision t with relevant set G(t) of size R_t, let S_k(t) be the top-k prefix
of the system's ranking over the candidate catalog.

    Recall@k = |S_k(t) ∩ G(t)| / |G(t)|
    P@R      = |S_{R_t}(t) ∩ G(t)| / R_t      (R-precision; equals Recall@R_t)

Single-class decisions use accuracy, and multi-label decisions example-based precision and recall with
``max(1, .)`` denominators (App. L). Policy decisions use route@1, the realized reward of the chosen arm
(App. B.7). ``normalized`` is the headroom score Norm = (m - chance) / (oracle - chance) (Sec. 5; App. B.7).

Every per-decision function returns a ``pd.Series`` indexed by decision id, so that seeds can be averaged
per decision and tasks can be resampled for confidence intervals (see ``stats``).

**Tied scores.** Metrics that depend on the order of candidates take ``tie_break``:

* ``"expected"`` (default; App. L, "Tied scores"): tied candidates are ordered uniformly at random and the
  exact expected metric is reported, so no method gains from the order in which candidates are listed;
* ``"candidate"``: ties are broken by candidate id, which gives one deterministic ranking. TabHead's decoders
  (``tabagent.tabhead.decode``) use this order for live decisions, since a deployed system needs one answer.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from tabagent.table import DecisionTable, LeakageError

TIE_BREAKS = ("expected", "candidate")


def _check_tie_break(tie_break: str) -> None:
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}, got {tie_break!r}")


def _frame(table: DecisionTable, scores: np.ndarray) -> pd.DataFrame:
    if not table.has_labels:
        raise LeakageError("metrics need a labeled table")
    scores = np.asarray(scores)
    if len(scores) != len(table):
        raise ValueError(f"{len(scores)} scores for {len(table)} rows")
    s = table.spec
    out = pd.DataFrame(
        {
            "decision": table.decision_ids.to_numpy(),
            "group": table.df[s.group].to_numpy(),
            "y": table.df[s.label].to_numpy(),
            "score": scores,
        }
    )
    if s.candidate:
        out["candidate"] = table.df[s.candidate].astype(str).to_numpy()
    if s.reward:
        out["reward"] = table.df[s.reward].to_numpy(dtype=float)
    return out


def ranked(table: DecisionTable, scores: np.ndarray) -> pd.DataFrame:
    """Rows sorted within each decision by descending score, ties by candidate id, with a 0-based ``rank``.

    This is the one deterministic ranking (``tie_break="candidate"``); the paper's metrics average over tie
    orders instead (``tie_break="expected"``), which has no single ranking.
    """
    f = _frame(table, scores)
    order = np.lexsort((f["candidate"].to_numpy(), -f["score"].to_numpy(float), f["decision"].to_numpy()))
    f = f.iloc[order].reset_index(drop=True)
    f["rank"] = f.groupby("decision", sort=False).cumcount()
    return f


def _per_decision(r: pd.DataFrame, fn) -> pd.Series:
    vals = {d: fn(g) for d, g in r.groupby("decision", sort=True)}
    return pd.Series({d: v for d, v in vals.items() if v is not None}, dtype=float, name="value")


def _rows(table: DecisionTable, scores: np.ndarray, tie_break: str) -> pd.DataFrame:
    """Ranked rows, or the unranked rows when the metric takes the expectation over tie orders."""
    _check_tie_break(tie_break)
    return _frame(table, scores) if tie_break == "expected" else ranked(table, scores)


def _cutoff(s: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Candidates scored strictly above the k-th best score, and those tied with it (who share the rest)."""
    cut = np.sort(s)[::-1][k - 1]
    return s > cut, s == cut


def _top_k_hits(g: pd.DataFrame, k: int, tie_break: str) -> float:
    """Relevant candidates in the top k; under ``"expected"``, their mean over random orders of ties."""
    y = g["y"].to_numpy(dtype=float)
    if tie_break != "expected":
        return float(y[:k].sum())
    s = g["score"].to_numpy(dtype=float)
    if k >= len(s):
        return float(y.sum())
    above, tie = _cutoff(s, k)
    return float(y[above].sum() + (k - above.sum()) * y[tie].sum() / tie.sum())


def recall_at_k(table: DecisionTable, scores: np.ndarray, k: int, tie_break: str = "expected") -> pd.Series:
    """Per-decision Recall@k: the share of the decision's relevant candidates in the top ``k`` of the ranking.
    Decisions with an empty relevant set are skipped. ``tie_break="expected"`` (the default) orders tied
    candidates uniformly at random and returns the exact expected value (App. L)."""
    r = _rows(table, scores, tie_break)

    def fn(g: pd.DataFrame):
        tot = g["y"].to_numpy(dtype=float).sum()
        return None if tot == 0 else _top_k_hits(g, k, tie_break) / tot

    return _per_decision(r, fn)


def p_at_r(table: DecisionTable, scores: np.ndarray, tie_break: str = "expected") -> pd.Series:
    """Per-decision R-precision (P@R): precision of the top-R_t candidates, R_t = |G(t)|, the size of the
    decision's relevant set. It equals Recall@R_t. ``tie_break="expected"`` (the default) orders tied
    candidates uniformly at random and returns the exact expected value (App. L)."""
    r = _rows(table, scores, tie_break)

    def fn(g: pd.DataFrame):
        big_r = int(g["y"].to_numpy(dtype=float).sum())
        return None if big_r == 0 else _top_k_hits(g, big_r, tie_break) / big_r

    return _per_decision(r, fn)


def hit_at_k(table: DecisionTable, scores: np.ndarray, k: int, tie_break: str = "expected") -> pd.Series:
    """Per-decision indicator that at least one relevant candidate is in the top k (its probability under
    ``tie_break="expected"``)."""
    r = _rows(table, scores, tie_break)

    def fn(g: pd.DataFrame):
        y = g["y"].to_numpy(dtype=float)
        if y.sum() == 0:
            return None
        if tie_break != "expected":
            return float(y[:k].sum() > 0)
        s = g["score"].to_numpy(dtype=float)
        if k >= len(s):
            return 1.0
        above, tie = _cutoff(s, k)
        if y[above].sum() > 0:
            return 1.0
        slots, m, q = int(k - above.sum()), int(tie.sum()), int(y[tie].sum())
        return 1.0 - math.comb(m - q, slots) / math.comb(m, slots)  # 1 - P(no relevant tie is drawn)

    return _per_decision(r, fn)


def route_at_1(table: DecisionTable, scores: np.ndarray, tie_break: str = "expected") -> pd.Series:
    """Per-decision realized reward of the top-scored arm (App. B.7); under ``tie_break="expected"``, the mean
    reward of the arms tied for the top score."""
    if not table.spec.reward:
        raise ValueError("route@1 needs a reward column")
    _check_tie_break(tie_break)
    if tie_break == "expected":
        f = _frame(table, scores)
        top = f[f["score"] == f.groupby("decision")["score"].transform("max")]
        return top.groupby("decision", sort=True)["reward"].mean().astype(float).rename("value")
    r = ranked(table, scores)
    return r[r["rank"] == 0].set_index("decision")["reward"].sort_index().astype(float).rename("value")


def chance_p_at_r(table: DecisionTable) -> pd.Series:
    """Expected P@R of a uniformly random ranking: R_t / n_t per decision."""
    f = _frame(table, np.zeros(len(table)))
    agg = f.groupby("decision", sort=True)["y"].agg(["sum", "size"])
    agg = agg[agg["sum"] > 0]
    return (agg["sum"] / agg["size"]).astype(float).rename("value")


def chance_route(table: DecisionTable) -> pd.Series:
    """Expected reward of a uniformly random policy per decision (the policy-family chance, App. B.7)."""
    f = _frame(table, np.zeros(len(table)))
    return f.groupby("decision", sort=True)["reward"].mean().rename("value")


def oracle_route(table: DecisionTable) -> pd.Series:
    """Reward of the best arm per decision."""
    f = _frame(table, np.zeros(len(table)))
    return f.groupby("decision", sort=True)["reward"].max().rename("value")


def accuracy(table: DecisionTable, predictions: np.ndarray) -> pd.Series:
    """Per-decision correctness of a single-class prediction (one row per decision; App. L)."""
    f = _frame(table, np.asarray(predictions, dtype=object))
    f["hit"] = (f["score"].astype(str) == f["y"].astype(str)).astype(float)
    return f.groupby("decision", sort=True)["hit"].mean().rename("value")


def macro_f1(table: DecisionTable, predictions: np.ndarray) -> float:
    """Label-macro F1 of single-class predictions."""
    from sklearn.metrics import f1_score

    f = _frame(table, np.asarray(predictions, dtype=object))
    return float(f1_score(f["y"].astype(str), f["score"].astype(str), average="macro"))


def multilabel_predict(table: DecisionTable, scores: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Binary-relevance decoding (Sec. 3.3; App. J): the candidates with p >= threshold, backing off to the
    single highest-scoring candidate (ties by candidate id) when the predicted set is empty. Returns the
    ranked rows with a 0/1 ``pred`` column."""
    r = ranked(table, scores)
    r["pred"] = (r["score"] >= threshold).astype(int)
    empty = r.groupby("decision")["pred"].transform("sum") == 0
    r.loc[empty & (r["rank"] == 0), "pred"] = 1
    return r


def multilabel_prf(
    table: DecisionTable, scores: np.ndarray, threshold: float = 0.5, tie_break: str = "expected"
) -> pd.DataFrame:
    """Per-decision example-based precision, recall and F1 with ``max(1, .)`` denominators (App. L).

    The top-1 back-off of an empty predicted set is the only place ties matter: under ``"expected"`` it picks
    one of the top-scored candidates uniformly at random, and the scores are their exact expectation.
    """
    _check_tie_break(tie_break)
    f = _frame(table, scores)
    by = f["decision"]
    pred = (f["score"] >= threshold).astype(float)
    empty = pred.groupby(by).transform("sum") == 0
    if tie_break == "expected":
        top = (f["score"] == f.groupby(by)["score"].transform("max")).astype(float)
        backoff = top / top.groupby(by).transform("sum")
    else:
        r = multilabel_predict(table, scores, threshold)
        chosen = set(zip(r.loc[r["pred"] == 1, "decision"], r.loc[r["pred"] == 1, "candidate"], strict=True))
        backoff = pd.Series([float(k in chosen) for k in zip(by, f["candidate"], strict=True)], index=f.index)
    pred = pred.where(~empty, backoff)
    tp = pred * f["y"].astype(float)
    g = pd.DataFrame({"tp": tp, "npred": pred, "npos": f["y"].astype(float)}).groupby(by, sort=True).sum()
    prec = g["tp"] / np.maximum(1, g["npred"])
    rec = g["tp"] / np.maximum(1, g["npos"])
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec).replace(0, 1), 0.0)
    return pd.DataFrame({"precision": prec, "recall": rec, "f1": f1}, index=g.index)


def normalized(value: float, chance: float, oracle: float) -> float:
    """Headroom score Norm = (m - chance) / (oracle - chance) (Sec. 5; App. B.7)."""
    if oracle == chance:
        return float("nan")
    return (value - chance) / (oracle - chance)


_K = re.compile(r"^(recall|hit)@(\d+)$")


def per_decision(table: DecisionTable, scores: np.ndarray, metric: str, **kw) -> pd.Series:
    """Dispatch a per-decision metric by name: ``p@r``, ``recall@K``, ``hit@K``, ``route@1``, ``accuracy``,
    ``ml_precision``, ``ml_recall``, ``ml_f1``. Keyword arguments (``tie_break``, ``threshold``) go to the
    metric; ``accuracy`` takes none, since a single-class prediction has no ties."""
    m = metric.lower()
    if m == "p@r":
        return p_at_r(table, scores, **kw)
    if match := _K.match(m):
        fn = recall_at_k if match.group(1) == "recall" else hit_at_k
        return fn(table, scores, int(match.group(2)), **kw)
    if m == "route@1":
        return route_at_1(table, scores, **kw)
    if m == "accuracy":
        return accuracy(table, scores)
    if m in ("ml_precision", "ml_recall", "ml_f1"):
        return multilabel_prf(table, scores, **kw)[m[3:]].rename("value")
    raise ValueError(f"unknown metric {metric!r}")
