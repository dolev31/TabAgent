"""Decision-quality metrics, implemented exactly as defined in the paper.

For shortlisting (paper Sec. 4.3): for a decision t with relevant set G(t) of size R_t, let S_k(t) be the
top-k prefix of the system's ranking over the candidate catalog, with ties broken by candidate id.

    Recall@k = |S_k(t) ∩ G(t)| / |G(t)|                                     (L451-454)
    P@R      = |S_{R_t}(t) ∩ G(t)| / R_t      (R-precision; equals Recall@R_t)  (L457-462)

Single-class decisions use accuracy (L2265-2269); multi-label decisions use example-based precision and
recall with ``max(1, .)`` denominators (L2272-2287); policy decisions use route@1, the realized reward of the
chosen arm (L1181). ``normalized`` is the headroom score Norm = (m - chance) / (oracle - chance).

Every per-decision function returns a ``pd.Series`` indexed by decision id, so that seeds can be averaged
per decision and tasks can be resampled for confidence intervals (see ``stats``).

Ranking metrics take ``tie_break``: ``"candidate"`` (default, the rule above), ``"expected"`` (the exact
expectation over a uniformly random order of tied scores; the paper reports BM25 this way, since BM25 gives
most candidates the same score of 0) and ``"legacy"`` (the pre-release code's sort, for the equivalence
harness).
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from tabagent.table import DecisionTable, LeakageError

TIE_BREAKS = ("candidate", "expected", "legacy")


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


def ranked(table: DecisionTable, scores: np.ndarray, tie_break: str = "candidate") -> pd.DataFrame:
    """Rows sorted within each decision by descending score, with a ``rank`` column (0-based).

    ``tie_break="candidate"`` is the paper's rule (ties broken by sort on candidate id, L448).
    ``tie_break="legacy"`` reproduces the pre-release code, which sorted each decision with
    ``DataFrame.sort_values(ascending=False)`` and left ties in an implementation-defined order; it exists
    only for the equivalence harness. ``tie_break="expected"`` averages over orders and has no single ranking.
    """
    if tie_break == "expected":
        raise ValueError('tie_break="expected" averages over tie orders; it has no single ranking')
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}")
    f = _frame(table, scores)
    if tie_break == "candidate":
        order = np.lexsort((f["candidate"].to_numpy(), -f["score"].to_numpy(float), f["decision"].to_numpy()))
        f = f.iloc[order]
    else:
        f = pd.concat([g.sort_values("score", ascending=False) for _, g in f.groupby("decision", sort=True)])
    f = f.reset_index(drop=True)
    f["rank"] = f.groupby("decision", sort=False).cumcount()
    return f


def _per_decision(r: pd.DataFrame, fn) -> pd.Series:
    vals = {d: fn(g) for d, g in r.groupby("decision", sort=True)}
    return pd.Series({d: v for d, v in vals.items() if v is not None}, dtype=float, name="value")


def _rows(table: DecisionTable, scores: np.ndarray, tie_break: str) -> pd.DataFrame:
    """Ranked rows, or the unranked rows when the metric takes the expectation over tie orders."""
    return _frame(table, scores) if tie_break == "expected" else ranked(table, scores, tie_break)


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


def recall_at_k(table: DecisionTable, scores: np.ndarray, k: int, tie_break: str = "candidate") -> pd.Series:
    """Per-decision Recall@k. Decisions with an empty relevant set are skipped."""
    r = _rows(table, scores, tie_break)

    def fn(g: pd.DataFrame):
        tot = g["y"].to_numpy(dtype=float).sum()
        return None if tot == 0 else _top_k_hits(g, k, tie_break) / tot

    return _per_decision(r, fn)


def p_at_r(table: DecisionTable, scores: np.ndarray, tie_break: str = "candidate") -> pd.Series:
    """Per-decision R-precision (P@R): precision of the top-R_t candidates, R_t = |G(t)|."""
    r = _rows(table, scores, tie_break)

    def fn(g: pd.DataFrame):
        big_r = int(g["y"].to_numpy(dtype=float).sum())
        return None if big_r == 0 else _top_k_hits(g, big_r, tie_break) / big_r

    return _per_decision(r, fn)


def hit_at_k(table: DecisionTable, scores: np.ndarray, k: int, tie_break: str = "candidate") -> pd.Series:
    """Per-decision indicator that at least one relevant candidate is in the top k."""
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


def route_at_1(table: DecisionTable, scores: np.ndarray, tie_break: str = "candidate") -> pd.Series:
    """Per-decision realized reward of the top-scored arm (policy decisions, L1181)."""
    if not table.spec.reward:
        raise ValueError("route@1 needs a reward column")
    if tie_break == "expected":  # the mean reward of the arms tied for the top score
        f = _frame(table, scores)
        top = f[f["score"] == f.groupby("decision")["score"].transform("max")]
        return top.groupby("decision", sort=True)["reward"].mean().astype(float).rename("value")
    r = ranked(table, scores, tie_break)
    return r[r["rank"] == 0].set_index("decision")["reward"].sort_index().astype(float).rename("value")


def chance_p_at_r(table: DecisionTable) -> pd.Series:
    """Expected P@R of a uniformly random ranking: R_t / n_t per decision."""
    f = _frame(table, np.zeros(len(table)))
    agg = f.groupby("decision", sort=True)["y"].agg(["sum", "size"])
    agg = agg[agg["sum"] > 0]
    return (agg["sum"] / agg["size"]).astype(float).rename("value")


def chance_route(table: DecisionTable) -> pd.Series:
    """Expected reward of a uniformly random policy per decision (the policy-family chance, L1181)."""
    f = _frame(table, np.zeros(len(table)))
    return f.groupby("decision", sort=True)["reward"].mean().rename("value")


def oracle_route(table: DecisionTable) -> pd.Series:
    """Reward of the best arm per decision."""
    f = _frame(table, np.zeros(len(table)))
    return f.groupby("decision", sort=True)["reward"].max().rename("value")


def accuracy(table: DecisionTable, predictions: np.ndarray) -> pd.Series:
    """Per-decision correctness of a single-class prediction (one row per decision, L2267)."""
    f = _frame(table, np.asarray(predictions, dtype=object))
    f["hit"] = (f["score"].astype(str) == f["y"].astype(str)).astype(float)
    return f.groupby("decision", sort=True)["hit"].mean().rename("value")


def macro_f1(table: DecisionTable, predictions: np.ndarray) -> float:
    """Label-macro F1 of single-class predictions."""
    from sklearn.metrics import f1_score

    f = _frame(table, np.asarray(predictions, dtype=object))
    return float(f1_score(f["y"].astype(str), f["score"].astype(str), average="macro"))


def multilabel_predict(table: DecisionTable, scores: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Binary-relevance decoding (L323, L2168): candidates with p >= threshold, backing off to the
    single highest-scoring candidate when the predicted set is empty."""
    r = ranked(table, scores)
    r["pred"] = (r["score"] >= threshold).astype(int)
    empty = r.groupby("decision")["pred"].transform("sum") == 0
    r.loc[empty & (r["rank"] == 0), "pred"] = 1
    return r


def multilabel_prf(table: DecisionTable, scores: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Per-decision example-based precision, recall and F1 with max(1, .) denominators (L2276-2287)."""
    r = multilabel_predict(table, scores, threshold)
    r["tp"] = r["pred"] * r["y"].astype(int)
    g = r.groupby("decision", sort=True).agg(tp=("tp", "sum"), npred=("pred", "sum"), npos=("y", "sum"))
    prec = g["tp"] / np.maximum(1, g["npred"])
    rec = g["tp"] / np.maximum(1, g["npos"])
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec).replace(0, 1), 0.0)
    return pd.DataFrame({"precision": prec, "recall": rec, "f1": f1}, index=g.index)


def normalized(value: float, chance: float, oracle: float) -> float:
    """Headroom score Norm = (m - chance) / (oracle - chance) (L640, L1181)."""
    if oracle == chance:
        return float("nan")
    return (value - chance) / (oracle - chance)


_K = re.compile(r"^(recall|hit)@(\d+)$")


def per_decision(table: DecisionTable, scores: np.ndarray, metric: str, **kw) -> pd.Series:
    """Dispatch a per-decision metric by name: ``p@r``, ``recall@K``, ``hit@K``, ``route@1``, ``accuracy``,
    ``ml_precision``, ``ml_recall``, ``ml_f1``."""
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
