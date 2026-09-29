"""Evaluation report: how good is the head, against what, and at what cost.

``evaluate`` runs grouped cross-validation of a TabAgent on a decision table and compares it with the
training-free baselines, then measures the per-decision latency of the fitted head and prices it with the
paper's CPU cost model (L1192, L1946): ``C_cpu = threads * t / 3600 * p_vCPU-h`` with 4 threads and
$0.07 per vCPU-hour. Pass the cost of one call of the LLM component you replace to see the savings.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabagent import baselines as B
from tabagent.evaluation import metrics as M
from tabagent.evaluation import stats as S
from tabagent.evaluation.cv import cross_validate
from tabagent.table import DecisionTable

VCPU_HOUR_USD = 0.07
THREADS = 4


@dataclass
class Report:
    kind: str
    dataset: dict[str, Any]
    results: pd.DataFrame
    latency_ms: float | None = None
    cost_per_decision_usd: float | None = None
    llm_cost_per_call_usd: float | None = None
    importance: pd.Series | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "kind": self.kind,
            "dataset": self.dataset,
            "results": self.results.reset_index().to_dict(orient="records"),
            "latency_ms": self.latency_ms,
            "cost_per_decision_usd": self.cost_per_decision_usd,
            "llm_cost_per_call_usd": self.llm_cost_per_call_usd,
            "notes": self.notes,
        }
        if self.importance is not None:
            d["importance"] = {k: float(v) for k, v in self.importance.items()}
        return d

    def to_markdown(self) -> str:
        ds = self.dataset
        lines = [
            "# TabAgent evaluation report",
            "",
            f"**Decision:** {self.kind} · **decisions:** {ds['n_decisions']} · **rows:** {ds['n_rows']}"
            + (
                f" · **candidates/decision:** {ds['candidates_per_decision']:.1f}"
                if "candidates_per_decision" in ds
                else ""
            ),
            "",
            "## Quality (grouped cross-validation, 95% bootstrap CI over decisions)",
            "",
            "| method | " + " | ".join(self.results.columns) + " |",
            "|---|" + "---|" * len(self.results.columns),
        ]
        for name, row in self.results.iterrows():
            lines.append(f"| {name} | " + " | ".join(str(v) for v in row.to_numpy()) + " |")
        if self.latency_ms is not None:
            lines += [
                "",
                "## Efficiency",
                "",
                f"* latency per decision: **{self.latency_ms:.2f} ms** (CPU, {THREADS} threads)",
            ]
            lines.append(
                f"* cost per decision: **${self.cost_per_decision_usd:.1e}** (at ${VCPU_HOUR_USD}/vCPU-hour)"
            )
            if self.llm_cost_per_call_usd:
                ratio = self.llm_cost_per_call_usd / self.cost_per_decision_usd
                lines.append(
                    f"* vs. the LLM component (${self.llm_cost_per_call_usd:g}/call): "
                    f"**{ratio:,.0f}x cheaper**"
                )
        if self.importance is not None:
            lines += ["", "## Feature importance (CatBoost, summed over each feature's columns)", ""]
            lines += [f"* {k}: {v:.1f}" for k, v in self.importance.head(15).items()]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"* {n}" for n in self.notes]
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_markdown()

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "report.md").write_text(self.to_markdown())
        (p / "report.json").write_text(json.dumps(self.to_dict(), indent=1, default=str))
        return p


def _fmt(iv: S.Interval) -> str:
    return f"{iv.estimate:.3f} [{iv.low:.3f}, {iv.high:.3f}]"


def _context_and_candidate_text(card) -> tuple[list[str], list[str]]:
    query = [f.name for f in card.features if f.type == "text" and f.family == "state"]
    doc = [f.name for f in card.features if f.family == "schema" and f.type in ("text", "cat")]
    return query, doc


def feature_importance(agent) -> pd.Series:
    head = agent.head_
    raw = np.asarray(head.model_.get_feature_importance())
    sizes = [len(state.vocabulary_) if kind == "text" else 1 for _, kind, state in head.featurizer_.parts]
    names = [col for col, _, _ in head.featurizer_.parts]
    out, i = {}, 0
    for name, n in zip(names, sizes, strict=True):
        out[name] = float(raw[i : i + n].sum())
        i += n
    return pd.Series(out).sort_values(ascending=False)


def measure_latency(agent, table: DecisionTable, n: int = 20) -> float:
    """Median wall-clock milliseconds to score all candidates of one decision."""
    ids = table.decision_ids
    times = []
    for d in list(dict.fromkeys(ids))[:n]:
        one = table.with_df(table.df[(ids == d).to_numpy()]).drop_labels() if table.has_labels else table
        t0 = time.perf_counter()
        agent.score(one)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def evaluate(
    agent,
    table: DecisionTable,
    *,
    folds=None,
    n_splits: int = 5,
    seeds=(0,),
    ks=(3, 5, 9),
    baselines: bool = True,
    e5: bool = False,
    llm_cost_per_call: float | None = None,
) -> Report:
    kind = table.spec.kind
    res = cross_validate(agent, table, folds=folds, n_splits=n_splits, seeds=seeds)
    ids = table.decision_ids
    dataset = {"n_rows": len(table), "n_decisions": int(ids.nunique()), "n_groups": int(len(table.groups))}
    rows: dict[str, dict[str, str]] = {}
    notes = []
    if kind in ("ranking", "multilabel"):
        dataset["candidates_per_decision"] = float(ids.value_counts().mean())
        dataset["relevant_per_decision"] = float(
            table.df.groupby(ids.to_numpy())[table.spec.label].sum().mean()
        )
        metrics = ["p@r"] + [f"recall@{k}" for k in ks]
        rows["TabAgent"] = {m: _fmt(res.ci(m, level="decision")) for m in metrics}
        if baselines:
            rows["chance"] = {"p@r": f"{M.chance_p_at_r(table).mean():.3f}"}
            card = agent.card if hasattr(agent.card, "features") else None
            if card is not None:
                q, d = _context_and_candidate_text(card)
                if q and d:
                    s = B.bm25(table, q, d)
                    rows["BM25"] = {
                        m: _fmt(S.bootstrap_ci(M.per_decision(table, s, m).to_numpy())) for m in metrics
                    }
                    if e5:
                        s = B.embedding_zero_shot(table, q, d)
                        rows["e5 zero-shot"] = {
                            m: _fmt(S.bootstrap_ci(M.per_decision(table, s, m).to_numpy())) for m in metrics
                        }
    elif kind == "policy":
        rows["TabAgent"] = {"route@1": _fmt(res.ci("route@1", level="decision"))}
        rows["chance"] = {"route@1": f"{M.chance_route(table).mean():.3f}"}
        rows["oracle"] = {"route@1": f"{M.oracle_route(table).mean():.3f}"}
        spec, df = table.spec, table.df
        best_arm = df.groupby(spec.candidate)[spec.reward].mean().max()
        rows["best fixed arm"] = {"route@1": f"{best_arm:.3f}"}
        notes.append("the best fixed arm is chosen on all data (an optimistic reference)")
    else:
        acc = res.ci("accuracy", level="decision")
        f1 = float(np.mean([M.macro_f1(table, res.oof[s].to_numpy()) for s in res.seeds]))
        rows["TabAgent"] = {"accuracy": _fmt(acc), "macro_f1": f"{f1:.3f}"}
        majority = table.df[table.spec.label].value_counts(normalize=True).iloc[0]
        rows["majority class"] = {"accuracy": f"{majority:.3f}"}
    results = pd.DataFrame(rows).T.fillna("")

    fitted = agent.clone().fit(table)
    latency = measure_latency(fitted, table)
    cost = THREADS * latency / 1000 / 3600 * VCPU_HOUR_USD
    try:
        importance = feature_importance(fitted)
    except Exception:  # non-CatBoost heads
        importance = None
    return Report(kind, dataset, results, latency, cost, llm_cost_per_call, importance, notes)
