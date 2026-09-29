"""scikit-learn estimators: bring your own dataset.

If you already have your decisions as a table, these estimators give you the full TabAgent recipe in the
familiar ``fit`` / ``predict`` API. They infer a feature card from your columns, pick TabSynth and the
candidate prior by nested grouped validation on the training data, and fit a CatBoost TabHead.

    from tabagent import TabAgentRanker
    ranker = TabAgentRanker(group_col="task_id", candidate_col="tool")
    ranker.fit(X, y)                 # X: one row per (decision, candidate); y: 1 if the candidate was chosen
    ranker.rank(X_new, k=5)          # top-5 candidates per decision
    cross_val_score(ranker, X, y, groups=X.task_id, cv=GroupKFold(5), scoring=tabagent.scorer("p@r"))

``TabAgentRanker``      tool/skill/element shortlisting (``kind="ranking"``; multi-label via ``predict``)
``TabAgentClassifier``  single-class decisions (operation type, task category): a standard sklearn classifier
``TabAgentRouter``      policy decisions: route each decision to the arm with the highest predicted success

Use :func:`pointwise` to turn one-row-per-decision logs into the pointwise layout.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from tabagent import auto
from tabagent.evaluation import metrics as M
from tabagent.tabhead import decode
from tabagent.table import DecisionSpec, DecisionTable

_G, _Y, _C = "__group__", "__label__", "__candidate__"


def _frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    arr = np.asarray(X)
    return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


class _TabAgentBase(BaseEstimator):
    _kind = "ranking"

    def __init__(
        self,
        *,
        group_col: str | None = None,
        candidate_col: str | None = None,
        text_cols: Sequence[str] | None = None,
        pair_embeddings: bool | None = None,
        synth: str | int = "auto",
        prior: str | bool = "auto",
        preset: str = "paper",
        search: str = "fast",
        inner_folds: int = 3,
        threshold: float = 0.5,
        iterations: int | None = None,
        random_state: int = 0,
    ):
        self.group_col = group_col
        self.candidate_col = candidate_col
        self.text_cols = text_cols
        self.pair_embeddings = pair_embeddings
        self.synth = synth
        self.prior = prior
        self.preset = preset
        self.search = search
        self.inner_folds = inner_folds
        self.threshold = threshold
        self.iterations = iterations
        self.random_state = random_state

    # ------------------------------------------------------------------ table construction
    def _groups(self, X: pd.DataFrame, groups) -> np.ndarray:
        if groups is not None:
            return np.asarray(groups)
        if self.group_col is not None:
            return X[self.group_col].to_numpy()
        if self._kind == "single":
            return np.arange(len(X))
        raise ValueError(f"{type(self).__name__} needs decision groups: set group_col= or pass groups=")

    def _table(self, X: pd.DataFrame, y=None, groups=None) -> DecisionTable:
        df = X.copy()
        df[_G] = self._groups(X, groups)
        if self._kind != "single":
            if self.candidate_col is None:
                raise ValueError(f"{type(self).__name__} needs candidate_col= (the candidate id column)")
            df[_C] = X[self.candidate_col].astype(str)
        has_labels = y is not None
        if has_labels:
            df[_Y] = np.asarray(y)
        spec = DecisionSpec(
            kind=self._kind,
            group=_G,
            label=_Y,
            candidate=None if self._kind == "single" else _C,
            reward=_Y if self._kind == "policy" else None,
        )
        return DecisionTable(df, spec, has_labels=has_labels)

    def _feature_columns(self, X: pd.DataFrame) -> list[str]:
        return [c for c in X.columns if c != self.group_col]

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, groups=None):
        X = _frame(X)
        y = np.asarray(y)
        table = self._table(X, y, groups)
        cols = self._feature_columns(X)
        if self.candidate_col is not None:
            cols = [c if c != self.candidate_col else _C for c in cols]
        card = auto.infer_card(
            table.df,
            table.spec,
            cols,
            add_prior=self.prior is not False and self._kind != "single",
            text_cols=list(self.text_cols) if self.text_cols else None,
            pair_embeddings=self.pair_embeddings,
            name=type(self).__name__,
        )
        overrides = {} if self.iterations is None else {"iterations": self.iterations}
        fixed_synth = self.synth not in ("auto", None)
        if self.synth == "auto" or self.prior == "auto":
            recipe, self.selection_ = auto.select(
                table,
                card,
                search=self.search,
                inner_folds=self.inner_folds,
                seed=self.random_state,
                **overrides,
            )
        else:
            recipe, self.selection_ = auto.Recipe(), pd.DataFrame()
        if fixed_synth:
            recipe.synth_k = 0 if self.synth in (0, "off", False) else int(self.synth)
        if self.prior in (True, False):
            recipe.prior = bool(self.prior)
        recipe.preset = self.preset if self.search == "fast" else recipe.preset
        self.recipe_ = recipe
        self.agent_ = auto.build_agent(card, recipe, self.random_state, **overrides).fit(table)
        self.agent_.head_.threshold = self.threshold
        self.card_ = self.agent_.card_
        self.feature_types_ = dict(self.card_.types)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        if self._kind == "single":
            self.classes_ = np.asarray(self.agent_.head_.classes_, dtype=object)
        else:
            self.classes_ = np.array([0, 1])
        return self

    # ------------------------------------------------------------------ inference
    def _scores(self, X, groups=None) -> tuple[np.ndarray, DecisionTable]:
        X = _frame(X)
        if groups is None and self.group_col is None:
            groups = np.arange(len(X))  # scoring is pointwise; groups only matter for decoding
        table = self._table(X, None, groups)
        return self.agent_.score(table), table

    def report(
        self,
        X,
        y,
        groups=None,
        *,
        n_splits: int = 5,
        llm_cost_per_call: float | None = None,
        e5: bool = False,
    ):
        """Grouped-CV evaluation report of the fitted recipe on ``X, y`` (see ``tabagent.report``).

        The recipe (TabSynth, prior, preset) was selected on the data passed to ``fit``; if that is the same
        data, the report is slightly optimistic about the selection step (noted in the report).
        """
        from tabagent.report import evaluate

        X = _frame(X)
        table = self._table(X, np.asarray(y), groups)
        overrides = {} if self.iterations is None else {"iterations": self.iterations}
        agent = auto.build_agent(self.card_.clone(), self.recipe_, self.random_state, **overrides)
        rep = evaluate(agent, table, n_splits=n_splits, llm_cost_per_call=llm_cost_per_call, e5=e5)
        rep.notes.append(f"recipe: {self.recipe_.label()} (selected by nested inner validation during fit)")
        return rep

    def decision_function(self, X, groups=None) -> np.ndarray:
        return self._scores(X, groups)[0]

    def predict_proba(self, X, groups=None) -> np.ndarray:
        if self._kind == "single":
            X = _frame(X)
            feats = self.agent_.features(self._table(X, None, groups))
            return self.agent_.head_.predict_proba(feats)
        p = self._scores(X, groups)[0]
        return np.column_stack([1 - p, p])


class TabAgentRanker(_TabAgentBase):
    """Rank the candidates of each decision (tool, skill, API, GUI element shortlisting)."""

    _kind = "ranking"

    def rank(self, X, groups=None, k: int | None = None) -> pd.DataFrame:
        """Top-``k`` candidates per decision: columns decision, candidate, score, rank."""
        p, table = self._scores(X, groups)
        return decode("ranking", p, table.decision_ids.to_numpy(), table.df[_C].to_numpy(), k=k)

    def predict(self, X, groups=None) -> np.ndarray:
        """Multi-label decoding: 1 where p >= threshold, backing off to the top-1 candidate per decision."""
        p, table = self._scores(X, groups)
        chosen = decode(
            "multilabel", p, table.decision_ids.to_numpy(), table.df[_C].to_numpy(), threshold=self.threshold
        )
        keys = set(zip(chosen["decision"], chosen["candidate"], strict=True))
        pairs = zip(table.decision_ids, table.df[_C], strict=True)
        return np.array([int(pair in keys) for pair in pairs])

    def score(self, X, y, groups=None) -> float:
        """Mean P@R over decisions (the paper's headline shortlisting metric)."""
        X = _frame(X)
        p = self._scores(X, groups)[0]
        return float(M.p_at_r(self._table(X, y, groups), p).mean())


class TabAgentRouter(_TabAgentBase):
    """Route each decision to one arm (model, sub-agent, skill on/off). ``y`` is the arm's success (0/1)."""

    _kind = "policy"

    def route(self, X, groups=None) -> pd.DataFrame:
        p, table = self._scores(X, groups)
        return decode("policy", p, table.decision_ids.to_numpy(), table.df[_C].to_numpy())

    def predict(self, X, groups=None) -> np.ndarray:
        p, table = self._scores(X, groups)
        chosen = self.route(X, groups)
        keys = set(zip(chosen["decision"], chosen["candidate"], strict=True))
        return np.array([int(pair in keys) for pair in zip(table.decision_ids, table.df[_C], strict=True)])

    def score(self, X, y, groups=None) -> float:
        """Mean route@1: the realized reward of the chosen arm."""
        X = _frame(X)
        p = self._scores(X, groups)[0]
        return float(M.route_at_1(self._table(X, y, groups), p).mean())


class TabAgentClassifier(ClassifierMixin, _TabAgentBase):
    """Single-class decisions (e.g. operation type): a standard sklearn classifier."""

    _kind = "single"

    def predict(self, X, groups=None) -> np.ndarray:
        return self._scores(X, groups)[0]


# --------------------------------------------------------------------------- helpers
def scorer(metric: str = "p@r"):
    """A ``scoring=`` callable for sklearn model selection, using the estimator's ``group_col``."""

    def _score(estimator, X, y) -> float:
        X = _frame(X)
        p = estimator._scores(X)[0]
        table = estimator._table(X, y)
        if metric in ("accuracy",):
            return float(M.accuracy(table, p).mean())
        return float(M.per_decision(table, p, metric).mean())

    _score.__name__ = f"tabagent_{metric}"
    return _score


def pointwise(
    decisions: pd.DataFrame,
    *,
    chosen: str,
    catalog: Iterable | pd.DataFrame,
    candidate_col: str = "candidate",
    candidates: str | None = None,
    label_col: str = "chosen",
) -> pd.DataFrame:
    """Expand one-row-per-decision logs into (decision, candidate) rows.

    ``chosen`` names a column of list-likes with the candidates the agent selected. ``catalog`` is either
    an iterable of candidate ids or a DataFrame whose ``candidate_col`` column holds the ids (its other
    columns, such as descriptions, are joined onto every row). ``candidates`` optionally names a column of
    per-decision candidate lists (when each decision has its own legal set).
    """
    cat = (
        catalog.copy() if isinstance(catalog, pd.DataFrame) else pd.DataFrame({candidate_col: list(catalog)})
    )
    cat[candidate_col] = cat[candidate_col].astype(str)
    rows = []
    for _, r in decisions.iterrows():
        legal = [str(c) for c in r[candidates]] if candidates else cat[candidate_col].tolist()
        picked = {str(c) for c in (r[chosen] or [])}
        ctx = r.drop([chosen] + ([candidates] if candidates else [])).to_dict()
        rows += [{**ctx, candidate_col: c, label_col: int(c in picked)} for c in legal]
    out = pd.DataFrame(rows)
    extra = [c for c in cat.columns if c != candidate_col]
    return out.merge(cat, on=candidate_col, how="left") if extra else out
