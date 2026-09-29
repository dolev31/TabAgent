"""Alternative TabHead estimators for the head ablation (paper Fig. 3, ``fig:card_new``; App. ``head_alt``).

Every entry of ``HEADS`` is a ``seed -> classifier`` factory that plugs into ``TabHead(estimator=...)``.
The classical heads consume the featurized card matrix (TF-IDF text, numeric, categorical codes); TabSTAR
reads the raw card columns itself, so it is used with ``TabHead(featurize=False)`` (see ``make_head``).

Hyperparameters are those of the pre-release head-ablation runs that produced Fig. 3 (E28): 300 trees of
depth 6 at learning rate 0.08 for the boosted heads, 300 trees for the random forest, a 256-256 MLP and an
L2 logistic regression on max-abs-scaled features. CatBoost follows a ``tabagent.tabhead.PRESETS`` entry
(``tmlr-runs`` reproduces the figure; ``paper`` is the configuration stated in Sec. 4, L384).

XGBoost, LightGBM and TabSTAR are optional: ``available()`` lists the heads whose packages import.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

from tabagent.tabhead import TabHead, catboost_estimator

Factory = Callable[[int], Any]


def logreg(seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, random_state=seed)
    )


def random_forest(seed: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def histgbm(seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6, random_state=seed)


def mlp(seed: int) -> Any:
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=200, random_state=seed)


def xgboost(seed: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, random_state=seed, n_jobs=4, eval_metric="logloss"
    )


def lightgbm(seed: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, random_state=seed, n_jobs=4, verbose=-1
    )


def catboost(preset: str = "tmlr-runs", kind: str = "ranking") -> Factory:
    """The main-paper head (CatBoost) as a factory, so it runs through the same code path as the others."""

    def factory(seed: int) -> Any:
        return catboost_estimator(preset, kind, seed)

    return factory


def tabstar(**kwargs) -> Factory:
    """TabSTAR with the paper's LoRA settings and the pre-release collapse-retry guard (``heads.tabstar``)."""
    from experiments.heads.tabstar import TabSTARHead

    def factory(seed: int) -> Any:
        return TabSTARHead(seed=seed, **kwargs)

    return factory


#: head name -> (factory builder, featurize, import needed). Order = plotting order of the paper figure.
HEADS: dict[str, tuple[Callable[..., Factory] | Factory, bool, str | None]] = {
    "catboost": (catboost, True, "catboost"),
    "histgbm": (histgbm, True, None),
    "random_forest": (random_forest, True, None),
    "logreg": (logreg, True, None),
    "mlp": (mlp, True, None),
    "xgboost": (xgboost, True, "xgboost"),
    "lightgbm": (lightgbm, True, "lightgbm"),
    "tabstar": (tabstar, False, "tabstar"),
}
CPU_HEADS = ("catboost", "histgbm", "random_forest", "logreg", "mlp", "xgboost", "lightgbm")
LABELS = {
    "catboost": "CatBoost (TabHead)",
    "histgbm": "HistGBM",
    "random_forest": "Random Forest",
    "logreg": "Logistic Reg.",
    "mlp": "MLP",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "tabstar": "TabSTAR (LoRA)",
}


def is_available(name: str) -> bool:
    if name not in HEADS:
        raise KeyError(f"unknown head {name!r}; choose from {list(HEADS)}")
    module = HEADS[name][2]
    return module is None or importlib.util.find_spec(module) is not None


def available(names=None) -> list[str]:
    return [h for h in (names or HEADS) if is_available(h)]


def featurizes(name: str) -> bool:
    return HEADS[name][1]


def factory(name: str, *, preset: str = "tmlr-runs", kind: str = "ranking", **kwargs) -> Factory:
    """The ``seed -> classifier`` factory of a head."""
    builder = HEADS[name][0]
    if name == "catboost":
        return builder(preset, kind)
    if name == "tabstar":
        return builder(**kwargs)
    return builder


def make_head(name: str, kind: str = "ranking", *, preset: str = "tmlr-runs", **kwargs) -> TabHead:
    """A ``TabHead`` running estimator ``name`` (``featurize=False`` for heads that read raw columns)."""
    if not is_available(name):
        raise ImportError(f"head {name!r} needs the {HEADS[name][2]!r} package")
    return TabHead(
        kind, estimator=factory(name, preset=preset, kind=kind, **kwargs), featurize=featurizes(name)
    )


__all__ = [
    "CPU_HEADS",
    "HEADS",
    "LABELS",
    "available",
    "factory",
    "featurizes",
    "is_available",
    "make_head",
]
