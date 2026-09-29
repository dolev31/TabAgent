"""Public benchmark datasets as TabAgent decision tables.

    from tabagent import datasets
    datasets.available()                                    # what can be built, with licenses and paper rows
    table = datasets.load("skillsbench/skill-select", download=True)
    card = datasets.card("skillsbench/skill-select")         # the paper's frozen feature card
    folds = datasets.folds("skillsbench/skill-select")       # the paper's task-level fold assignment

Nothing is redistributed: ``load`` rebuilds each table from the upstream release at a pinned revision
(verified by SHA-256) and caches the result under ``$TABAGENT_CACHE``. Some sources need an explicit opt-in:
LLMRouterBench (license unresolved upstream, ``accept_license=True``) and AppWorld (obtained only through the
official ``appworld download data`` command).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from importlib import resources

import pandas as pd

from tabagent.datasets._core import DatasetInfo, Source, cache_root
from tabagent.table import DecisionTable
from tabagent.tabschema.card import FeatureCard

_MODULES = ("tau2", "skillsbench", "mind2web", "llmrouterbench", "appworld")


def _registry() -> dict[str, DatasetInfo]:
    reg: dict[str, DatasetInfo] = {}
    for m in _MODULES:
        mod = importlib.import_module(f"tabagent.datasets.{m}")
        for info in getattr(mod, "DATASETS", []):
            reg[info.name] = info
    return reg


def available() -> pd.DataFrame:
    rows = [
        {
            "name": i.name,
            "decision": i.decision,
            "benchmark": i.source.benchmark,
            "license": i.source.license,
            "redistribution": i.source.redistribution,
            "paper": i.paper,
        }
        for i in _registry().values()
    ]
    return pd.DataFrame(rows).set_index("name") if rows else pd.DataFrame()


def info(name: str) -> DatasetInfo:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown dataset {name!r}; see tabagent.datasets.available()")
    return reg[name]


def card(name: str) -> FeatureCard:
    return FeatureCard.load(info(name).card)


def folds(name: str) -> dict:
    ds = info(name)
    if not ds.folds:
        raise KeyError(f"{name} has no pinned folds")
    res = resources.files("tabagent.datasets") / "folds" / f"{ds.folds}.json"
    return {g: int(f) for g, f in json.loads(res.read_text())["by_group"].items()}


def _cache_file(stem, ext: str):
    # Not Path.with_suffix: dataset names contain dots (gpt-4.1), which it would treat as a suffix.
    return stem.parent / f"{stem.name}.{ext}"


def _read_cache(stem) -> pd.DataFrame | None:
    """Cached tables are written by this library: parquet when pyarrow is installed, else pickle."""
    if _cache_file(stem, "parquet").exists():
        return pd.read_parquet(_cache_file(stem, "parquet"))
    if _cache_file(stem, "pkl").exists():
        return pd.read_pickle(_cache_file(stem, "pkl"))
    return None


def _write_cache(stem, df: pd.DataFrame) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(_cache_file(stem, "parquet"), index=False)
    except ImportError:
        df.to_pickle(_cache_file(stem, "pkl"))


def load(
    name: str, *, download: bool = False, accept_license: bool = False, rebuild: bool = False, **build_kwargs
) -> DecisionTable:
    """Build (or read from cache) the decision table ``name``, ready for its bundled card.

    ``build_kwargs`` go to the dataset's builder (for example ``root=`` for AppWorld's data directory).
    """
    ds = info(name)
    if ds.requires_license_acceptance and not accept_license:
        raise PermissionError(
            f"{name}: {ds.source.license}. {ds.source.notes} Pass accept_license=True to confirm "
            "that your use complies with the upstream terms."
        )
    tag = f"{name.replace('/', '__')}__v{ds.version}"
    if build_kwargs:
        tag += (
            "__"
            + hashlib.sha256(json.dumps(build_kwargs, sort_keys=True, default=str).encode()).hexdigest()[:12]
        )
    cached = cache_root() / "tables" / ds.source.release_id / tag
    df = None if rebuild else _read_cache(cached)
    if df is None:
        df = ds.build(ds.source, download=download, **{**ds.args, **build_kwargs})
        _write_cache(cached, df)
    return DecisionTable(df, card(name).spec)


__all__ = ["DatasetInfo", "Source", "available", "card", "folds", "info", "load"]
