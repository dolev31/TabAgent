"""Portable, pickle-free TabAgent artifacts.

A fitted TabAgent is saved as a directory that can be inspected, diffed, shared and loaded safely. There is
no pickle, so loading an artifact from someone else runs none of their code: card features that import a
module outside ``tabagent`` need ``trust_code=True``, and a TabSchema ``extract.py`` runs in the sandbox
unless explicitly trusted.

    card.yaml        the feature card (features, decision roles, provenance)
    fitted.json      fitted dependency statistics (priors, ...) and their training-group footprint
    fitted.npz       array state of fitted statistics (embedding centers, k-NN tables)
    featurizer.json  TF-IDF vocabularies and idf weights, numeric medians, categorical codes
    model.cbm        the CatBoost head in CatBoost's native format
    manifest.json    library version, head settings, sizes, metrics, and a SHA-256 of every file

``extract.py`` is copied alongside when the card was compiled by TabSchema.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1
FILES = ("card.yaml", "fitted.json", "fitted.npz", "featurizer.json", "model.cbm")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- featurizer
def featurizer_state(fz) -> dict[str, Any]:
    """The JSON-able state of a fitted ``Featurizer``."""
    parts = []
    for col, kind, state in fz.parts:
        if kind == "text":
            parts.append(
                {
                    "col": col,
                    "kind": kind,
                    "max_features": state.max_features,
                    "ngram_range": list(state.ngram_range),
                    "min_df": state.min_df,
                    "vocabulary": {t: int(i) for t, i in state.vocabulary_.items()},
                    "idf": [float(v) for v in state.idf_],
                }
            )
        elif kind == "num":
            parts.append({"col": col, "kind": kind, "median": None if np.isnan(state) else float(state)})
        else:
            parts.append({"col": col, "kind": kind, "codes": {str(k): int(v) for k, v in state.items()}})
    return {"types": fz.types, "text_max_features": fz.text_max_features, "parts": parts}


def featurizer_from_state(d: dict[str, Any]):
    """Rebuild a fitted ``Featurizer`` from :func:`featurizer_state` output."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    from tabagent.tabhead import Featurizer

    fz = Featurizer(d["types"], d["text_max_features"])
    for p in d["parts"]:
        if p["kind"] == "text":
            vec = TfidfVectorizer(
                max_features=p["max_features"], ngram_range=tuple(p["ngram_range"]), min_df=p["min_df"]
            )
            vec.vocabulary_ = dict(p["vocabulary"])
            vec.idf_ = np.asarray(p["idf"], dtype=float)
            fz.parts.append((p["col"], "text", vec))
        elif p["kind"] == "num":
            fz.parts.append((p["col"], "num", float("nan") if p["median"] is None else p["median"]))
        else:
            fz.parts.append((p["col"], "cat", dict(p["codes"])))
    return fz


# --------------------------------------------------------------------------- save / load
def save(agent, path: str | Path, metrics: dict[str, Any] | None = None) -> Path:
    """Write a fitted ``TabAgent`` to ``path`` (a directory)."""
    from catboost import CatBoost

    if agent.head_ is None:
        raise RuntimeError("only fitted agents can be saved")
    head = agent.head_
    if not isinstance(head.model_, CatBoost) or not head.featurize:
        raise TypeError("pickle-free artifacts support the CatBoost TabHead; other heads are experiment-only")

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    card = agent.card_
    card.save(out / "card.yaml")

    fitted_meta, arrays = {}, {}
    for name, feat in (card._fitted or {}).items():
        meta, arr = feat.state()
        fitted_meta[name] = {"class": type(feat).__name__, "args": feat.args, "state": meta}
        arrays.update({f"{name}::{k}": v for k, v in arr.items()})
    (out / "fitted.json").write_text(json.dumps(fitted_meta, indent=1, default=str))
    np.savez(out / "fitted.npz", **arrays)

    (out / "featurizer.json").write_text(json.dumps(featurizer_state(head.featurizer_)))
    head.model_.save_model(str(out / "model.cbm"), format="cbm")

    if card.extractor and card.extractor.get("path"):
        src = (
            Path(card.extractor["source_dir"]) / card.extractor["path"]
            if "source_dir" in card.extractor
            else None
        )
        if src and src.exists():
            shutil.copy(src, out / card.extractor["path"])

    import tabagent

    manifest = {
        "format_version": FORMAT_VERSION,
        "tabagent_version": tabagent.__version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "card": card.name,
        "head": {
            "kind": head.kind,
            "preset": head.preset,
            "threshold": head.threshold,
            "seed": head.seed,
            "text_max_features": head.text_max_features,
            "catboost_overrides": head.catboost_overrides,
            "classes": [c if isinstance(c, str) else int(c) for c in head.classes_],
            "columns": head.columns_,
        },
        "training": {"n_synthetic": agent.n_synthetic_},
        "metrics": metrics or {},
        "sha256": {f: _sha256(out / f) for f in FILES},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def load(path: str | Path, verify: bool = True, trust_code: bool = False):
    """Load a ``TabAgent`` saved with :func:`save`, from a directory or ``hf://OWNER/REPO[@REVISION]``
    (downloaded first, see ``tabagent.hub``).

    With ``verify``, the manifest must list every artifact file and each must match its SHA-256 (a missing
    entry is refused, not skipped). ``card.yaml`` in turn pins the hash of a TabSchema ``extract.py``. This
    detects modified or corrupted files; it is not a signature (whoever can rewrite every file and the
    manifest can make them agree). ``trust_code`` allows card features that import a module outside
    ``tabagent`` (see ``FeatureCard.load``).
    """
    from catboost import CatBoostClassifier

    from tabagent import hub
    from tabagent.agent import TabAgent
    from tabagent.tabhead import TabHead
    from tabagent.tabschema.card import FeatureCard
    from tabagent.tabschema.fitted import FITTED

    src = hub.resolve(path)
    manifest = json.loads((src / "manifest.json").read_text())
    if manifest["format_version"] > FORMAT_VERSION:
        raise ValueError("artifact format is newer than this library")
    if verify:
        unlisted = [f for f in FILES if f not in (manifest.get("sha256") or {})]
        if unlisted:
            raise ValueError(f"the manifest lists no hash for {', '.join(unlisted)} (modified artifact)")
        for f, digest in manifest["sha256"].items():
            if _sha256(src / f) != digest:
                raise ValueError(f"{f} does not match the manifest hash (modified or corrupted artifact)")

    card = FeatureCard.load(src / "card.yaml", trust_code=trust_code)
    arrays = dict(np.load(src / "fitted.npz", allow_pickle=False))
    card._fitted = {}
    classes = {cls.__name__: cls for cls in FITTED.values()}
    for name, entry in json.loads((src / "fitted.json").read_text()).items():
        feat = classes[entry["class"]](**entry["args"])
        own = {k.split("::", 1)[1]: v for k, v in arrays.items() if k.startswith(f"{name}::")}
        card._fitted[name] = feat.restore(entry["state"], own)

    h = manifest["head"]
    head = TabHead(
        h["kind"],
        preset=h["preset"],
        threshold=h["threshold"],
        seed=h["seed"],
        text_max_features=h["text_max_features"],
        **h["catboost_overrides"],
    )
    head.columns_ = h["columns"]
    head.featurizer_ = featurizer_from_state(json.loads((src / "featurizer.json").read_text()))
    model = CatBoostClassifier()
    model.load_model(str(src / "model.cbm"), format="cbm")
    head.model_ = model

    agent = TabAgent(card.clone(), head=head.clone())
    agent.card_, agent.head_ = card, head
    agent.n_synthetic_ = manifest["training"]["n_synthetic"]
    agent.manifest_ = manifest
    return agent
