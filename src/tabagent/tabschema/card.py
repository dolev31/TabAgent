"""The feature card: TabSchema's output and TabHead's input (paper Sec. 3, Fig. 1; example in App. I).

A card declares, for one decision:

* the decision roles (a ``DecisionSpec``: kind, group, candidate, label, ...);
* the features, each with a family (schema / state / dependency, paper Fig. 1), a declared type
  (text / num / cat) and a source:

  - ``column`` or a stateless primitive (``features.STATELESS``), or ``python:<module>:<fn>`` (modules
    outside ``tabagent`` only when the card is loaded with ``trust_code=True``);
  - ``fitted:<name>``, a dependency statistic fitted on the real training rows of a fold
    (``fitted.FITTED``), such as candidate priors;
  - ``extractor``, a column produced by the TabSchema-compiled extractor (``extract.py``);

* the free-text columns TabSynth may paraphrase, and provenance (who or what authored the card).

A card is serialized as YAML (``card.yaml``), next to ``extract.py`` when TabSchema compiled one.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from tabagent.table import DecisionSpec, DecisionTable, LeakageError
from tabagent.tabschema import features as stateless
from tabagent.tabschema.fitted import FITTED, FittedFeature

Family = Literal["schema", "state", "dependency"]
FeatureType = Literal["text", "num", "cat"]
CARD_VERSION = 1


@dataclass
class Feature:
    """One feature of a card.

    Parameters
    ----------
    name: the output column name.
    family: ``schema``, ``state`` or ``dependency`` (paper Fig. 1).
    type: ``text``, ``num`` or ``cat``; declared, never inferred per fold.
    source: ``column`` (the column ``name`` itself), a stateless primitive, ``python:<module>:<fn>``,
        ``fitted:<name>`` (fitted on training rows) or ``extractor`` (produced by TabSchema's extractor).
    args: arguments of the source.
    description, provenance: free-form documentation.
    """

    name: str
    family: Family
    type: FeatureType
    source: str = "column"
    args: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in ("schema", "state", "dependency"):
            raise ValueError(f"{self.name}: family must be schema, state or dependency")
        if self.type not in ("text", "num", "cat"):
            raise ValueError(f"{self.name}: type must be text, num or cat")

    @property
    def fitted(self) -> bool:
        """True for a dependency statistic fitted on training rows (``fitted:<name>``)."""
        return self.source.startswith("fitted:")


@dataclass
class FeatureCard:
    """A decision's feature card: its ``DecisionSpec``, its features and the text TabSynth may paraphrase.

    ``fit`` learns the fitted features' statistics on labeled training rows; ``transform`` computes every
    feature for any rows. Cards are saved and loaded as YAML (``save``, ``load``).

    Parameters
    ----------
    name: the card name.
    spec: the decision roles.
    features: the ``Feature`` list, in column order.
    synth_text: free-text feature columns TabSynth may paraphrase.
    synth_k: the card's TabSynth variants per row (``TabSynth`` uses it when no ``k`` is given).
    provenance: who or what authored the card, and the paper result it backs.
    extractor: the compiled TabSchema extractor it needs (``path``, ``sha256``), if any.
    """

    name: str
    spec: DecisionSpec
    features: list[Feature]
    synth_text: list[str] = field(default_factory=list)
    synth_k: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)
    extractor: dict[str, Any] | None = None
    card_version: int = CARD_VERSION
    _fitted: dict[str, FittedFeature] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = [f.name for f in self.features]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicate feature names {sorted(dup)}")
        clash = set(names) & set(self.spec.outcome_cols)
        if clash:
            raise LeakageError(f"features cannot reuse outcome columns {sorted(clash)}")
        for f in self.features:
            if f.fitted and f.source[len("fitted:") :] not in FITTED:
                raise KeyError(f"{f.name}: unknown fitted source {f.source!r}")

    # ------------------------------------------------------------------ properties
    @property
    def types(self) -> dict[str, str]:
        """Feature name -> declared type."""
        return {f.name: f.type for f in self.features}

    @property
    def feature_names(self) -> list[str]:
        """Feature names in column order."""
        return [f.name for f in self.features]

    @property
    def is_fitted(self) -> bool:
        """True after ``fit``."""
        return self._fitted is not None

    def context_text_columns(self) -> list[str]:
        """Free-text features describing the decision context (text features of the ``state`` family)."""
        return [f.name for f in self.features if f.type == "text" and f.family == "state"]

    def candidate_text_columns(self) -> list[str]:
        """Features describing the candidate in words (text or categorical features of the ``schema`` family),
        for example its id, name and description."""
        return [f.name for f in self.features if f.family == "schema" and f.type in ("text", "cat")]

    def clone(self) -> FeatureCard:
        """An unfitted deep copy."""
        new = copy.deepcopy(self)
        new._fitted = None
        return new

    # ------------------------------------------------------------------ fit / transform
    def fit(self, table: DecisionTable) -> FeatureCard:
        """Fit the dependency statistics on the (real, labeled) training rows."""
        if not table.has_labels:
            raise LeakageError("fitted features must be fitted on labeled training rows")
        self._check_spec(table)
        self._fitted = {}
        for f in self.features:
            if f.fitted:
                feat = FITTED[f.source[len("fitted:") :]](**f.args)
                self._fitted[f.name] = feat.fit(table.df, self.spec)
        return self

    def transform(self, table: DecisionTable) -> pd.DataFrame:
        """The feature frame for ``table`` (labeled or label-free), columns in card order."""
        if not self.is_fitted:
            raise RuntimeError("call fit() on training rows before transform()")
        self._check_spec(table)
        df = table.df
        out = {}
        for f in self.features:
            if f.fitted:
                col = self._fitted[f.name].transform(df, self.spec)
            elif f.source == "extractor":
                if f.name not in df.columns:
                    raise KeyError(f"{f.name}: run the card's extractor to produce this column first")
                col = df[f.name]
            else:
                args = f.args or ({"col": f.name} if f.source == "column" else {})
                col = stateless.resolve(f.source)(df, **args)
            out[f.name] = col.to_numpy() if isinstance(col, pd.Series) else np.asarray(col)
        return pd.DataFrame(out, index=df.index)

    def _check_spec(self, table: DecisionTable) -> None:
        if table.spec != self.spec:
            raise ValueError(f"table spec {table.spec} does not match card spec {self.spec}")

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        """The card as the plain dict written to ``card.yaml``."""
        spec = {k: v for k, v in asdict(self.spec).items() if v not in (None, (), [])}
        spec = {k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()}
        feats = []
        for f in self.features:
            d = {"name": f.name, "family": f.family, "type": f.type}
            if f.source != "column":
                d["source"] = f.source
            for key in ("args", "description", "provenance"):
                if getattr(f, key):
                    d[key] = getattr(f, key)
            feats.append(d)
        out: dict[str, Any] = {"card_version": self.card_version, "name": self.name, "decision": spec}
        out["features"] = feats
        if self.synth_text:
            out["synth"] = {"text_cols": list(self.synth_text), "k": self.synth_k}
        if self.extractor:
            out["extractor"] = self.extractor
        if self.provenance:
            out["provenance"] = self.provenance
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any], trust_code: bool = False) -> FeatureCard:
        """Build a card from ``to_dict`` output. Features computed by modules outside ``tabagent``
        (``python:`` sources) are refused unless ``trust_code``."""
        if d.get("card_version", 1) > CARD_VERSION:
            raise ValueError(f"card_version {d['card_version']} is newer than this library supports")
        if not trust_code:
            for f in d["features"]:
                module = stateless.python_module(f.get("source", "column"))
                if module is not None and not stateless.is_trusted_module(module):
                    raise PermissionError(
                        f"card feature {f['name']!r} runs code from module {module!r}. Load the card with "
                        "trust_code=True only if you trust that code."
                    )
        dec = dict(d["decision"])
        for key in ("decision", "eval"):
            if key in dec:
                dec[key] = tuple(dec[key])
        synth = d.get("synth") or {}
        return cls(
            name=d["name"],
            spec=DecisionSpec(**dec),
            features=[Feature(**f) for f in d["features"]],
            synth_text=list(synth.get("text_cols", [])),
            synth_k=int(synth.get("k", 0)),
            provenance=d.get("provenance", {}),
            extractor=d.get("extractor"),
            card_version=d.get("card_version", CARD_VERSION),
        )

    def save(self, path: str | Path) -> Path:
        """Write ``card.yaml`` (``path`` may be a directory or a .yaml file)."""
        p = Path(path)
        if p.suffix not in (".yaml", ".yml"):
            p.mkdir(parents=True, exist_ok=True)
            p = p / "card.yaml"
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=110))
        return p

    @classmethod
    def load(cls, path_or_name: str | Path, trust_code: bool = False) -> FeatureCard:
        """Load from a YAML file, a directory containing ``card.yaml``, or a bundled card name.

        A card whose features import code from a module outside ``tabagent`` (``python:<module>:<fn>``)
        raises ``PermissionError`` unless ``trust_code=True``.
        """
        p = Path(path_or_name)
        if p.is_dir():
            p = p / "card.yaml"
        if p.is_file():
            return cls.from_dict(yaml.safe_load(p.read_text()), trust_code)
        bundled = resources.files("tabagent.cards") / f"{path_or_name}.yaml"
        if bundled.is_file():
            return cls.from_dict(yaml.safe_load(bundled.read_text()), trust_code)
        raise FileNotFoundError(f"no card at {path_or_name!r} and no bundled card of that name")
