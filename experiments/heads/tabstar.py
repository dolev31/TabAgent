"""TabSTAR as a TabHead estimator (the neural head of the head ablation, paper App. ``head_alt``, L2261).

TabSTAR reads raw text and numeric columns itself, so it is used with ``TabHead(featurize=False)``: the
card's feature frame goes to it unchanged. Settings follow the paper: LoRA fine-tuning with learning rate
0.001, rank 16 and at most 50 epochs (the library default rank is 32, so it is set explicitly).

The collapse guard is ported from the pre-release evaluation harness. Some folds fall into a
constant-output attractor (every probability identical). After each fit the head probes the first
``probe_rows`` training rows; if the standard deviation of p(positive) is at most ``collapse_std`` it
refits once with a 5x lower LoRA learning rate (2e-4), best-checkpoint selection when the installed TabSTAR
supports it, and a different row shuffle. A second collapse is kept and recorded (``collapsed_``), never
silently dropped. Training sets under ``small_data_rows`` rows use the pre-release small-card schedule
(15 epochs, patience 6, best-checkpoint selection); skill selection (about 13.5K rows per fold) never does.

Constructor arguments are passed only if the installed TabSTAR accepts them (versions differ).
"""

from __future__ import annotations

import inspect
import logging
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PAPER_LORA = {"lora_lr": 1e-3, "lora_r": 16, "max_epochs": 50}
RETRY_LORA_LR = 2e-4
SMALL_DATA = {"max_epochs": 15, "patience": 6, "is_paper_version": True}


def _tabstar_classifier():
    try:
        from tabstar.tabstar_model import TabSTARClassifier
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("the TabSTAR head needs `pip install tabstar`") from e
    return TabSTARClassifier


class TabSTARHead:
    """sklearn-style binary classifier (``fit``/``predict_proba``/``classes_``) wrapping TabSTAR."""

    def __init__(
        self,
        seed: int = 0,
        lora_lr: float = PAPER_LORA["lora_lr"],
        lora_r: int = PAPER_LORA["lora_r"],
        max_epochs: int = PAPER_LORA["max_epochs"],
        device: str | None = None,
        verbose: bool = False,
        retry_lora_lr: float = RETRY_LORA_LR,
        small_data_rows: int = 1000,
        probe_rows: int = 256,
        collapse_std: float = 1e-4,
        classifier_cls: Any = None,
    ):
        self.seed = seed
        self.lora_lr, self.lora_r, self.max_epochs = lora_lr, lora_r, max_epochs
        self.device, self.verbose = device, verbose
        self.retry_lora_lr = retry_lora_lr
        self.small_data_rows = small_data_rows
        self.probe_rows = probe_rows
        self.collapse_std = collapse_std
        self.classifier_cls = classifier_cls
        self.model_: Any = None
        self.classes_ = np.array([0, 1])
        self.attempts_ = 0
        self.collapsed_ = False

    def _kwargs(self, accepted, n_rows: int, retry: bool) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "lora_lr": self.retry_lora_lr if retry else self.lora_lr,
            "lora_r": self.lora_r,
            "max_epochs": self.max_epochs,
            "random_state": self.seed,
            "verbose": self.verbose,
        }
        if self.device:
            kw["device"] = self.device
        if n_rows < self.small_data_rows:
            kw.update(SMALL_DATA)
        if retry:
            kw["is_paper_version"] = True  # best-checkpoint selection instead of checkpoint averaging
        return {k: v for k, v in kw.items() if k in accepted}

    def fit(self, X: pd.DataFrame, y) -> TabSTARHead:
        cls = self.classifier_cls or _tabstar_classifier()
        accepted = set(inspect.signature(cls.__init__).parameters)
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        for attempt in (0, 1):
            kw = self._kwargs(accepted, len(X), retry=attempt == 1)
            ckpt = None
            if "output_dir" in accepted:  # one disposable checkpoint dir per fit (concurrent jobs collide)
                ckpt = kw["output_dir"] = tempfile.mkdtemp(prefix="tabstar_")
            if "keep_model" in accepted:
                kw["keep_model"] = False
            try:
                clf = cls(**kw)
                perm = np.random.default_rng(self.seed + attempt).permutation(len(X))
                clf.fit(X.iloc[perm].reset_index(drop=True), pd.Series(y[perm]))
            finally:
                if ckpt:
                    shutil.rmtree(ckpt, ignore_errors=True)
            self.model_, self.attempts_ = clf, attempt + 1
            probe = self._positive(clf.predict_proba(X.head(self.probe_rows)))
            self.collapsed_ = float(np.std(probe)) <= self.collapse_std
            if not self.collapsed_:
                break
            action = f"retrying with lora_lr={self.retry_lora_lr:g}" if attempt == 0 else "kept and recorded"
            log.warning("TabSTAR collapsed to a constant output (attempt %d): %s", attempt + 1, action)
        return self

    @staticmethod
    def _positive(proba) -> np.ndarray:
        p = np.asarray(proba, dtype=float)
        return p if p.ndim == 1 else p[:, -1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("TabSTARHead is not fitted")
        p = np.asarray(self.model_.predict_proba(pd.DataFrame(X).reset_index(drop=True)), dtype=float)
        return np.column_stack([1.0 - p, p]) if p.ndim == 1 else p
