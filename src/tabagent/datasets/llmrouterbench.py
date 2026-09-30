"""LLMRouterBench model routing on τ²-bench (paper Table 1, policy family; App. B.7).

LLMRouterBench (li2026llmrouterbench) ships, for each of its models, one cached evaluation per task. Its
τ²-bench test partition is dense: the same 278 tasks were run by 12 models, so every (task, model) pair has
an executed binary outcome and a realized cost. Each row of the decision table is one (task, model) pair:

  task_uid        group id (``stable_uid`` of the task index and a hash of its query)
  task_index      the task's index in the upstream files
  origin_query    the τ² task instruction, the only pre-decision context
  model_id        the candidate model
  success         executed binary outcome (label and reward)
  eval_*          realized cost (USD) and token counts, kept for scoring only

Static model metadata (family, reasoning flag, list prices from the upstream README) and the query features
are computed by the bundled card from these columns; every label-derived feature (candidate priors, the
model's success on similar tasks) is fitted inside the training folds.

The upstream release is a single 1.3 GB archive on Hugging Face. ``download=True`` fetches it once
(SHA-256 verified), extracts the 12 τ²-bench test files (66 MB, each verified) and deletes the archive.
LLMRouterBench is MIT-licensed, as its README declares (the repository ships no LICENSE file). TabAgent
redistributes none of the data; ``load`` asks for ``accept_license=True`` to confirm that your use complies
with the upstream terms.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from tabagent.datasets import _core
from tabagent.datasets._core import DatasetInfo, Source, sha256_file
from tabagent.evaluation.cv import stable_uid

HF_REVISION = "0e5af1b84bf73437a01a1849c0f1d2468baa93fc"
CODE_REVISION = "c77cb0506949d8f959e97967d2fefca0e8ff1b05"
ARCHIVE = "bench-release.tar.gz"

SOURCE = Source(
    id="llmrouterbench_tau2_dense",
    benchmark="LLMRouterBench",
    url="https://huggingface.co/datasets/NPULH/LLMRouterBench",
    revision=HF_REVISION,
    release_id=f"hf-{HF_REVISION}",
    license="MIT (declared in the upstream README; the repository ships no LICENSE file)",
    redistribution="redistributable",
    citation="li2026llmrouterbench",
    notes=(
        f"The LLMRouterBench README (github.com/ynulihao/LLMRouterBench at {CODE_REVISION[:7]}) declares the "
        "MIT license. TabAgent redistributes none of the data: it downloads the pinned archive from Hugging "
        "Face and keeps only the 12 tau2-bench test files."
    ),
)

# Public list prices ($ per 1M input / output tokens), README "Performance-Cost Setting" table.
PRICE = {
    "claude-sonnet-4": (3.00, 15.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "qwen3-235b-a22b-2507": (0.09, 0.60),
    "qwen3-235b-a22b-thinking-2507": (0.30, 2.90),
    "deepseek-v3-0324": (0.25, 0.88),
    "deepseek-v3.1-terminus": (0.27, 1.00),
    "deepseek-r1-0528": (0.50, 2.15),
    "glm-4.6": (0.60, 2.20),
    "kimi-k2-0905": (0.50, 2.00),
    "intern-s1": (0.18, 0.54),
}
REASONING = {"deepseek-r1-0528", "qwen3-235b-a22b-thinking-2507"}
FAMILIES = ("claude", "gemini", "gpt", "deepseek", "glm", "kimi", "qwen", "intern")
DOMAIN_KEYWORDS = {
    "airline": ("reservation", "flight", "cancel", "insurance", "baggage", "booking"),
    "retail": ("order", "return", "exchange", "refund", "item", "product"),
    "telecom": ("line", "plan", "sim", "network", "data", "phone number"),
}


# --------------------------------------------------------------------------- stateless card features
def _family(model: str) -> str:
    return next((f for f in FAMILIES if str(model).startswith(f)), "other")


def _domain(text: str) -> str:
    """The τ² domain with the most keyword hits (at least two), else ``unknown``."""
    t = (text or "").lower()
    best, score = "unknown", 0
    for dom, kws in DOMAIN_KEYWORDS.items():
        s = sum(k in t for k in kws)
        if s > score:
            best, score = dom, s
    return best if score >= 2 else "unknown"


def model_family(df: pd.DataFrame, col: str = "model_id") -> pd.Series:
    """The model's family (claude, gemini, gpt, ...), from its name."""
    return df[col].map(_family)


def is_reasoning_model(df: pd.DataFrame, col: str = "model_id") -> pd.Series:
    """1.0 for the two reasoning models of the release."""
    return df[col].isin(REASONING).astype(float)


def list_price(df: pd.DataFrame, col: str = "model_id", side: str = "in") -> pd.Series:
    """The model's public list price (USD per 1M ``in``put or ``out``put tokens)."""
    i = {"in": 0, "out": 1}[side]
    return df[col].map(lambda m: PRICE.get(m, (np.nan, np.nan))[i]).astype(float)


def price_rank(df: pd.DataFrame, col: str = "model_id") -> pd.Series:
    """Rank of the model's output price among the 12 models (1 = cheapest)."""
    rank = pd.Series({m: p[1] for m, p in PRICE.items()}).rank(method="min")
    return df[col].map(rank).astype(float)


def query_domain(df: pd.DataFrame, col: str = "origin_query") -> pd.Series:
    """The τ² domain the query's keywords point to (airline, retail, telecom or unknown)."""
    return df[col].fillna("").astype(str).map(_domain)


# --------------------------------------------------------------------------- source files
def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_members(archive: Path, entries: dict[str, dict], dest: Path) -> None:
    """Stream ``archive`` once and write each wanted member to ``dest/<path>``, checking its SHA-256.

    ``entries`` maps a release path to its manifest entry (``member``, ``sha256``).
    """
    wanted = {e["member"]: rel for rel, e in entries.items()}
    done = set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            rel = wanted.get(member.name)
            if rel is None or not member.isfile():
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            with tempfile.NamedTemporaryFile(dir=out.parent, delete=False) as tmp:
                while chunk := src.read(1 << 20):
                    tmp.write(chunk)
            tmp_path = Path(tmp.name)
            if sha256_file(tmp_path) != entries[rel]["sha256"]:
                tmp_path.unlink(missing_ok=True)
                raise ValueError(f"checksum mismatch for archive member {member.name}")
            tmp_path.replace(out)
            done.add(rel)
    missing = sorted(set(entries) - done)
    if missing:
        raise FileNotFoundError(f"{archive.name} does not contain {missing}")


def model_files(source: Source) -> list[str]:
    """The per-model τ² result files of the release (every manifest entry extracted from the archive)."""
    return sorted(p for p, e in source.manifest.items() if e.get("archive") == ARCHIVE)


def _resolve(source: Source, files: list[str], download: bool) -> dict[str, Path]:
    found, missing = {}, []
    for rel in files:
        try:
            found[rel] = source.path(rel)
        except FileNotFoundError:
            missing.append(rel)
    if not missing:
        return found
    if not download:
        raise FileNotFoundError(
            f"{source.id}: {len(missing)} of {len(files)} model files are not cached. Call with "
            f"download=True to fetch the pinned {ARCHIVE} (1.3 GB) from {source.url} at revision "
            f"{source.revision}; only the tau2-bench test files are kept."
        )
    manifest = source.manifest
    try:
        archive, fetched = source.path(ARCHIVE), False  # a mirrored or cached copy is kept
    except FileNotFoundError:
        entry = manifest[ARCHIVE]
        archive, fetched = _core.download(entry["url"], source.local_dir() / ARCHIVE, entry["sha256"]), True
    extract_members(archive, {rel: manifest[rel] for rel in missing}, source.local_dir())
    if fetched:
        archive.unlink(missing_ok=True)
    return {rel: source.path(rel) for rel in files}


# --------------------------------------------------------------------------- builder
def build_tau2(source: Source, download: bool = False, files: Iterable[str] | None = None) -> pd.DataFrame:
    """The dense (task x model) table; raises if any model misses a task."""
    files = sorted(files) if files is not None else model_files(source)
    if not files:
        raise FileNotFoundError(f"{source.id}: the manifest lists no model files")
    rows = []
    for rel, path in _resolve(source, files, download).items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model_name") or Path(rel).parts[0]
        for i, rec in enumerate(payload.get("records") or []):
            query = rec.get("origin_query") or rec.get("prompt")
            qhash = _canonical_hash(query)
            index = str(rec.get("index", i))
            rows.append(
                {
                    "task_uid": stable_uid("task", SOURCE.id, f"tau2:{index}:{qhash[:16]}", qhash),
                    "task_index": int(index),
                    "origin_query": query,
                    "model_id": model,
                    "success": int(round(float(rec["score"]))),
                    "eval_cost_usd": float(rec["cost"]),
                    "eval_prompt_tokens": int(rec["prompt_tokens"]),
                    "eval_completion_tokens": int(rec["completion_tokens"]),
                }
            )
    df = pd.DataFrame(rows)
    per_task = df.groupby("task_uid")["model_id"].nunique()
    n_models = df["model_id"].nunique()
    if len(files) != n_models or (per_task != n_models).any():
        raise ValueError(
            f"{source.id}: expected a dense matrix of {len(files)} models, got {n_models} models and "
            f"{int((per_task != n_models).sum())} incomplete tasks"
        )
    if df.duplicated(["task_uid", "model_id"]).any():
        raise ValueError(f"{source.id}: duplicate (task, model) records")
    return df.sort_values(["task_index", "model_id"]).reset_index(drop=True)


DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        name="llmrouterbench/tau2",
        source=SOURCE,
        build=build_tau2,
        card="llmrouterbench_model_routing",
        folds="llmrouterbench_tau2",
        decision="model routing (policy): route each tau2-bench task to one of 12 LLMs",
        paper="Table 1 (model routing); App. B.7; App. M, Table 20",
        description=(
            "278 tau2-bench test tasks x 12 models (3,336 rows); label and reward = executed success, "
            "with realized cost and tokens as evaluation columns."
        ),
        requires_license_acceptance=True,
    )
]
