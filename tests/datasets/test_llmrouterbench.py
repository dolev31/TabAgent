"""LLMRouterBench tau2 model routing, on fabricated model files (no upstream data)."""

import hashlib
import io
import json
import tarfile

import pytest

from tabagent import datasets
from tabagent.datasets import llmrouterbench as lrb
from tabagent.datasets._core import Source
from tabagent.evaluation.cv import stable_uid
from tabagent.table import DecisionTable

MODELS = ["gpt-5", "glm-4.6", "deepseek-r1-0528"]
QUERIES = [
    "You want to cancel reservation ABC123 and ask about the flight insurance refund.",
    "Return the item from order #W1 and exchange the product for a larger size.",
    "Your phone line has no network; check the sim and the data plan.",
    "Say hello.",
]


def _payload(model: str, drop: int | None = None) -> bytes:
    records = [
        {
            "index": i + 1,
            "origin_query": q,
            "prompt": q,
            "score": float((i + len(model)) % 2),
            "cost": 0.01 * (i + 1),
            "prompt_tokens": 100 + i,
            "completion_tokens": 10 + i,
        }
        for i, q in enumerate(QUERIES)
        if i != drop
    ]
    return json.dumps({"model_name": model, "records": records}).encode()


def _source(sid: str = "demo_router") -> Source:
    return Source(sid, "demo", "https://example.org", "rev", "rel", "MIT", "redistributable", "demo2026")


def _mirror(tmp_path, monkeypatch, drop: int | None = None) -> list[str]:
    files = []
    for m in MODELS:
        rel = f"{m}/tau2-test-{m}.json"
        p = tmp_path / "mirror" / "demo_router" / "rel" / "files" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_payload(m, drop if m == "glm-4.6" else None))
        files.append(rel)
    monkeypatch.setenv("TABAGENT_SOURCE_MIRROR", str(tmp_path / "mirror"))
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    return files


def _archive(tmp_path) -> tuple:
    """A fabricated bench-release.tar.gz and the matching manifest entries."""
    buf, entries = io.BytesIO(), {}
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for m in MODELS:
            data = _payload(m)
            member = f"bench-release/tau2/test/{m}/tau2-test-{m}.json"
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            entries[f"{m}/tau2-test-{m}.json"] = {
                "path": f"{m}/tau2-test-{m}.json",
                "archive": lrb.ARCHIVE,
                "member": member,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    path = tmp_path / "upstream" / lrb.ARCHIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(buf.getvalue())
    return path, entries


def test_build_dense_table(tmp_path, monkeypatch):
    files = _mirror(tmp_path, monkeypatch)
    df = lrb.build_tau2(_source(), files=files)
    assert len(df) == len(MODELS) * len(QUERIES)
    assert df.groupby("task_uid")["model_id"].nunique().eq(len(MODELS)).all()
    assert set(df["success"]) == {0, 1}
    row = df[(df.task_index == 2) & (df.model_id == "gpt-5")].iloc[0]
    assert row.success == (1 + len("gpt-5")) % 2 and row.eval_cost_usd == pytest.approx(0.02)
    q = QUERIES[1]
    qhash = hashlib.sha256(json.dumps(q, ensure_ascii=False).encode()).hexdigest()
    assert row.task_uid == stable_uid("task", "llmrouterbench_tau2_dense", f"tau2:2:{qhash[:16]}", qhash)
    DecisionTable(df, datasets.card("llmrouterbench/tau2").spec)


def test_incomplete_matrix_is_rejected(tmp_path, monkeypatch):
    files = _mirror(tmp_path, monkeypatch, drop=3)
    with pytest.raises(ValueError, match="dense"):
        lrb.build_tau2(_source(), files=files)


def test_missing_files_need_download(tmp_path, monkeypatch):
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TABAGENT_SOURCE_MIRROR", raising=False)
    with pytest.raises(FileNotFoundError, match="download=True"):
        lrb.build_tau2(_source(), files=["gpt-5/x.json"])


def test_extract_members_verifies_checksums(tmp_path):
    archive, entries = _archive(tmp_path)
    lrb.extract_members(archive, entries, tmp_path / "out")
    for rel in entries:
        assert (tmp_path / "out" / rel).is_file()
    bad = {k: {**v, "sha256": "0" * 64} for k, v in list(entries.items())[:1]}
    with pytest.raises(ValueError, match="checksum"):
        lrb.extract_members(archive, bad, tmp_path / "bad")
    ghost = {"x.json": {"member": "bench-release/tau2/test/x.json", "sha256": "0" * 64}}
    with pytest.raises(FileNotFoundError, match="does not contain"):
        lrb.extract_members(archive, ghost, tmp_path / "ghost")


def test_download_extracts_from_the_archive(tmp_path, monkeypatch):
    archive, entries = _archive(tmp_path)
    arch_entry = {
        "path": lrb.ARCHIVE,
        "url": archive.as_uri(),
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    manifest = {lrb.ARCHIVE: arch_entry, **entries}
    monkeypatch.setattr(Source, "manifest", property(lambda self: manifest))
    monkeypatch.setenv("TABAGENT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TABAGENT_SOURCE_MIRROR", raising=False)
    src = _source()
    assert lrb.model_files(src) == sorted(entries)
    df = lrb.build_tau2(src, download=True)
    assert df["model_id"].nunique() == len(MODELS)
    assert not (src.local_dir() / lrb.ARCHIVE).exists()  # the archive is deleted after extraction
    assert len(lrb.build_tau2(src)) == len(df)  # now served from the cache


def test_card_features(tmp_path, monkeypatch):
    df = lrb.build_tau2(_source(), files=_mirror(tmp_path, monkeypatch))
    card = datasets.card("llmrouterbench/tau2").clone()
    for f in card.features:
        if f.source == "fitted:knn_outcome":
            f.args["encoder"] = "hashing"
    table = DecisionTable(df, card.spec)
    train = table.subset(table.groups[:3])
    X = card.fit(train).transform(table.drop_labels())
    by_model = X.assign(model=df["model_id"]).groupby("model").first()
    assert by_model.loc["gpt-5", "price_out_per_1m"] == 10.0
    assert by_model.loc["deepseek-r1-0528", "is_reasoning_model"] == 1.0
    assert by_model.loc["glm-4.6", "model_family"] == "glm"
    assert list(X["query_domain"][::3]) == ["airline", "retail", "telecom", "unknown"]
    rate = train.df.groupby("model_id")["success"].mean()
    assert by_model.loc["gpt-5", "prior_success_rate"] == pytest.approx(rate["gpt-5"])
    assert X["model_success_on_similar"].between(0, 1).all()


def test_registry_entry():
    info = datasets.info("llmrouterbench/tau2")
    assert info.requires_license_acceptance and info.source.redistribution == "unresolved"
    with pytest.raises(PermissionError, match="accept_license"):
        datasets.load("llmrouterbench/tau2")
    manifest = info.source.manifest
    assert len(lrb.model_files(info.source)) == 12 and manifest[lrb.ARCHIVE]["url"].startswith("https://")
    assert all(len(e["sha256"]) == 64 for e in manifest.values())
    folds = datasets.folds("llmrouterbench/tau2")
    assert len(folds) == 278 and set(folds.values()) == set(range(5))
    feats = {f.name for f in datasets.card("llmrouterbench/tau2").features}
    assert {"prior_success_rate", "prior_mean_cost", "model_success_on_similar"} <= feats
