"""Backward compatibility: a head whose extract.py has one function (the paper's entry point only) loads,
extracts and decides exactly as it did when it was saved, and onboarding it again gives the same files.

The golden (``tests/fixtures/golden_single_function_head.json``, fabricated toy data) was built on commit
5f1cce7, before extractors could define per-candidate features, by :func:`build_golden`: a FakeLLM onboarding
of the toy shortlister of ``test_discover``, saved with ``onboard(out=...)``, and the head's decisions and
features on four fabricated prefixes. The repository holds no binary files, so the fixture keeps the head's
text files as they were written, its ``model.cbm`` as CatBoost's JSON model (rebuilt here, which changes the
file's bytes and so its manifest hash, not its predictions beyond rounding) and rebuilds the empty
``fitted.npz``."""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from test_discover import TOOLS, adapter, responder, write_logs

import tabagent
from tabagent.tabschema import FeatureCard, TabSchemaConfig
from tabagent.tabschema.card import has_relative
from tabagent.tabschema.llm import ScriptedLLM

GOLDEN = Path(__file__).parents[1] / "fixtures" / "golden_single_function_head.json"
#: Files of the saved head that onboarding again must reproduce byte for byte. The fitted model's files are
#: not among them: a CatBoost fit can differ between CPU architectures (the golden was built on arm64). The
#: card is compared without the relative features added since (``card.with_relative``).
SAME_BYTES = ("extract.py", "extractor_tests.json", "head.json")


def build_golden(dest: Path, tmp: Path, **config) -> dict:
    """Onboard the toy shortlister with the FakeLLM, save the head to ``dest / "head"`` and return its
    decisions and features on the golden prefixes (how the fixture was made, with no ``config``)."""
    logs = tmp / "logs"
    logs.mkdir()
    write_logs(logs)
    tabagent.onboard(
        logs,
        "ShortlisterAgent",
        adapter=adapter(),
        llm=ScriptedLLM(responder, model="scripted"),
        config=TabSchemaConfig(n_analyze=3, workers=2, pair_embeddings=False, **config),
        run_dir=tmp / "run",
        n_splits=4,
        out=dest / "head",
        iterations=40,
    )
    return decisions(dest / "head", PREFIXES)


def decisions(head_dir: Path, prefixes: list[dict], **load) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        head = tabagent.load(head_dir, **({"trust_extractor": True} | load))
    try:
        out = []
        for p in prefixes:
            for k in (None, 2):
                answer = head.decide(p, list(TOOLS), k=k)
                out.append({"prefix": p, "k": k, "answer": answer, "features": head.features_for(p)})
        return {"tools": list(TOOLS), "decisions": out}
    finally:
        head.close()


PREFIXES = [
    {
        "intent": "find a flight to rome and book it",
        "steps": [
            {"name": "TaskAnalyzerAgent", "data": "apps needed for: find a flight to rome and book it"},
            {"name": "PlanControllerAgent", "data": "plan: search then book"},
        ],
    },
    {
        "intent": "what's the weather in oslo",
        "steps": [
            {"name": "TaskAnalyzerAgent", "data": "apps needed for: what's the weather in oslo"},
            {"name": "PlanControllerAgent", "data": "plan: get"},
        ],
    },
    {
        "intent": "reply to the email about lima",
        "steps": [
            {"name": "TaskAnalyzerAgent", "data": "apps needed for: reply to the email about lima"},
            {"name": "PlanControllerAgent", "data": "plan: read then send"},
        ],
    },
    {"intent": "play a song from rome", "steps": []},
]


@pytest.fixture(scope="module")
def golden(tmp_path_factory) -> Path:
    """The golden head written out as a saved head directory (``head``), with ``decisions.json``."""
    from catboost import CatBoost

    rec = json.loads(GOLDEN.read_text())
    out = tmp_path_factory.mktemp("golden")
    head = out / "head"
    for name, text in rec["files"].items():
        (head / name).parent.mkdir(parents=True, exist_ok=True)
        (head / name).write_text(text)
    (out / "model.json").write_text(json.dumps(rec["model_json"]))
    model = CatBoost()
    model.load_model(str(out / "model.json"), format="json")
    model.save_model(str(head / "model" / "model.cbm"), format="cbm")
    np.savez(head / "model" / "fitted.npz")
    manifest = json.loads((head / "model" / "manifest.json").read_text())
    sha = hashlib.sha256((head / "model" / "fitted.npz").read_bytes()).hexdigest()
    assert manifest["sha256"]["fitted.npz"] == sha  # the very file the head was saved with
    manifest["sha256"]["model.cbm"] = hashlib.sha256((head / "model" / "model.cbm").read_bytes()).hexdigest()
    (head / "model" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "decisions.json").write_text(json.dumps(rec["decisions"]))
    return out


def _same_decisions(got: dict, want: dict) -> None:
    """The same features, rankings and deferrals; the same scores up to floating-point rounding."""
    assert got["tools"] == want["tools"] and len(got["decisions"]) == len(want["decisions"])
    for g, w in zip(got["decisions"], want["decisions"], strict=True):
        assert (g["prefix"], g["k"], g["features"]) == (w["prefix"], w["k"], w["features"])
        a, b = g["answer"], w["answer"]
        assert a["candidates"] == b["candidates"] and (a["deferred"], a["error"]) == (
            b["deferred"],
            b["error"],
        )
        assert a["scores"] == pytest.approx(b["scores"], rel=1e-9, abs=1e-12)
        assert a["confidence"] == pytest.approx(b["confidence"], rel=1e-9, abs=1e-12)


def test_the_golden_was_built_from_the_golden_prefixes(golden):
    saved = json.loads((golden / "decisions.json").read_text())
    assert [d["prefix"] for d in saved["decisions"][::2]] == PREFIXES and saved["tools"] == list(TOOLS)
    assert "extract_candidate_features" not in (golden / "head" / "extract.py").read_text()


@pytest.mark.parametrize(
    "load", [{}, {"trust_extractor": False, "sandbox": "subprocess"}, {"trust_extractor": False}]
)
def test_a_saved_single_function_head_decides_as_it_did(golden, load):
    """In-process, in the subprocess sandbox and in warm workers: the same features and the same scores."""
    saved = json.loads((golden / "decisions.json").read_text())
    _same_decisions(json.loads(json.dumps(decisions(golden / "head", PREFIXES, **load))), saved)


def test_onboarding_again_reproduces_the_golden_head(golden, tmp_path):
    """Without what was added since (the last-step text, the relative features' challenger): the same
    extractor, tests, head.json and card, byte for byte."""
    rec = build_golden(tmp_path / "again", tmp_path, last_step_text=False, relative_features=False)
    for name in (*SAME_BYTES, "model/card.yaml"):
        assert (tmp_path / "again" / "head" / name).read_bytes() == (golden / "head" / name).read_bytes(), (
            name
        )
    assert not has_relative(FeatureCard.load(tmp_path / "again" / "head" / "model"))
    saved = json.loads((golden / "decisions.json").read_text())
    assert [d["features"] for d in rec["decisions"]] == [d["features"] for d in saved["decisions"]]


def test_onboarding_again_adds_only_the_last_step_text(golden, tmp_path):
    """With the defaults, extract.py is the golden's with the fixed last-step-text code appended. In the toy
    runs the last step is always the plan note, so the LLM's ``plan_note`` repeats it word for word and is
    dropped as redundant (the appended text is kept); the other features are unchanged."""
    from tabagent.tabschema.compile import last_step_text_code

    rec = build_golden(tmp_path / "again", tmp_path)
    old = (golden / "head" / "extract.py").read_text()
    new = (tmp_path / "again" / "head" / "extract.py").read_text()
    skip = adapter().outcome_keys
    assert new == old + last_step_text_code("last_step_text", skip)
    saved = json.loads((golden / "decisions.json").read_text())
    for got, want in zip(rec["decisions"], saved["decisions"], strict=True):
        last = got["features"].pop("last_step_text")
        assert last == (got["prefix"]["steps"] or [{"data": ""}])[-1]["data"]
        assert got["features"] == {k: v for k, v in want["features"].items() if k != "plan_note"}
    card = json.dumps(json.loads(json.dumps(tabagent.load(tmp_path / "again" / "head").card.to_dict())))
    assert '"plan_note"' not in card and '"last_step_text"' in card
    # the recipe search tried the relative features, and head.json records whether it kept them
    meta = json.loads((tmp_path / "again" / "head" / "head.json").read_text())
    fitted = FeatureCard.load(tmp_path / "again" / "head" / "model")
    assert meta["relative_features"] is has_relative(fitted)
