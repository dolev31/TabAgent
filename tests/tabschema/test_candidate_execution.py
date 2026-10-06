"""The per-candidate function of a compiled extractor, ``extract_candidate_features(trajectory_data,
candidate)``: its contract and its execution, the same in the subprocess sandbox, the warm workers and the
in-process runner (fabricated toy runs; no LLM)."""

from __future__ import annotations

import json
import time
import warnings

import pytest
from test_robustness import TOOLS, adapter, raw_run

from tabagent.tabschema import checks
from tabagent.tabschema.compile import ExtractorRunner, with_flatten
from tabagent.tabschema.sandbox import SubprocessSandbox, item_timeout
from tabagent.tabschema.trajectory import CANDIDATE_KEYS, Candidate, candidate_dict, spec_text, union_paths
from tabagent.tabschema.warm import WarmSandbox

#: Both functions: the prefix's intent, and per candidate its overlap with the request and whether the plan
#: names it. The prefix is parsed once per decision, in a cache the per-candidate calls share.
BOTH = """
_cache = {}


def _context(trajectory_data):
    if "words" not in _cache:
        plan = [s['data'] for s in trajectory_data['steps'] if s['name'] == 'PlanControllerAgent']
        _cache["words"] = set(trajectory_data['intent'].lower().split())
        _cache["plan"] = plan[-1] if plan else ""
    return _cache


def extract_features_for_classification(trajectory_data):
    return {"user_intent": trajectory_data['intent']}


def extract_candidate_features(trajectory_data, candidate):
    ctx = _context(trajectory_data)
    words = set((candidate['name'] + ' ' + candidate['description']).lower().split())
    return {
        "intent_overlap": len(ctx["words"] & words),
        "candidate_in_plan": int((candidate['description'].split() or [''])[0] in ctx["plan"]),
    }


features = extract_features_for_classification(trajectory)
"""
#: The same ideas as columns that index a catalog by position (the defect this design removes): each
#: catalog entry's text and its overlap with the request.
POSITIONAL = (
    "CATALOG = "
    + repr([f"{t.replace('_', ' ')} {d}" for t, d in TOOLS.items()])
    + """


def extract_features_for_classification(trajectory_data):
    words = set(trajectory_data['intent'].lower().split())
    out = {"user_intent": trajectory_data['intent']}
    for i, text in enumerate(CATALOG):
        out[f"candidate_{i}_text"] = text
        out[f"candidate_{i}_intent_overlap"] = len(words & set(text.split()))
    return out


features = extract_features_for_classification(trajectory)
"""
)


def overlap(intent: str, c: Candidate) -> int:
    return len(set(intent.lower().split()) & set(f"{c.name} {c.description}".lower().split()))


# --------------------------------------------------------------------------- the contract
def test_candidate_dict_holds_the_candidate_and_its_catalog_entry_never_a_label():
    c = Candidate("send_email", "", "send an email message", "to, body")
    assert candidate_dict(c) == {
        "id": "send_email",
        "name": "send_email",
        "description": "send an email message",
        "args": "to, body",
        "spec": None,
    }
    assert tuple(candidate_dict(c)) == CANDIDATE_KEYS
    entry = {"name": "send_email", "parameters": [{"name": "to", "required": True}], "canary_string": "x1"}
    whole = Candidate("send_email", spec=spec_text(entry))
    assert candidate_dict(whole)["spec"] == entry  # an entry is kept whole unless volatile keys are named
    with_spec = Candidate("send_email", spec=spec_text(entry, volatile=("canary_string",)))
    assert candidate_dict(with_spec)["spec"] == {
        "name": "send_email",
        "parameters": [{"name": "to", "required": True}],
    }
    assert spec_text(json.dumps(entry)) == whole.spec and spec_text(None) == spec_text("not json") == ""


def test_checks_allow_the_candidate_keys_and_refuse_any_other():
    known = union_paths([raw_run(0)])
    ok = BOTH.replace("candidate['description']", "candidate['description'] + candidate['args']")
    assert [i for i in checks.check_code(ok, known) if i.kind == "unknown_field"] == []
    for key in ("label", "chosen"):
        bad = BOTH.replace("candidate['description']", f"str(candidate[{key!r}])")
        issues = [i for i in checks.check_code(bad, known) if i.kind == "unknown_field"]
        assert issues and all(f"candidate[{key!r}]" in i.detail for i in issues)
        assert "id, name, description, args" in issues[0].detail
    # trajectory paths are checked as before, outcome fields stay forbidden, in either function
    reads = BOTH.replace(
        "ctx = _context(trajectory_data)", "ctx = _context(trajectory_data)\n    trajectory_data['score']"
    )
    assert any(i.kind == "outcome_field" for i in checks.check_code(reads, known, ("score",)))
    # a loop variable named candidate outside the function is not a candidate
    other = "for candidate in trajectory['steps']:\n    x = candidate['data']\nfeatures = {}\n"
    assert checks.check_code(other, union_paths([raw_run(0)])) == []
    assert checks.defines_candidate_function(BOTH) and not checks.defines_candidate_function(POSITIONAL)
    assert not checks.defines_candidate_function("def (:")


# --------------------------------------------------------------------------- execution
def _boxes():
    return [("in-process", None), ("subprocess", SubprocessSandbox()), ("warm", WarmSandbox(size=2))]


def test_every_runner_gives_the_same_per_candidate_features_and_isolates_a_raising_candidate():
    code = with_flatten(
        BOTH.replace(
            "ctx = _context(trajectory_data)",
            "ctx = _context(trajectory_data)\n"
            "    if candidate['id'] == 'play_song':\n"
            "        raise ValueError('no music')",
        )
    )
    cands = [candidate_dict(c) for c in adapter().candidates(None)]
    prefixes = [raw_run(i) for i in range(3)]
    got = {}
    for name, box in _boxes():
        runner = ExtractorRunner(code, trust_extractor=box is None, sandbox=box)
        results = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got[name] = runner.run_items(prefixes, [cands] * 3, results_out=results)
        assert runner.n_candidate_failed == 3 and any("no music" in str(w.message) for w in caught)
        assert all("ValueError: no music" in r.candidate_errors[0] for r in results)
        assert all(len(r.candidate_errors) == 1 for r in results)  # one distinct error, recorded once
        if box is not None and hasattr(box, "close"):
            box.close()
    assert got["in-process"] == got["subprocess"] == got["warm"]
    feats, per = got["warm"]
    assert feats == [{"user_intent": raw_run(i)["intent"]} for i in range(3)]
    play = [c["id"] for c in cands].index("play_song")
    assert all(p[play] is None and all(v is not None for k, v in enumerate(p) if k != play) for p in per)
    intent = raw_run(0)["intent"]
    assert [v["intent_overlap"] for v in per[0] if v] == [
        overlap(intent, c) for c in adapter().candidates(None) if c.id != "play_song"
    ]


def test_an_interrupt_in_the_per_candidate_function_stops_the_in_process_run():
    code = with_flatten(BOTH.replace("ctx = _context(trajectory_data)", "raise KeyboardInterrupt"))
    runner = ExtractorRunner(code, trust_extractor=True)
    with pytest.raises(KeyboardInterrupt):
        runner.run_items([raw_run(0)], [[candidate_dict(Candidate("a"))]])
    failing = with_flatten(BOTH.replace("ctx = _context(trajectory_data)", "raise ValueError('bad')"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out, per = ExtractorRunner(failing, trust_extractor=True).run_items(
            [raw_run(0)], [[candidate_dict(Candidate("a"))]]
        )
    assert out[0] == {"user_intent": raw_run(0)["intent"]} and per == [[None]]  # an error is the candidate's


def test_a_single_function_extractor_runs_as_before_even_when_given_candidates():
    code = with_flatten("features = {'n': len(trajectory['steps'])}\n")
    runner = ExtractorRunner(code)
    assert not runner.has_candidate_function
    out, per = runner.run_items([raw_run(0)], [[candidate_dict(Candidate("a"))]])
    assert out == [{"n": 5}] and per == [None]


def test_a_serving_deadline_caps_the_limit_grown_by_the_candidates_in_every_sandbox():
    from tabagent.serve import Service

    cands = [[candidate_dict(Candidate(str(i))) for i in range(400)]]
    box = SubprocessSandbox(timeout_s=1.0)
    assert box._limits([{}], cands) == [9.0]  # 400 candidates: 9 times one prefix's limit
    box.max_timeout_s = 2.0
    assert box._limits([{}], cands) == [2.0] and box._limits([{}], None) == [1.0]
    assert json.loads(box._payload("features = {}", [{}], cands))["item_timeouts"] == [2.0]

    class Head:  # a served head whose extractor runs in a sandbox
        extractor_code = "features = {}"

        def __init__(self, sandbox):
            self.extractor_sandbox = sandbox

        def decide(self, run=None, candidates=None, k=None):
            return {}

    for sandbox in (SubprocessSandbox(timeout_s=10.0), WarmSandbox(size=1, timeout_s=10.0)):
        served = Service(Head(sandbox), request_timeout=3, warm=False, drift=False).sandbox
        assert (served.timeout_s, served.max_timeout_s) == (3.0, 3.0), type(sandbox).__name__
        assert item_timeout(served.timeout_s, 400, served.max_timeout_s) == 3.0


def test_the_time_limit_grows_with_the_candidates():
    assert item_timeout(30, 0) == 30 and item_timeout(30, 400) == 270 and item_timeout(30, 400, 60) == 60
    slow = with_flatten(
        "import time\n"
        "def extract_candidate_features(trajectory_data, candidate):\n"
        "    time.sleep(0.02)\n"
        "    return {'x': len(candidate['id'])}\n"
        "features = {}\n"
    )
    cands = [{"id": str(i), "name": "", "description": "", "args": ""} for i in range(100)]
    for box in (SubprocessSandbox(timeout_s=1.0), WarmSandbox(size=1, timeout_s=1.0)):
        # 100 calls take 2 s, past one prefix's 1 s, within the limit grown by the candidates (3 s)
        t0 = time.monotonic()
        (r,) = box.run_batch(slow, [raw_run(0)], [cands])
        assert time.monotonic() - t0 > 1.0
        assert r.ok and all(v is not None for v in r.candidates), r.candidate_errors
        if hasattr(box, "close"):
            box.close()
    hang = with_flatten(
        "def extract_candidate_features(trajectory_data, candidate):\n"
        "    if candidate['id'] == '5':\n"
        "        while True:\n"
        "            pass\n"
        "    return {'x': 1}\n"
        "features = {'n': 1}\n"
    )
    r = SubprocessSandbox(timeout_s=0.5).run(hang, raw_run(0), candidates=cands[:10])
    # the time ran out on candidate 5: the prefix's features and the candidates before it are kept
    assert r.ok and r.features == {"n": 1}
    assert [v is not None for v in r.candidates] == [True] * 5 + [False] * 5
    assert "time limit" in r.candidate_errors[-1]


def test_a_400_candidate_decision_is_one_call_per_decision():
    code = with_flatten(BOTH)
    cands = [
        {
            "id": f"api_{i}",
            "name": f"get thing {i}",
            "description": f"returns thing {i} of the flight",
            "args": "",
        }
        for i in range(400)
    ]
    for name, box in _boxes():
        runner = ExtractorRunner(code, trust_extractor=box is None, sandbox=box)
        runner.run_items([raw_run(0)], [cands[:2]])  # warm the pool
        t0 = time.monotonic()
        out, per = runner.run_items([raw_run(0)], [cands])
        took = time.monotonic() - t0
        assert len(per[0]) == 400 and all(v is not None for v in per[0])
        assert took < item_timeout(30, 400), name
        if hasattr(box, "close"):
            box.close()
