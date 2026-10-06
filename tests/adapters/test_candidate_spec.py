"""A candidate's catalog entry (``Candidate.spec``; ``candidate["spec"]`` for the per-candidate function):
read from every catalog format, without CUGA's volatile fields, never a label, and the same at training and at
serving through every entry point (fabricated catalogs and runs)."""

from __future__ import annotations

import copy
import json
import re
import time

from fake_logs import API_DOCS, cuga_api_docs, cuga_trajectory, messages_api_tools, openai_run, openai_tools

from tabagent.adapters import CugaAdapter, OpenAIMessagesAdapter, parse_catalog
from tabagent.adapters.cuga import VOLATILE_API_KEYS
from tabagent.api import _as_candidates
from tabagent.integrations.cuga import CugaAppMatcher, CugaShortlister, run_so_far
from tabagent.tabschema.compile import ExtractorRunner, with_flatten
from tabagent.tabschema.sandbox import SubprocessSandbox
from tabagent.tabschema.trajectory import Candidate, candidate_dict, spec_text

ROUNDS = [
    ("phone", list(API_DOCS["phone"]), ["search_contacts"]),
    ("venmo", list(API_DOCS["venmo"]), ["send_money"]),
]


def test_every_catalog_format_keeps_its_entry():
    openai = openai_tools()[0]
    assert candidate_dict(parse_catalog([openai])["search_flights"])["spec"] == openai
    anthropic = messages_api_tools()[1]
    assert candidate_dict(parse_catalog([anthropic])["book_flight"])["spec"] == anthropic
    mcp = {"name": "get_weather", "description": "weather", "inputSchema": {"properties": {"city": {}}}}
    assert candidate_dict(parse_catalog([mcp])["get_weather"])["spec"] == mcp
    docs = cuga_api_docs("venmo")
    cuga = candidate_dict(parse_catalog({"venmo": docs})["venmo_send_money"])["spec"]
    assert (
        cuga == docs["venmo_send_money"] and cuga["method"] == "GET" and cuga["path"] == "/venmo_send_money"
    )
    assert cuga["parameters"][0] == {"name": "receiver_email", "type": "string", "required": True}
    assert parse_catalog({"f": "a description only"})["f"].spec == ""  # no entry: spec None
    assert candidate_dict(Candidate("f"))["spec"] is None


def test_cugas_volatile_fields_are_left_out():
    docs = cuga_api_docs("venmo")
    marked = copy.deepcopy(docs)
    for i, entry in enumerate(marked.values()):
        entry["canary_string"] = f"canary-{i}"
    assert parse_catalog({"venmo": marked}, VOLATILE_API_KEYS) == parse_catalog({"venmo": docs})
    # only the CUGA adapter and hot swap name them; any other catalog is kept whole
    kept = parse_catalog({"venmo": marked})["venmo_send_money"]
    assert candidate_dict(kept)["spec"]["canary_string"] == "canary-0"


def test_a_tool_parameter_named_like_a_volatile_field_is_kept():
    tool = {
        "type": "function",
        "function": {
            "name": "deploy",
            "description": "Deploy a build behind a canary.",
            "parameters": {
                "type": "object",
                "properties": {"canary_string": {"type": "string", "description": "the canary's tag"}},
                "required": ["canary_string"],
            },
        },
    }
    for volatile in ((), VOLATILE_API_KEYS):  # top level only, even when the keys are named
        spec = candidate_dict(parse_catalog([tool], volatile)["deploy"])["spec"]
        assert spec == tool and spec["function"]["parameters"]["required"] == ["canary_string"]
    docs = cuga_api_docs("venmo")  # a CUGA entry: only its own top-level canary goes
    entry = docs["venmo_send_money"]
    entry["response_schemas"] = {"success": {"canary_string": "str"}}
    spec = candidate_dict(parse_catalog({"venmo": docs}, VOLATILE_API_KEYS)["venmo_send_money"])["spec"]
    assert spec["response_schemas"] == {"success": {"canary_string": "str"}}


def _available(step):
    text = next(p["value"] for p in step["prompts"] if "Available APIs" in str(p["value"]))
    return json.loads(re.search(r'`Available APIs`:\s*"""(.*?)"""', text, re.S).group(1))


class Step:
    def __init__(self, d):
        self.d = d

    def model_dump(self):
        return dict(self.d)


class Tracker:
    def __init__(self, raw, upto):
        self.intent, self.dataset_name = raw["intent"], raw["dataset_name"]
        self.steps = [Step(s) for s in raw["steps"][:upto]]


class StubHead:
    def __init__(self):
        self.calls = []

    def decide(self, run=None, candidates=None, k=None, defer_below=None):
        self.calls.append(candidates)
        return {"candidates": [c.id for c in candidates][: k or 1], "scores": {}, "deferred": False}


def test_the_cuga_hot_swap_offers_the_training_candidates_and_entries(tmp_path):
    raw = cuga_trajectory("t1", ROUNDS, offered_apps=["phone", "venmo"])
    (tmp_path / "t1.json").write_text(json.dumps(raw))
    ad = CugaAdapter()
    (traj,) = ad.load(tmp_path)
    head = StubHead()
    swap = CugaShortlister(head, k=1)
    for dp in ad.decision_points(traj, "ShortlisterAgent"):
        apis = _available(dp.target_step)
        for entry in apis.values():  # live: CUGA writes another canary string into each entry
            for doc in entry.values():
                doc["canary_string"] = "a canary of the live run"
        swap.shortlist(run_so_far(Tracker(raw, dp.step_index)), apis)
        live, training = head.calls[-1], ad.candidates(dp)
        assert live == training and [candidate_dict(c) for c in live] == [candidate_dict(c) for c in training]
        assert all(candidate_dict(c)["spec"]["method"] == "GET" for c in live)
    (dp,) = ad.decision_points(traj, "TaskAnalyzerAgent")
    offered = [{"name": c.id, "description": c.description} for c in ad.candidates(dp)]
    CugaAppMatcher(head).match(run_so_far(Tracker(raw, 0)), offered)
    assert head.calls[-1] == ad.candidates(dp)
    assert candidate_dict(head.calls[-1][0])["spec"] == offered[0]


def test_a_catalog_given_to_the_adapter_or_to_decide_gives_the_hot_swaps_entries(tmp_path):
    """CUGA's API docs, with the canary of one run, read by ``CugaAdapter(catalog=...)``, by the hot swap and
    by ``decide``/``serve`` as raw entries: the same candidates, catalog entries included."""
    docs = cuga_api_docs("venmo")
    marked = copy.deepcopy(docs)
    for i, entry in enumerate(marked.values()):
        entry["canary_string"] = f"canary of run {i}"
    adapter = CugaAdapter(catalog={"venmo": marked}).catalog
    head = StubHead()
    CugaShortlister(head, k=1).shortlist(
        run_so_far(Tracker(cuga_trajectory("t1", ROUNDS), 1)), {"venmo": docs}
    )
    swapped = {c.id: c for c in head.calls[-1]}
    assert set(adapter) == set(swapped)
    assert all(candidate_dict(adapter[k]) == candidate_dict(swapped[k]) for k in adapter)
    raw = _as_candidates(list(marked.values()))  # the raw entries, as a tool filter or a request sends them
    assert [candidate_dict(c)["spec"] for c in raw] == [candidate_dict(adapter[c.id])["spec"] for c in raw]
    assert all("canary_string" not in candidate_dict(c)["spec"] for c in raw)


def test_an_entry_never_holds_a_label(tmp_path):
    raw = cuga_trajectory("t1", ROUNDS, offered_apps=["phone", "venmo"])
    (tmp_path / "t1.json").write_text(json.dumps(raw))
    for ad in (CugaAdapter(), CugaAdapter(app_labels="offered")):
        (traj,) = ad.load(tmp_path)
        for dp in ad.decision_points(traj, "ShortlisterAgent"):
            labels = ad.labels(dp)
            for c in ad.candidates(dp):
                spec = candidate_dict(c)["spec"]
                app = spec["app_name"]
                assert spec == cuga_api_docs(app)[c.id]  # the catalog entry, as offered, and nothing else
                assert "relevance_score" not in c.spec and "chosen" not in c.spec
            assert labels  # the labels exist, elsewhere
        (dp,) = ad.decision_points(traj, "TaskAnalyzerAgent")
        assert [json.loads(c.spec) for c in ad.candidates(dp)] == [
            {"description": c.description, "name": c.id} for c in ad.candidates(dp)
        ]  # the same entries whether the labels are the apps used or the apps chosen


def test_decide_and_the_service_read_the_same_candidates_from_every_form():
    docs = cuga_api_docs("phone")
    cands = list(parse_catalog({"phone": docs}).values())
    assert _as_candidates(cands) == cands  # Candidate objects
    records = [candidate_dict(c) for c in cands]
    assert _as_candidates(json.loads(json.dumps(records))) == cands  # the JSON a RemoteHead sends
    as_text = [{**r, "spec": json.dumps(r["spec"])} for r in records]
    assert _as_candidates(as_text) == cands  # the entry as JSON text
    assert _as_candidates(list(docs.values())) == [
        Candidate(c.id, c.id, c.description, c.arg_hints, c.spec) for c in cands
    ]  # the raw entries, as a tool filter sends them; the description is then the raw one
    assert _as_candidates(openai_tools()) == list(parse_catalog(openai_tools()).values())


def test_the_tool_filter_offers_the_training_candidates(tmp_path):
    run = openai_run(0)
    ad = OpenAIMessagesAdapter()
    (tmp_path / "r0.json").write_text(json.dumps(run))
    (traj,) = ad.load(tmp_path)
    dp = ad.decision_points(traj, "ToolShortlist")[0]
    training = sorted(ad.candidates(dp), key=lambda c: c.id)
    live = sorted(parse_catalog(run["tools"]).values(), key=lambda c: c.id)  # what ToolFilter passes
    assert live == training and all(candidate_dict(c)["spec"]["type"] == "function" for c in live)


def test_a_400_candidate_decision_with_1kb_entries_fits_the_sandbox():
    entry = {
        "method": "POST",
        "path": "/api/x",
        "description": "does a thing. " * 20,
        "parameters": [{"name": f"p{i}", "type": "string", "required": i % 2 == 0} for i in range(12)],
    }
    cands = [
        candidate_dict(Candidate(f"api_{i}", f"api {i}", "does a thing", "p0", spec_text({**entry, "id": i})))
        for i in range(400)
    ]
    assert 900 <= len(json.dumps(cands[0]["spec"])) <= 1200
    code = with_flatten(
        "def extract_features_for_classification(trajectory_data):\n"
        "    return {'n': len(trajectory_data['steps'])}\n"
        "def extract_candidate_features(trajectory_data, candidate):\n"
        "    spec = candidate['spec']\n"
        "    required = [p['name'] for p in spec['parameters'] if 'required' in p and p['required']]\n"
        "    return {'n_required': len(required), 'is_post': int(spec['method'] == 'POST')}\n"
        "features = extract_features_for_classification(trajectory)\n"
    )
    box = SubprocessSandbox()
    payload = box._payload(code, [{"steps": []}] * 50, [cands] * 50)  # a full chunk of such decisions
    assert len(payload) < box.chunk_bytes
    t0 = time.monotonic()
    out, per = ExtractorRunner(code, sandbox=box).run_items([{"steps": []}], [cands])
    assert time.monotonic() - t0 < 30
    assert out == [{"n": 0}] and per[0][0] == {"n_required": 6, "is_post": 1} and len(per[0]) == 400
