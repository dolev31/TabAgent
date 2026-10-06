"""The last-step text TabAgent appends to every compiled extractor (not LLM-written): the text of the step
that handed control to the decision, which the context text can dilute among long payloads."""

from __future__ import annotations

import json

from test_robustness import TARGET, FakeLLM, adapter, discover, raw_run, runs

from tabagent import cli
from tabagent.tabschema import TabSchemaConfig
from tabagent.tabschema.compile import (
    ExtractorRunner,
    context_texts,
    last_step_text_code,
    last_step_texts,
)


def text_of(steps, **prefix) -> str:
    (out,) = last_step_texts([{**prefix, "steps": steps}])
    return out


def test_the_last_step_text_joins_its_fields_in_order():
    step = {
        "name": "Planner",
        "data": " d ",
        "thoughts": "t",
        "content": "c",
        "text": "x",
        "intent": "i",
        "n": 3,
    }
    assert text_of([{"name": "A", "data": "earlier"}, step]) == "c\nx\nt\ni\nd"


def test_a_long_last_step_is_clipped_as_the_context_text_is():
    long = "".join(f"word{i} " for i in range(2000)).strip()
    clipped = text_of([{"name": "A", "data": long}])
    assert clipped == long[:1000] + "\n...\n" + long[-3000:]
    assert context_texts([{"steps": [{"name": "A", "data": long}]}]) == [clipped]  # the same clipping
    assert text_of([{"name": "A", "data": "short"}]) == "short"


def test_a_last_step_without_text_passes_to_the_nearest_earlier_one():
    steps = [
        {"name": "Planner", "data": "plan: search then book"},
        {"name": "CodeAgent", "calls": ["search"], "data": "   "},
        "not a step",
        {"name": "system", "content": "You are an agent."},
    ]
    assert text_of(steps) == "plan: search then book"
    assert text_of([]) == "" and text_of([{"name": "system", "content": "x"}]) == ""
    assert last_step_texts([{"intent": "no steps at all"}]) == [""]


def test_outcome_fields_never_reach_it():
    prefix = {"intent": "go", "steps": [{"name": "A", "data": "plan", "content": "secret score 1.0"}]}
    assert last_step_texts([prefix], skip=("content",)) == ["plan"]
    with_outcome = {**prefix, "score": 1.0, "eval": {"passed": True}}
    assert last_step_texts([with_outcome], skip=("score", "eval")) == last_step_texts([prefix])


def test_it_ignores_the_run_after_the_decision():
    def changed(i):
        raw = raw_run(i)
        raw["steps"][2:] = [
            {"name": TARGET, "data": "play_song"},
            {"name": "PlanControllerAgent", "data": "a later plan, written after the decision"},
        ]
        return raw

    ad = adapter()
    before = [dp.prefix for t in runs() for dp in ad.decision_points(t, TARGET)]
    after = [dp.prefix for t in runs(make=changed) for dp in ad.decision_points(t, TARGET)]
    assert last_step_texts(before) == last_step_texts(after)
    assert all(t.startswith("plan: ") for t in last_step_texts(before))


def test_it_is_appended_kept_and_used_as_a_context_text(tmp_path):
    result = discover(tmp_path, FakeLLM(), last_step_text=True, pair_embeddings=True)
    ext, card = result.extractor, result.card
    assert ext.code.endswith(last_step_text_code("last_step_text", adapter().outcome_keys))
    assert ext.specs["last_step_text"]["added_by"] == "tabagent"
    feats = {f.name: f for f in card.features}
    assert (feats["last_step_text"].family, feats["last_step_text"].type) == ("state", "text")
    assert "last_step_text" in feats["context_candidate_lex"].args["a"]
    assert "last_step_text" in feats["context_candidate_sim"].args["query"]
    assert "last_step_text" in card.synth_text
    # the LLM's plan_note repeats the toy runs' last step word for word: the appended text is kept
    assert ext.dropped["plan_note"].startswith("redundant with last_step_text")
    # extract.py computes it in the sandbox exactly as here
    dps = [dp for t in runs() for dp in adapter().decision_points(t, TARGET)]
    sandboxed = ExtractorRunner(ext.code).run([dp.prefix for dp in dps])
    assert [f["last_step_text"] for f in sandboxed] == last_step_texts(
        [dp.prefix for dp in dps], skip=adapter().outcome_keys
    )


def test_it_is_on_by_default_and_can_be_turned_off(tmp_path):
    assert TabSchemaConfig().last_step_text is True
    off = discover(tmp_path, FakeLLM(), last_step_text=False)
    assert (
        "last_step_text" not in off.extractor.features
        and "_tabagent_last_step_text" not in off.extractor.code
    )
    parser = cli.build_parser()
    args = parser.parse_args(["onboard", "logs", "--decision", "X", "--no-last-step-text"])
    assert cli.tabschema_config(args, "m").last_step_text is False
    assert json.loads(json.dumps(TabSchemaConfig().to_dict()))["last_step_text"] is True
