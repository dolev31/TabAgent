"""The FeatureAnalyzer's section on features of a candidate (``tabagent_analyzer_candidates``) and the note on
them for the FeatureAggregator, the judges and the MetaJudge (``tabagent_candidate_note``), on fabricated toy
runs with a scripted LLM (no network).

A chat agent's logs rarely print the tools a decision chooses among: the paper's FeatureAnalyzer prompt shows
the prefix and the target step, so without the section the analyzer never learns that the decision chooses
among candidates, and never proposes a feature of one."""

from __future__ import annotations

import json

from test_robustness import TARGET, TOOLS, FakeLLM, adapter, names_in, raw_run, runs, spec, tagged

from tabagent.tabschema import TabSchema, TabSchemaConfig, prompts
from tabagent.tabschema.agents import (
    EXAMPLE_CANDIDATES,
    CodeExtractor,
    FeatureAggregator,
    FeatureAnalyzer,
    FeatureSpec,
    _dump,
    analyzer_prompt,
    candidate_examples,
    fit_for_prompt,
    pick_examples,
)
from tabagent.tabschema.judges import Judge, MetaJudge
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.trajectory import Candidate, candidate_dict, union_schema

SECTION = "each row of the classification dataset is one candidate of one decision"
NOTE = "such a feature is obtainable"


def point():
    return adapter().decision_points(runs(1)[0], TARGET)[0]


def candidates(dp) -> list[dict]:
    return [candidate_dict(c) for c in adapter().candidates(dp)]


def paper_prompt(dp) -> str:
    """The FeatureAnalyzer prompt as the paper's template alone renders it (how every run asked before)."""
    return prompts.render(
        "feature_analyzer",
        trajectory_json=_dump(fit_for_prompt(dp.prefix, 400_000)),
        trajectory_schema=union_schema([dp.prefix]),
        target_name=TARGET,
        target_json=_dump(fit_for_prompt(dp.target_step, 100_000)),
        trajectory_file=dp.source or dp.trajectory_id,
    )


# --------------------------------------------------------------------------- the prompts
def test_the_analyzer_section_follows_the_paper_prompt_only_for_a_decision_with_candidates():
    dp, cands = point(), candidates(point())
    plain = analyzer_prompt(dp, TARGET)
    assert plain == paper_prompt(dp) == analyzer_prompt(dp, TARGET, candidates=[])
    full = analyzer_prompt(dp, TARGET, candidates=cands)
    assert full.startswith(plain.rstrip("\n"))
    tail = full[len(plain.rstrip("\n")) :]
    assert tail.count("# Additional rules") == 1 and SECTION in tail
    # three candidates picked from the seed and the point (pick_examples), as the extractor's prompt shows
    # them (values clipped, the entry fit to 2,000 characters of JSON)
    shown = pick_examples(cands, 0, dp.id)
    assert len(shown) == EXAMPLE_CANDIDATES and _dump(candidate_examples(shown)) in tail
    assert sum(f'"id": "{c["id"]}"' in tail for c in cands) == EXAMPLE_CANDIDATES
    for words in (
        "how many times the candidate was already called",
        "whether the latest user message mentions the candidate's name",
        "whether the candidate's required arguments appear in the conversation",
        '"id", "name", "description", "args" and "spec"',
        "computable before the component",
        "must never encode which candidate was chosen",
    ):
        assert words in tail, words
    assert "paper" not in tail.lower()
    huge = {**cands[0], "spec": {"description": "x" * 50_000}}
    assert len(analyzer_prompt(dp, TARGET, candidates=[huge])) < len(plain) + 6_000  # the entry is fit


def test_the_note_follows_the_aggregator_judge_and_metajudge_prompts_only_with_candidates():
    llm = FakeLLM()
    for stage in (FeatureAggregator(llm), Judge("relevance", llm), MetaJudge(llm)):
        assert stage.noted("prompt\n") == "prompt\n"  # without candidates: the paper's prompt, unchanged
        stage.per_candidate = True
        noted = stage.noted("prompt\n")
        assert noted.startswith("prompt\n\n# Additional rules") and NOTE in noted
        assert "never encode which candidate was chosen" in noted
        assert "its inputs are available for every candidate of every decision" in noted
        assert "judge its generality as you would any other feature's" in noted
        assert "hard-codes particular candidates is still specific" in noted
        assert "less general" not in noted
    assert names_in(prompts.render("tabagent_candidate_note")) == []  # no feature name to confuse a parser
    for kind, ids, on in (("ranking", frozenset({"a"}), True), ("single", frozenset({"a"}), False)):
        assert TabSchema._noted(Judge("impact", llm), kind, ids).per_candidate is on
    assert TabSchema._noted(Judge("impact", llm), "ranking", frozenset()).per_candidate is False


def test_cache_keys_change_only_for_decisions_with_candidates():
    """A run directory's stored analyzer answers stay valid for a decision without candidates (the same
    prompt, the same key); a ranking or multi-label decision's analyzer is asked again."""
    dp, cands = point(), candidates(point())
    stage = FeatureAnalyzer(FakeLLM())
    old = stage._key(paper_prompt(dp))
    assert stage._key(stage.prompt(dp, TARGET)) == old
    assert stage._key(stage.prompt(dp, TARGET, candidates=[])) == old
    assert stage._key(stage.prompt(dp, TARGET, candidates=cands)) != old


# --------------------------------------------------------------------------- end to end
#: The per-candidate function the scripted CodeExtractor writes once the plan asks for the feature: whether
#: the plan written before the decision names the candidate.
PER_CANDIDATE = """
def extract_features_for_classification(trajectory_data):
    return {"user_intent": trajectory_data['intent']}


def extract_candidate_features(trajectory_data, candidate):
    plan = [s['data'] for s in trajectory_data['steps'] if s['name'] == 'PlanControllerAgent']
    first = candidate['id'].split('_')[0]
    return {"candidate_named_in_plan": int(bool(plan) and first in plan[-1])}


features = extract_features_for_classification(trajectory)
"""
PLANNED = spec(
    "candidate_named_in_plan",
    "numeric",
    description="whether the plan written before the decision names the candidate",
    computation="for one candidate: whether the first word of its id is in the last plan step",
)


def analyzer(p, n):
    """An analyzer that proposes a feature of a candidate only when its prompt says the decision chooses
    among candidates, as the τ²-bench chat logs, whose steps do not list the tools, showed."""
    proposals = [spec("user_intent", "text")]
    if SECTION in p:
        proposals.append(PLANNED)
    return json.dumps({"potential_features": proposals})


def aggregator(p, n):
    known = {s["feature_name"]: s for s in (spec("user_intent", "text"), PLANNED)}
    kept = [known[x] for x in names_in(p) if x in known]
    return json.dumps({"filtered_features": kept, "rejected_features": []})


#: The prefix's function alone, for a plan without a feature of a candidate.
PREFIX_ONLY = PER_CANDIDATE[: PER_CANDIDATE.index("def extract_candidate_features")] + (
    "features = extract_features_for_classification(trajectory)\n"
)


def extractor(p, n):
    return tagged(PER_CANDIDATE if "candidate_named_in_plan" in p else PREFIX_ONLY)


def discover(tmp_path, llm, kind="ranking", run_id="r1", ad=None, **kw):
    cfg = TabSchemaConfig(
        **{
            "n_analyze": 3,
            "workers": 2,
            "pair_embeddings": False,
            "context_text": False,
            "last_step_text": False,
            "usefulness": False,
            **kw,
        }
    )
    schema = TabSchema(llm, cfg, ad or adapter(), store=RunStore(tmp_path / "runs", run_id))
    return schema.discover(runs(), TARGET, kind)


def test_a_proposed_feature_of_a_candidate_becomes_a_per_candidate_column(tmp_path):
    llm = FakeLLM(analyzer=analyzer, aggregator=aggregator, extractor=extractor)
    result = discover(tmp_path, llm)
    assert all(SECTION in p for p in llm.prompts("analyzer"))
    for role in ("aggregator", "judge_relevance", "judge_generality", "judge_impact", "meta_judge"):
        asked = llm.prompts(role)
        assert asked and all(NOTE in p and "candidate_named_in_plan" in p for p in asked), role
    ext = result.extractor
    assert ext.candidate_columns == ["candidate_named_in_plan"]
    assert "candidate_named_in_plan" not in ext.dropped
    feats = {f.name: f for f in result.card.features}
    assert feats["candidate_named_in_plan"].source == "extractor_candidate"
    by_id = {c.id: c for c in adapter().candidates(None)}
    table = TabSchema(llm, TabSchemaConfig(pair_embeddings=False), adapter()).build_table(
        result, runs(), TARGET
    )
    for row in table.df.to_dict("records"):
        plan = raw_run(int(row["task"].removeprefix("task")))["steps"][1]["data"]
        assert row["candidate_named_in_plan"] == int(by_id[row["candidate"]].id.split("_")[0] in plan)


def test_without_candidates_the_analyzer_and_the_judges_are_asked_as_before(tmp_path):
    llm = FakeLLM(analyzer=analyzer, aggregator=aggregator, extractor=extractor)
    result = discover(tmp_path, llm, kind="single")
    assert llm.prompts("analyzer") and not any(SECTION in p for p in llm.prompts("analyzer"))
    for role in ("aggregator", "judge_relevance", "meta_judge"):
        assert not any(NOTE in p or "# Additional rules" in p for p in llm.prompts(role)), role
    assert "candidate_named_in_plan" not in result.extractor.features


# --------------------------------------------------------------------------- label-free examples
def chosen_first(dp) -> list[Candidate]:
    """The toy catalog with the decision's chosen tools first: an adapter order that leaks the label."""
    chosen = dp.target_step["data"].split(", ")
    rest = [t for t in TOOLS if t not in chosen]
    return [Candidate(t, t.replace("_", " "), TOOLS[t]) for t in [*chosen, *rest]]


def test_no_prompt_changes_with_the_labels_or_with_the_chosen_candidate_listed_first(tmp_path):
    """The example candidates are picked from the seed, the decision point and the candidates alone: listing
    the chosen candidate first, or other labels, leaves every analyzer and extractor prompt byte-identical
    (the "tasks" sample, which reads no label, so that the same points are analyzed)."""
    asked = {}
    for name, ad in (
        ("as given", adapter()),
        ("chosen first", adapter(candidates_fn=chosen_first)),
        ("relabeled", adapter(labels_fn=lambda dp, raw: {"play_song"})),
    ):
        llm = FakeLLM(analyzer=analyzer, aggregator=aggregator, extractor=extractor)
        discover(tmp_path, llm, run_id=name.replace(" ", "_"), ad=ad, sample_strategy="tasks")
        asked[name] = {role: sorted(llm.prompts(role)) for role in ("analyzer", "extractor")}
    assert asked["as given"]["extractor"] and asked["as given"]["analyzer"]
    assert asked["chosen first"] == asked["as given"] == asked["relabeled"]
    # at the function level too: any order of a decision's candidates gives the same prompts
    dp, cands = point(), candidates(point())
    for order in (cands[::-1], [cands[2], *cands[:2], *cands[3:]]):
        assert analyzer_prompt(dp, TARGET, candidates=order) == analyzer_prompt(dp, TARGET, candidates=cands)
        assert pick_examples(order, 7, dp.id) == pick_examples(cands, 7, dp.id)
        ext = CodeExtractor(FakeLLM())
        plans = [FeatureSpec.from_dict(PLANNED)]
        assert ext.prompt(plans, dp.prefix, TARGET, candidates=pick_examples(order, 0, dp.id)) == ext.prompt(
            plans, dp.prefix, TARGET, candidates=pick_examples(cands, 0, dp.id)
        )


def test_the_points_of_a_fixed_catalog_show_different_examples():
    """With one catalog at every decision point, each point shows its own three candidates (not the same
    alphabetically first three everywhere), and a point shows the same ones on every run."""
    dps = [dp for t in runs() for dp in adapter().decision_points(t, TARGET)]
    shown = [tuple(c["id"] for c in pick_examples(candidates(dp), 0, dp.id)) for dp in dps]
    assert all(len(s) == EXAMPLE_CANDIDATES and list(s) == sorted(s) for s in shown)
    assert len(set(shown)) >= 5 and len({i for s in shown for i in s}) == len(TOOLS)
    again = [tuple(c["id"] for c in pick_examples(candidates(dp), 0, dp.id)) for dp in dps]
    assert again == shown
    other_seed = [tuple(c["id"] for c in pick_examples(candidates(dp), 1, dp.id)) for dp in dps]
    assert other_seed != shown
    prompts_ = {
        analyzer_prompt(dp, TARGET, candidates=candidates(dp)).split("# Additional rules")[1] for dp in dps
    }
    assert len(prompts_) >= 5
