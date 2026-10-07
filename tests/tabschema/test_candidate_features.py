"""Per-candidate features: the optional ``extract_candidate_features(trajectory_data, candidate)`` of a
compiled extractor, from the prompts to serving, on fabricated toy runs with a scripted LLM (no network).

A ranking or multi-label decision table has one row per (decision, candidate). A feature that relates the
context to a candidate (the request's words in the candidate's description, the candidate named in the plan)
belongs to that row; written as columns that index candidates by position (``candidate_3_overlap``), it says
nothing about the candidate the row scores."""

from __future__ import annotations

import dataclasses
import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from test_candidate_execution import BOTH, POSITIONAL, overlap
from test_robustness import (
    TARGET,
    TOOLS,
    FakeLLM,
    adapter,
    names_in,
    raw_run,
    runs,
    spec,
    tagged,
)

import tabagent
from tabagent import auto
from tabagent.tabschema import TabSchema, TabSchemaConfig, prompts
from tabagent.tabschema.agents import CodeExecutor, CodeExtractor, CodeValidator, FeatureSpec, pick_examples
from tabagent.tabschema.compile import (
    POSITIONAL_REASON,
    TAKEN_REASON,
    ExtractorRunner,
    positional_columns,
)
from tabagent.tabschema.orchestrator import CANDIDATE, DECISION, extract_rows, rows_to_table
from tabagent.tabschema.runstore import RunStore
from tabagent.tabschema.trajectory import CANDIDATE_KEYS, candidate_dict, union_paths

# --------------------------------------------------------------------------- the scripted extractor
PROPOSALS = [
    spec("user_intent", "text"),
    spec("intent_overlap", "numeric", description="words the request shares with the candidate's text"),
    spec("candidate_in_plan", "numeric", description="whether the plan names the candidate"),
]


def extractor(p, n):
    """The positional code first; the per-candidate code once the repair round asks for it."""
    if "index the candidates by position" in p:
        return tagged(BOTH)
    return tagged(POSITIONAL if "extract_candidate_features" in p else BOTH)


def fake(**handlers) -> FakeLLM:
    return FakeLLM(
        **{
            "analyzer": lambda p, n: json.dumps({"potential_features": PROPOSALS}),
            "aggregator": lambda p, n: json.dumps({"filtered_features": PROPOSALS, "rejected_features": []}),
            "extractor": extractor,
            **handlers,
        }
    )


def config(**kw) -> TabSchemaConfig:
    return TabSchemaConfig(
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


def discover(tmp_path, llm, trajs=None, run_id="r1", kind="ranking", ad=None, **kw):
    schema = TabSchema(llm, config(**kw), ad or adapter(), store=RunStore(tmp_path / "runs", run_id))
    result = schema.discover(runs() if trajs is None else trajs, TARGET, kind)
    return schema, result


#: The decision's candidates as a live caller passes them (the dicts ``decide`` takes), as in training.
LIVE = [{"id": t, "name": t.replace("_", " "), "description": d} for t, d in TOOLS.items()]


# --------------------------------------------------------------------------- the contract
def test_a_candidate_label_read_fails_the_attempt_statically():
    bad = BOTH.replace("candidate['description']", "str(candidate['chosen'])")
    att = CodeExecutor(known_paths=union_paths([raw_run(0)])).run(bad, [raw_run(0)], [[]])
    assert not att.ok and "candidate['chosen']" in att.error


def test_the_prompt_additions_follow_the_verbatim_paper_prompts():
    specs = [FeatureSpec.from_dict(p) for p in PROPOSALS]
    ext = CodeExtractor(None)
    cands = [candidate_dict(c) for c in adapter().candidates(None)]
    plain = ext.prompt(specs, raw_run(0), TARGET)
    paper = prompts.render(
        "code_extractor",
        potential_features=json.dumps(
            [
                {
                    k: v
                    for k, v in s.to_dict().items()
                    if k
                    in (
                        "feature_name",
                        "feature_type",
                        "description",
                        "extraction_source",
                        "computation",
                        "classification_relevance",
                        "rationale",
                    )
                }
                for s in specs
            ],
            indent=2,
        ),
        trajectory_schema="",
        features_schema="",
        target_name=TARGET,
        retry_block="",
    )
    assert plain.startswith(paper.split("${")[0][:200])  # the paper's prompt comes first, unchanged
    assert "emit that text once, as plain text, under the feature's name" in plain
    assert "extract_candidate_features" not in plain
    with_cands = ext.prompt(specs, raw_run(0), TARGET, candidates=cands)
    assert with_cands.startswith(plain.rstrip("\n"))
    tail = with_cands[len(plain.rstrip("\n")) :]
    assert "extract_candidate_features(trajectory_data, candidate)" in tail
    shown = {c["id"] for c in pick_examples(cands)}  # three, picked label-free, never the first three
    assert len(shown) == 3 and all(f'"id": "{c["id"]}"' in tail for c in cands if c["id"] in shown)
    assert not any(f'"id": "{c["id"]}"' in tail for c in cands if c["id"] not in shown)
    assert "index the decision's candidates by position" in tail and "global dict" in tail
    assert "the code may define a second function" in tail  # optional, never required
    assert "a model's, a skill's or an application's description" in tail  # examples beyond APIs
    assert "When the catalog entries describe HTTP APIs" in tail  # the API examples only for APIs
    assert "history of earlier calls" in tail and "belong in `extract_features_for_classification`" in tail
    assert plain.count("# Additional rules") == 1 and with_cands.count("# Additional rules") == 1
    for name in prompts.ADDITIONS:  # the additions never mention the paper
        assert "paper" not in prompts.text(name).lower(), name
    assert set(prompts.ADDITIONS) & set(prompts.PROMPTS) == set()


def test_the_validator_sees_the_candidates_and_their_outputs():
    llm = FakeLLM()
    v = CodeValidator(llm)
    cands = [candidate_dict(c) for c in adapter().candidates(None)]
    outs = [{"candidate": cands[0], "features": {"intent_overlap": 2}}]
    v([FeatureSpec("intent_overlap")], BOTH, raw_run(0), [{"user_intent": "x"}], TARGET, cands, outs)
    (prompt,) = llm.prompts("validator")
    assert 'any field under `candidate["spec"]`' in prompt and '"intent_overlap": 2' in prompt
    v([FeatureSpec("intent_overlap")], BOTH, raw_run(0), [{"user_intent": "x"}], TARGET)
    assert "per-candidate features" not in llm.prompts("validator")[1]  # no candidates, no addition


#: Decision points with 4 to 6 candidates (the toy catalog, rotated: each point lists them in another order).
POINT_SIZES = [4, 5, 6, 4, 5]
#: Three candidates at every decision point: a family's size alone cannot tell a fixed-size list from them.
CONSTANT = [3] * 5


def _point_candidates(sizes=POINT_SIZES) -> list[list[dict]]:
    ids = list(TOOLS)
    return [
        [
            {"id": t, "name": t, "description": TOOLS[t], "args": "", "spec": None}
            for t in (ids[p:] + ids[:p])[:n]
        ]
        for p, n in enumerate(sizes)
    ]


def _family_rows(pattern: str, start: int, count, values: str, sizes=POINT_SIZES):
    """One prefix output per decision point with the family ``pattern`` at indices ``start..``: as many as the
    point's candidates (``count="all"``), a fixed number, or the first ``k`` of them (``("first", k)``).
    Values: the candidates' ids in order (``id``) or in reverse (``reversed``), numbers (``num``), numbers
    emitted only where they are not zero (``sparse``), or texts unrelated to the candidates (``text``)."""
    cands, rows = _point_candidates(sizes), []
    for p, cs in enumerate(cands):
        n = len(cs) if count == "all" else min(count[1], len(cs)) if isinstance(count, tuple) else count
        row = {"user_intent": "book a flight"}
        for j in range(n):
            if values == "sparse" and (p + j) % 3 == 0:
                continue  # a zero is not emitted
            col = pattern.format(i=start + j)
            row[col] = {
                "id": cs[j % len(cs)]["id"],
                "reversed": cs[-1 - j % len(cs)]["id"],
                "num": 10 * p + j,
                "sparse": (p + j) % 3,
                "text": f"a note {p}",
            }[values]
        rows.append(row)
    return rows, cands


def _family(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(k for r in rows for k in r if k != "user_intent"))


#: Indexed names that are not the decision's candidates: lists of earlier calls or steps (even when they hold
#: candidate ids, one per candidate), fixed-size lists (also with as many candidates at every point), the
#: first few of a longer enumeration with no candidate named, and families with no word for a candidate whose
#: texts do not name the candidates.
NOT_POSITIONAL = [
    ("prev_tool_{i}", 1, "all", "id", POINT_SIZES),
    ("last_tool_{i}", 1, "all", "id", POINT_SIZES),
    ("tool_history_{i}", 0, "all", "id", POINT_SIZES),
    ("tool_{i}_ago", 1, "all", "id", POINT_SIZES),
    ("last_app_{i}", 1, "all", "num", POINT_SIZES),
    ("recent_api_{i}", 0, "all", "id", POINT_SIZES),
    ("item_{i}_price", 1, 3, "num", POINT_SIZES),
    ("option_{i}_text", 1, 3, "text", POINT_SIZES),
    ("step_{i}_name", 1, "all", "id", POINT_SIZES),
    ("tools_used_{i}", 1, "all", "reversed", CONSTANT),
    ("item_{i}_price", 1, "all", "num", CONSTANT),
    ("candidate_{i}_overlap", 0, ("first", 3), "num", POINT_SIZES),
    ("entry_{i}_score", 0, "all", "num", POINT_SIZES),
]
#: Families that list the candidates by position: as many indices as candidates under a word for a candidate
#: anywhere in the name, emitted only where set but reaching the largest decision's last candidate, or texts
#: naming the candidates in their order (a fixed number of slots, a constant count, no word for a candidate).
POSITIONAL_FAMILIES = [
    ("candidate_{i}_overlap", 0, "all", "num", POINT_SIZES),
    ("api{i}_method", 1, "all", "text", POINT_SIZES),
    ("model_{i}_cost", 0, "all", "num", POINT_SIZES),
    ("skill_{i}_overlap", 1, "all", "num", POINT_SIZES),
    ("agent_{i}_score", 0, "all", "num", POINT_SIZES),
    ("function_{i}_match", 0, "all", "num", POINT_SIZES),
    ("action_{i}_overlap", 1, "all", "num", POINT_SIZES),
    ("endpoint_{i}_method", 0, "all", "text", POINT_SIZES),
    ("route_{i}_overlap", 0, "all", "num", POINT_SIZES),
    ("tool_overlap_{i}", 0, "all", "num", POINT_SIZES),
    ("available_api_{i}_path_len", 0, "all", "num", POINT_SIZES),
    ("application_{i}_score", 1, "all", "num", POINT_SIZES),
    ("element_{i}_text", 0, "all", "id", POINT_SIZES),
    ("subagent_{i}_fit", 0, "all", "num", POINT_SIZES),
    ("doc_{i}_overlap", 1, "all", "num", POINT_SIZES),
    ("candidate_{i}_overlap", 0, "all", "sparse", POINT_SIZES),
    ("api_{i}_name", 1, 3, "id", POINT_SIZES),
    ("option_{i}", 0, 3, "id", POINT_SIZES),
    ("tools_listed_{i}", 1, "all", "id", CONSTANT),
    ("entry_{i}", 1, ("first", 3), "id", POINT_SIZES),
]


@pytest.mark.parametrize(("pattern", "start", "count", "values", "sizes"), NOT_POSITIONAL)
def test_indexed_columns_that_are_not_the_candidates_stay(pattern, start, count, values, sizes):
    rows, cands = _family_rows(pattern, start, count, values, sizes)
    assert positional_columns(rows, cands) == []


@pytest.mark.parametrize(("pattern", "start", "count", "values", "sizes"), POSITIONAL_FAMILIES)
def test_columns_that_list_the_candidates_by_position_are_found(pattern, start, count, values, sizes):
    rows, cands = _family_rows(pattern, start, count, values, sizes)
    assert positional_columns(rows, cands) == _family(rows)


def test_positional_is_decided_from_the_values_not_the_name_alone():
    rows, cands = _family_rows("item_{i}_price", 1, "all", "num")  # one price per candidate: positional
    assert positional_columns(rows, cands) == _family(rows)
    one = [{"candidate_3_overlap": 1.0} for _ in cands]  # one index tells nothing
    assert positional_columns(one, cands) == []
    assert positional_columns(rows[:2], cands[:2]) == []  # too few decision points to tell
    # with as many candidates at every point, a numeric family cannot be told from a fixed-size list: kept
    rows, cands = _family_rows("candidate_{i}_overlap", 0, "all", "num", CONSTANT)
    assert positional_columns(rows, cands) == []
    rows, cands = _family_rows("entry_{i}_score", 0, "all", "num")  # no word for a candidate, no texts
    assert positional_columns(rows, cands) == []


# --------------------------------------------------------------------------- discovery and compile
def test_positional_columns_are_dropped_and_the_repair_round_moves_them(tmp_path):
    llm = fake()
    schema, result = discover(tmp_path, llm)
    ext = result.extractor
    assert ext.candidate_columns == ["intent_overlap", "candidate_in_plan"]
    assert ext.context_columns == ["user_intent"]
    asked = llm.prompts("extractor")
    assert any("index the candidates by position" in p and "candidate_0_intent_overlap" in p for p in asked)
    # one repair round: the positional features (the catalog's texts, which name the candidates in order, and
    # the overlaps listed with them), and candidate_in_plan, which the first code did not produce
    fixed = "candidate_#_text, candidate_in_plan, intent_overlap"
    note = f"the repair round fixed 3 of 3 flagged feature(s): {fixed}"
    assert any(note in n for n in ext.notes) and ext.not_produced == {}
    assert any(
        "The code does not produce these accepted features" in p and "candidate_in_plan" in p for p in asked
    )
    # the card: the per-candidate columns are schema features of their own source, never paraphrased
    feats = {f.name: f for f in result.card.features}
    assert (feats["intent_overlap"].family, feats["intent_overlap"].source) == (
        "schema",
        "extractor_candidate",
    )
    assert result.card.extractor["candidate_entrypoint"] == "extract_candidate_features"
    assert "intent_overlap" not in result.card.synth_text
    assert "intent_overlap" not in result.card.candidate_text_columns()
    record = json.loads((tmp_path / "runs" / "r1" / "compile.json").read_text())
    assert record["kept"]["intent_overlap"]["per_candidate"] is True
    md = result.report.to_markdown()
    assert "### Per-decision columns (1, `extract_features_for_classification`)" in md
    assert "### Per-candidate columns (2, `extract_candidate_features`)" in md
    assert record["per_decision"] == ["user_intent"]
    assert record["per_candidate"] == ["intent_overlap", "candidate_in_plan"]
    assert record["not_produced"] == {"count": 0, "by_reason": {}, "features": {}}
    # positional columns next to a per-candidate function that computes their feature are dropped
    helpers = BOTH[: BOTH.index("def extract_features_for_classification")]
    per_candidate = BOTH[BOTH.index("def extract_candidate_features") : BOTH.index("features = ")]
    mixed = POSITIONAL.replace("features = ", helpers + per_candidate + "features = ")
    _, both = discover(tmp_path, fake(extractor=lambda p, n: tagged(mixed)), run_id="r4")
    assert both.extractor.dropped["candidate_3_intent_overlap"] == POSITIONAL_REASON
    assert both.extractor.candidate_columns == ["intent_overlap", "candidate_in_plan"]
    assert any("indexed candidates by position and were dropped" in n for n in both.extractor.notes)
    # when the per-candidate function does not compute the feature, the positional columns stay, with a note
    _, plain = discover(tmp_path, fake(extractor=lambda p, n: tagged(POSITIONAL)), run_id="r2")
    assert "candidate_3_intent_overlap" in plain.extractor.features
    assert "candidate_3_intent_overlap" not in plain.extractor.dropped
    assert any("index candidates by position" in n and "kept" in n for n in plain.extractor.notes)
    assert "user_intent" in plain.extractor.features
    # the guard off: nothing is asked for or dropped for its position
    off_llm = fake()
    _, off = discover(tmp_path, off_llm, run_id="r3", positional_guard=False)
    assert not any("index the candidates by position" in p for p in off_llm.prompts("extractor"))
    assert POSITIONAL_REASON not in off.extractor.dropped.values()
    assert not any("by position" in n for n in off.extractor.notes)


def test_the_relative_features_and_the_positional_guard_are_switches(tmp_path):
    from tabagent import cli
    from tabagent.tabschema.card import has_relative

    _, on = discover(tmp_path, fake())
    _, off = discover(tmp_path, fake(), run_id="r2", relative_features=False)
    # never in the discovered card (usefulness, prune_harmful, a TabAgent on the card): the recipe search
    # tries them as a challenger when relative_features is on
    assert not has_relative(on.card) and on.card.to_dict() == off.card.to_dict()
    assert TabSchemaConfig().relative_features is True and TabSchemaConfig().positional_guard is True
    args = cli.build_parser().parse_args(
        ["onboard", "logs", "--decision", "X", "--no-relative-features", "--no-positional-guard"]
    )
    cfg = cli.tabschema_config(args, "m")
    assert cfg.relative_features is False and cfg.positional_guard is False
    default = cli.tabschema_config(cli.build_parser().parse_args(["onboard", "logs", "--decision", "X"]), "m")
    assert default.relative_features is True and default.positional_guard is True


def test_an_adapter_error_stops_discovery_and_no_candidates_turn_the_function_off(tmp_path, caplog):
    def broken(dp):
        raise RuntimeError("the catalog file is unreadable")

    ad = adapter()
    ad.candidates = broken
    llm = fake()
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="unreadable"):
        discover(tmp_path, llm, ad=ad)
    assert "the adapter's candidates failed: RuntimeError" in caplog.text
    assert llm.prompts("analyzer") == []  # before any LLM call
    schema = TabSchema(FakeLLM(), config(), ad, store=RunStore(tmp_path / "runs", "r2"))
    dp = adapter().decision_points(runs(1)[0], TARGET)[0]
    with pytest.raises(RuntimeError):
        schema._candidates(dp, "ranking")
    assert schema._candidates(dp, "single") == []  # a single-class decision reads no candidates

    def none(dp):
        raise NotImplementedError

    ad.candidates = none
    assert schema._candidates(dp, "ranking") == []  # an adapter without candidates: no per-candidate prompt


def test_rows_to_table_puts_each_candidates_features_on_its_own_row(tmp_path):
    _, result = discover(tmp_path, fake())
    ad = adapter()
    dps = [dp for t in runs()[:3] for dp in ad.decision_points(t, TARGET)]
    feats, per = extract_rows(
        ExtractorRunner(result.extractor.code, trust_extractor=True), result.card, dps, ad
    )
    table = rows_to_table(result.card, dps, feats, ad, per)
    by_id = {c.id: c for c in ad.candidates(None)}
    for row in table.df.to_dict("records"):
        assert row["intent_overlap"] == overlap(row["user_intent"], by_id[row[CANDIDATE]])
    flipped = [dict(reversed(m.items())) for m in per]  # the mapping's order does not matter
    pd.testing.assert_frame_equal(rows_to_table(result.card, dps, feats, ad, flipped).df, table.df)
    gone = next(iter(per[0]))  # a candidate without features gets none; the others keep theirs
    partial = [{k: v for k, v in per[0].items() if k != gone}, *per[1:]]
    df = rows_to_table(result.card, dps, feats, ad, partial).df
    first = df[DECISION] == dps[0].id
    assert df.loc[first & (df[CANDIDATE] == gone), "intent_overlap"].isna().all()
    rest = ~(first & (df[CANDIDATE] == gone))
    pd.testing.assert_frame_equal(df[rest], table.df[rest], check_dtype=False)


def test_candidate_columns_land_on_their_rows(tmp_path):
    schema, result = discover(tmp_path, fake())
    trajs = runs()
    table = schema.build_table(result, trajs, TARGET)
    by_id = {c.id: c for c in adapter().candidates(None)}
    for row in table.df.itertuples(index=False):
        row = row._asdict()
        assert row["intent_overlap"] == overlap(row["user_intent"], by_id[row[CANDIDATE]])
    plan_rows = table.df[table.df["candidate_in_plan"] == 1]
    assert plan_rows["chosen"].all() and not table.df["candidate_in_plan"].equals(table.df["chosen"])
    # in-process extraction gives the same table
    same = schema.build_table(result, trajs, TARGET, trust_extractor=True)
    assert same.df.equals(table.df)


def test_a_name_the_prefix_function_or_the_table_uses_is_dropped(tmp_path):
    code = BOTH.replace(
        '"candidate_in_plan": int(', '"user_intent": 1, "candidate_name": 2, "candidate_in_plan": int('
    )
    _, result = discover(tmp_path, fake(extractor=lambda p, n: tagged(code)))
    assert result.extractor.dropped["candidate_name"] == "name reserved by the decision table"
    assert result.extractor.dropped["user_intent (per-candidate)"] == TAKEN_REASON
    assert result.extractor.features["user_intent"] == "text"  # the prefix's column stands
    assert result.extractor.candidate_columns == ["intent_overlap", "candidate_in_plan"]


def test_a_failing_candidate_function_is_retried_and_otherwise_gated(tmp_path):
    flaky = BOTH.replace("ctx = _context(trajectory_data)", "ctx = _context(trajectory_data)\n    1 / 0")

    def ext(p, n):
        return tagged(BOTH if "extract_candidate_features failed on" in p else flaky)

    llm = fake(extractor=ext)
    _, result = discover(tmp_path, llm)
    assert result.extractor.candidate_columns  # fixed on the retry, which carried the traceback
    assert any("ZeroDivisionError" in p for p in llm.prompts("extractor"))
    _, never = discover(tmp_path, fake(extractor=lambda p, n: tagged(flaky)), run_id="r2")
    assert never.extractor.features == {"user_intent": "text"}  # the prefix's features stand
    assert never.extractor.dropped["intent_overlap"] == "accepted but not produced by the compiled extractor"
    # failing on most candidates: every per-candidate column is dropped with the success rate, recorded apart
    # from a prefix column of the same name
    mostly = BOTH.replace(
        "ctx = _context(trajectory_data)",
        "ctx = _context(trajectory_data)\n    if candidate['id'] != 'play_song':\n        1 / 0",
    ).replace('"intent_overlap":', '"user_intent": 1,\n        "intent_overlap":')
    _, low = discover(tmp_path, fake(extractor=lambda p, n: tagged(mostly)), run_id="r3")
    assert low.extractor.features == {"user_intent": "text"}
    rate = "the per-candidate function ran on 16 of 96 (decision, candidate) rows, under compile_min_success"
    assert low.extractor.dropped["intent_overlap"].startswith(rate)
    assert low.extractor.dropped["user_intent (per-candidate)"].startswith(rate)
    assert "user_intent" not in low.extractor.dropped


def test_quality_gates_caps_and_precap_apply_to_candidate_columns(tmp_path, monkeypatch):
    import tabagent.tabschema.compile as C

    many = BOTH.replace(
        '    return {\n        "intent_overlap"',
        "    out = {f'intent_overlap_w{i}': int(w in ctx['words']) for i, w in enumerate("
        "(candidate['name'] + ' ' + candidate['description']).split())}\n"
        '    return {**out,\n        "intent_overlap"',
    )
    monkeypatch.setattr(C, "CELL_BUDGET", 50)  # every gate on statistics (_precap)
    _, result = discover(
        tmp_path,
        fake(extractor=lambda p, n: tagged(many)),
        max_columns_per_feature=2,
        max_extractor_columns=4,
    )
    ext = result.extractor
    per = [c for c in ext.features if c.startswith("intent_overlap")]
    assert len(per) <= 2 and len(ext.features) <= 4
    capped = [r for c, r in ext.dropped.items() if c.startswith("intent_overlap_w")]
    assert any("max_columns_per_feature" in r for r in capped) and any("null rate" in r for r in capped)
    assert any("ran on statistics over every decision point" in n for n in ext.notes)  # _precap ran


def test_verification_checks_the_candidate_values(tmp_path):
    _, result = discover(tmp_path, fake(), verify=True)
    v = result.extractor.verification
    assert v["determinism"]["candidates"] == 16 * len(TOOLS) and not v["determinism"]["unstable"]
    assert v["outcome"]["changed"] == {}
    tests = result.extractor.tests
    assert tests["candidate_columns"] == ["intent_overlap", "candidate_in_plan"]
    golden = tests["goldens"][0]
    assert len(golden["candidates"]) == 3 and set(golden["candidates"][0]["candidate"]) == set(CANDIDATE_KEYS)
    # a value that depends on the candidates scored before it is nondeterministic, and dropped
    stateful = BOTH.replace(
        "    ctx = _context(trajectory_data)\n",
        "    ctx = _context(trajectory_data)\n    _cache['k'] = _cache['k'] + 1 if 'k' in _cache else 0\n",
    ).replace('"candidate_in_plan": int(', '"calls_before": _cache["k"],\n        "candidate_in_plan": int(')
    props = [*PROPOSALS, spec("calls_before", "numeric")]
    llm = fake(
        extractor=lambda p, n: tagged(stateful),
        analyzer=lambda p, n: json.dumps({"potential_features": props}),
        aggregator=lambda p, n: json.dumps({"filtered_features": props, "rejected_features": []}),
    )
    _, shaky = discover(tmp_path, llm, run_id="r2", verify=True)
    assert shaky.extractor.dropped["calls_before"].startswith("nondeterministic")
    assert "intent_overlap" in shaky.extractor.features


def test_the_auditor_judges_three_candidates_per_point_chosen_without_labels(tmp_path):
    def auditor(p, n):
        names = names_in(p)
        verdict = {
            x: "incorrect" if x.startswith("candidate_in_plan (candidate") else "correct" for x in names
        }
        return json.dumps(
            {"features": [{"feature_name": x, "verdict": v, "reason": "."} for x, v in verdict.items()]}
        )

    llm = fake(auditor=auditor)
    _, result = discover(tmp_path, llm, audit=True, audit_samples=4)
    prompts_ = llm.prompts("auditor")
    assert all("# Additional rules" in p and "intent_overlap (candidate 3)" in p for p in prompts_)
    assert not any("(candidate 4)" in p for p in prompts_)
    audit = json.loads((tmp_path / "runs" / "r1" / "audit.json").read_text())
    rec = audit["features"]["candidate_in_plan"]
    assert rec["per_candidate"] and rec["incorrect"] == 3 * audit["answered"]
    assert result.extractor.dropped["candidate_in_plan"].startswith("OutputAuditor: incorrect on")
    assert "(decision, candidate) pairs" in result.extractor.dropped["candidate_in_plan"]
    assert "intent_overlap" in result.extractor.features
    # each point's candidates come from the seed and the point alone (never its label)
    from tabagent.tabschema.verify import pick_candidates

    cands = [candidate_dict(c) for c in adapter().candidates(None)]
    expected = sorted(
        [c["id"] for c in pick_candidates(cands, config().sample_seed, point["decision_point"], 3)]
        for point in audit["points"]
    )
    shown = sorted(re.findall(r'"id": "(\w+)"', p.split("# Additional rules")[-1]) for p in prompts_)
    assert shown == expected
    assert pick_candidates(cands, 0, "run1#2") == pick_candidates([dict(c) for c in cands], 0, "run1#2")


def test_usefulness_reports_the_candidate_features(tmp_path):
    _, result = discover(tmp_path, fake(), usefulness=True, usefulness_max_decisions=16)
    features = result.report.usefulness["features"]
    assert {"intent_overlap", "candidate_in_plan", "user_intent"} <= set(features)


# --------------------------------------------------------------------------- leakage
def test_candidate_outputs_ignore_labels_later_steps_and_outcome_fields(tmp_path):
    """The per-candidate values of a decision are the same whatever its label, whatever happens after it,
    and with the run's outcome fields added to the prefix."""
    schema, result = discover(tmp_path, fake())
    runner = ExtractorRunner(result.extractor.code)

    def values(trajs, ad):
        dps = [dp for t in trajs for dp in ad.decision_points(t, TARGET)]
        feats, per = extract_rows(runner, result.card, dps, ad)
        return rows_to_table(result.card, dps, feats, ad, per).df

    base = values(runs(), adapter())
    relabeled = values(runs(), adapter(labels_fn=lambda dp, raw: {"play_song"}))
    cols = ["decision_id", CANDIDATE, "intent_overlap", "candidate_in_plan"]
    assert base[cols].equals(relabeled[cols]) and not base["chosen"].equals(relabeled["chosen"])

    def changed_future(i):
        raw = raw_run(i)
        raw["steps"][2:] = [
            {"name": TARGET, "data": "play_song"},
            {"name": "CodeAgent", "calls": ["play_song"]},
            {"name": "PlanControllerAgent", "data": "plan: play the song after all"},
        ]
        raw["score"] = 0.0
        return raw

    future = values(runs(make=changed_future), adapter())
    assert base[cols].equals(future[cols])
    dps = [dp for t in runs() for dp in adapter().decision_points(t, TARGET)]
    cands = [[candidate_dict(c) for c in adapter().candidates(dp)] for dp in dps]
    _, plain = runner.run_items([dp.prefix for dp in dps], cands)
    outcome = [{**dp.prefix, "score": 0.0, "eval": {"passed": False}} for dp in dps]
    _, with_outcome = runner.run_items(outcome, cands)
    assert plain == with_outcome


# --------------------------------------------------------------------------- serving
@pytest.fixture(scope="module")
def head_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cand_head")
    logs = tmp / "logs"
    logs.mkdir()
    for i in range(24):
        (logs / f"run{i}.json").write_text(json.dumps(raw_run(i)))
    select = auto.select

    def relative_on(table, card, **kw):
        # the relative features on (the recipe search keeps them only when strictly better, which a toy
        # decides by chance), so serving is checked to compute them as training did
        recipe, rows = select(table, card, **kw)
        return dataclasses.replace(recipe, relative=True), rows

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(auto, "select", relative_on)
        tabagent.onboard(
            logs,
            TARGET,
            adapter=adapter(),
            llm=fake(),
            config=config(),
            run_dir=tmp / "run",
            n_splits=3,
            out=tmp / "head",
            iterations=40,
        )
    return tmp / "head"


def test_a_saved_head_decides_alike_in_process_in_the_sandboxes_and_served(head_dir):
    from conftest import served

    meta = json.loads((head_dir / "head.json").read_text())
    assert meta["candidate_function"] is True and meta["relative_features"] is True  # the fixture's recipe
    prefix = adapter().decision_points(runs(1)[0], TARGET)[0].prefix
    answers, frames = {}, {}
    for name, load in {
        "in-process": {"trust_extractor": True},
        "subprocess": {"sandbox": "subprocess"},
        "warm": {},
    }.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            head = tabagent.load(head_dir, **load)
        trace = {}
        answers[name] = head.decide(prefix, LIVE, k=2, trace=trace)
        frames[name] = trace["frame"]
        if name == "warm":
            with served(head) as url:
                answers["remote"] = tabagent.serve.RemoteHead(url).decide(prefix, LIVE, k=2)
            per = head.candidate_features_for(prefix, LIVE)
            assert [p["intent_overlap"] for p in per] == [
                overlap(prefix["intent"], c) for c in adapter().candidates(None)
            ]
        head.close()
    first = answers["in-process"]
    for name, a in answers.items():
        assert a["candidates"] == first["candidates"] and a["scores"] == pytest.approx(first["scores"]), name
    assert frames["in-process"].equals(frames["subprocess"]) and frames["warm"].equals(frames["subprocess"])
    assert first["candidates"] == ["book_flight", "search_flights"] and first["deferred"] is False
    # the live per-candidate values are the training rows'
    assert list(frames["warm"]["intent_overlap"]) == [
        overlap(prefix["intent"], c) for c in adapter().candidates(None)
    ]
    # and so are their ranks and gaps within the decision, which decide computes over all its candidates
    head = tabagent.load(head_dir, trust_extractor=True)
    ad = adapter()
    dp = ad.decision_points(runs(1)[0], TARGET)[0]
    feats, per = extract_rows(ExtractorRunner(head.extractor_code), head.card, [dp], ad)
    trained = head.agent.features(rows_to_table(head.card, [dp], feats, ad, per))
    relative = [c for c in trained.columns if c.endswith(("_rank_in_decision", "_gap_to_best"))]
    assert {"intent_overlap_rank_in_decision", "candidate_prior_gap_to_best"} <= set(relative)
    np.testing.assert_allclose(frames["warm"][relative].to_numpy(float), trained[relative].to_numpy(float))
    alone = {}
    head.decide(prefix, LIVE[:1], trace=alone)  # a decision's only candidate is its best
    assert (alone["frame"][[c for c in relative if c.endswith("rank_in_decision")]] == 1.0).all(axis=None)
    assert (alone["frame"][[c for c in relative if c.endswith("gap_to_best")]] == 0.0).all(axis=None)


def test_verify_head_reruns_the_candidate_goldens(head_dir):
    report = tabagent.verify_head(head_dir)
    assert report.passed, str(report)
    names = {c["name"]: c for c in report.checks}
    assert "candidate" not in names["coverage"]["detail"] or names["coverage"]["passed"]


def test_tabsynth_copies_candidate_text_columns_verbatim():
    import pandas as pd

    from tabagent import TabSynth

    X = pd.DataFrame(
        {
            "user_intent": ["book a flight to rome please", "play a song from lima now"],
            "candidate_verb": ["Book the flight now", "Play some music now"],
        }
    )
    syn, y = TabSynth(k=2, text_cols=["user_intent"]).augment(X, [1, 0])
    assert set(syn["candidate_verb"]) <= set(X["candidate_verb"]) and len(syn)
    assert not set(syn["user_intent"]) & set(X["user_intent"])
