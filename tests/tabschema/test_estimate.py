"""The dry-run cost estimate (``tabagent.tabschema.estimate``) lands within 30% of what discovery pays.

Two kinds of fixtures. Recorded: the sizes the estimate reads from the data, and the tokens four real
discovery runs paid per stage, on the public τ²-bench GPT-4.1 logs (next tool call; aggregate counts only, no
trajectory text). Scripted: a fake LLM that behaves as the estimate's ``Profile`` says LLMs do (how many
features they propose and keep, how long they answer, how often code needs a retry), run through the real
pipeline with the audit and the compile step's repair round.
"""

from __future__ import annotations

import json
import re

import pytest
from test_robustness import (
    PROPOSALS,
    TARGET,
    TASKS,
    FakeLLM,
    adapter,
    names_in,
    runs,
    schema_for,
    spec,
    waits,  # noqa: F401 - autouse: backoff is recorded, not slept
)

import tabagent
from tabagent.tabschema.estimate import DataSummary, Profile, estimate, summarize
from tabagent.tabschema.sampling import sample_decision_points

CLAUDE, GPT = "aws/claude-opus-5-5", "azure/gpt-6-sol"
PRICES = {CLAUDE: (3.0, 15.0), GPT: (2.0, 10.0)}  # round prices: the check is on tokens, priced alike
SCHEMA_20 = [
    233,
    233,
    241,
    241,
    233,
    233,
    233,
    233,
    233,
    233,
    241,
    233,
    233,
    233,
    241,
    233,
    241,
    233,
    233,
    233,
]
#: Public τ²-bench runs (Claude Opus 5.5, GPT-6 Sol writing the code; accept=("accept",)): the FeatureAnalyzer
#: prompt of each analyzed point (characters), each point's schema, and the (calls, input, output tokens) paid
#: per stage. The final run's audit predates the 48-column cap of the OutputAuditor, so it is left out.
TAU2 = {
    "n10": {
        "analyzer": [19506, 3061, 5851, 3808, 5201, 11492, 16693, 9169, 9158, 13770],
        "schema": SCHEMA_20[:10],
        "paid": {
            "aggregator": (4, 38432, 33078),
            "analyzer": (10, 41862, 45507),
            "extractor": (25, 138962, 95505),
            "judge_generality": (4, 27744, 34574),
            "judge_impact": (4, 27888, 31870),
            "judge_relevance": (4, 27672, 30835),
            "meta_judge": (4, 102405, 38119),
            "validator": (11, 117665, 32465),
        },
    },
    "n20b": {
        "analyzer": [15394, 16693, 3119, 3645, 16521, 13227, 9169, 17385, 9908, 3367]
        + [23840, 9365, 10154, 25520, 3410, 3348, 5294, 16849, 3749, 10930],
        "schema": [233, 233, 241, 241, 233, 233, 233, 233, 233, 241, 233, 233, 233, 233, 233, 241, 233, 233]
        + [233, 233],
        "paid": {
            "aggregator": (6, 72983, 57762),
            "analyzer": (20, 94375, 90332),
            "extractor": (43, 224075, 159399),
            "judge_generality": (7, 49071, 54374),
            "judge_impact": (7, 49323, 53816),
            "judge_relevance": (7, 48945, 57314),
            "meta_judge": (7, 174923, 65074),
            "validator": (21, 231079, 57857),
        },
    },
    "n20c": {
        "analyzer": [13217, 9169, 2986, 3448, 17886, 10961, 5029, 9297, 9306, 13168]
        + [3550, 2855, 5256, 11060, 13274, 17229, 8816, 18416, 2460, 10112],
        "schema": [233, 233, 241, 241, 233, 233, 233, 233, 233, 233, 241, 241, 233, 233, 233, 233, 233, 233]
        + [241, 233],
        "paid": {
            "aggregator": (7, 75461, 62353),
            "analyzer": (20, 78185, 92657),
            "extractor": (45, 241789, 166802),
            "judge_generality": (8, 53205, 70366),
            "judge_impact": (8, 53493, 60965),
            "judge_relevance": (8, 53061, 65126),
            "meta_judge": (8, 206657, 74449),
            "validator": (21, 225473, 63532),
        },
    },
    "n20": {
        "analyzer": [19506, 3061, 5851, 3808, 5201, 11492, 16693, 9169, 9158, 13770]
        + [3310, 10131, 7197, 19857, 3145, 18991, 2996, 4322, 19040, 5229],
        "schema": SCHEMA_20,
        "paid": {
            "aggregator": (6, 70412, 56672),
            "analyzer": (20, 80270, 88669),
            "extractor": (52, 316172, 201264),
            "judge_generality": (8, 53099, 66672),
            "judge_impact": (8, 53387, 59334),
            "judge_relevance": (8, 52955, 62226),
            "meta_judge": (8, 200467, 73266),
            "validator": (23, 263916, 68928),
        },
    },
}


def cost(paid: dict[str, tuple[int, int, int]]) -> float:
    total = 0.0
    for stage, (_, tin, tout) in paid.items():
        pin, pout = PRICES[GPT if stage == "extractor" else CLAUDE]
        total += (tin * pin + tout * pout) / 1e6
    return total


@pytest.mark.parametrize("run", list(TAU2))
def test_the_estimate_of_recorded_public_runs_is_within_30_percent(run):
    rec = TAU2[run]
    data = DataSummary(rec["analyzer"], rec["schema"], 233, [], 3226)
    est = estimate(data, model=CLAUDE, prices=PRICES, stage_models={"extractor": GPT}, accept=("accept",))
    actual = cost(rec["paid"])
    assert abs(est.usd / actual - 1) <= 0.30, (run, est.usd, actual)
    tin = sum(t for _, t, _ in rec["paid"].values())
    tout = sum(t for _, _, t in rec["paid"].values())
    assert abs(est.input_tokens / tin - 1) <= 0.30 and abs(est.output_tokens / tout - 1) <= 0.30
    analyzer = est.stages["analyzer"]
    assert abs(analyzer.input_tokens / rec["paid"]["analyzer"][1] - 1) <= 0.15  # rendered, not guessed
    assert est.stages["extractor"].model == GPT and est.stages["meta_judge"].model == CLAUDE


def test_long_prompts_are_counted_with_the_model_tokenizer():
    """The analyzer prompts of long logs dominate their cost; Claude's tokenizer makes about 1.5 times the
    tokens of 4 characters per token on them, which the earlier estimate missed."""
    data = DataSummary([400_000] * 20, [500] * 20, 500, [], 800)
    claude = estimate(data, model=CLAUDE).stages["analyzer"].input_tokens
    gpt = estimate(data, model="gpt-5.4").stages["analyzer"].input_tokens
    assert claude == pytest.approx(20 * 400_000 / 2.3, rel=0.01) and gpt < claude / 1.4


# --------------------------------------------------------------------------- scripted runs
P = Profile()


def pad(text: str, tokens: int) -> str:
    """``text`` padded to about ``tokens`` tokens (4 characters each, as the fake counts them)."""
    return text + " " * max(0, tokens * 4 - len(text))


def planned(p: str) -> list[str]:
    return names_in(p)


def profiled_code(names: list[str]) -> str:
    """Code computing every planned feature (and, for every other one, a sub-key: about 1.5 columns each),
    padded with a comment to the profile's code length."""
    lines = ["intent = trajectory['intent']", "features = {"]
    for k, name in enumerate(names):
        lines.append(f"    {name!r}: (len(intent) * {k + 3}) % {k + 5},")
        if k % 2 == 0:
            lines.append(f"    {name + '_n'!r}: intent.count('o') + {k},")
    lines.append("}")
    code = "\n".join(lines) + "\n"
    return code + "# " + "c" * max(0, P.code_chars - len(code)) + "\n"


def profiled_llm() -> FakeLLM:
    """A fake LLM that behaves as ``Profile`` says: its proposals, survivals, answer lengths and retries."""

    def analyzer(p, n):
        task = next(i for i, (t, _) in enumerate(TASKS) if t.split("{")[0] in p)
        shared = [*PROPOSALS, *(spec(f"shared_{k}") for k in range(3))]
        own = [spec(f"task{task}_feature_{k}") for k in range(9)]
        items = [{**s, "description": s["description"] + " " + "d" * 330} for s in shared + own]
        return pad(json.dumps({"potential_features": items}), P.analyzer_out)

    def compiling(p: str) -> bool:  # the compile step plans the features of several analyzed points
        return len(set(re.findall(r"task(\d)_feature", p))) >= 2

    def extractor(p, n):  # the first attempt of each analyzed point fails; the compile step repairs once
        names = planned(p)
        if "CodeValidator found" in p:
            code = profiled_code(names) + "# repaired\n"
        elif "PREVIOUS ATTEMPT FAILED" in p or compiling(p):
            code = profiled_code(names)
        else:
            code = "def broken(:\n"
        return pad(f"<thoughts>{'t' * 400}</thoughts><code>{code}</code>", P.extractor_out)

    def validator(p, n):  # at compile, the first code has one feature flagged
        names = planned(p)
        flag = compiling(p) and "# repaired" not in p
        verdicts = [
            {"feature_name": x, "verdict": "plan_mismatch" if flag and k == 0 else "valid", "evidence": "e"}
            for k, x in enumerate(names)
        ]
        text = json.dumps({"features": verdicts})
        return pad(text, max(P.validator_out, len(text) // 4 + 10))

    def aggregator(p, n):
        names = planned(p)
        keep = max(1, round(P.kept_share * len(names)))
        rich = lambda x: {**spec(x), "description": "a" * 330}  # noqa: E731
        out = {"filtered_features": [rich(x) for x in names[:keep]], "rejected_features": []}
        out["rejected_features"] = [rich(x) for x in names[keep:]]
        return pad(json.dumps(out), P.aggregator_out_per_proposal * len(names))

    def judge(p, n):
        items = [
            {"feature_name": x, "score": 4, "confidence": 4, "assessment": "s" * 900, "key_factors": ["k"]}
            for x in planned(p)
        ]
        return pad(json.dumps({"features": items}), P.judge_out_per_feature * len(items))

    def meta(p, n):
        shares = dict(P.decision_share)
        names = planned(p)
        cut1 = round(shares["accept"] * len(names))
        cut2 = cut1 + round(shares["conditional"] * len(names))
        decision = ["accept"] * cut1 + ["conditional"] * (cut2 - cut1) + ["reject"] * (len(names) - cut2)
        items = [{"feature_name": x, "final_decision": d} for x, d in zip(names, decision, strict=True)]
        return pad(json.dumps(items), P.meta_out_per_feature * len(items))

    def auditor(p, n):
        names = re.findall(r'"feature_name": "([^"]+)"', p)
        items = [{"feature_name": x, "verdict": "correct", "reason": "r"} for x in names]
        return pad(json.dumps({"features": items}), P.auditor_out_per_column * len(items))

    return FakeLLM(
        analyzer=analyzer,
        extractor=extractor,
        validator=validator,
        aggregator=aggregator,
        judge=judge,
        meta_judge=meta,
        auditor=auditor,
    )


def test_a_scripted_run_lands_within_30_percent_of_its_estimate(tmp_path):
    llm = profiled_llm()
    schema = schema_for(tmp_path, llm, n_analyze=4, prices={"fake": (1.0, 1.0)})
    result = schema.discover(runs(), TARGET)
    assert any("repair round" in n for n in result.extractor.notes)  # the repair round ran
    cfg = schema.config
    ad = adapter()
    dps = [dp for t in runs() for dp in ad.decision_points(t, TARGET)]
    sample = sample_decision_points(dps, cfg.n_analyze, cfg.sample_seed, ad.labels, cfg.sample_strategy)
    seen = {dp.id for dp in sample}
    audit = sample_decision_points([dp for dp in dps if dp.id not in seen], 12, cfg.sample_seed, ad.labels)
    est = estimate(
        summarize(sample, dps, TARGET, cfg.max_prompt_chars, audit),
        model="fake",
        prices={"fake": (1.0, 1.0)},
        accept=cfg.accept,
        chars_per_token_of=lambda model: 4.0,  # the fake counts 4 characters per token
    )
    actual = schema.meter.total_usd
    assert abs(est.usd / actual - 1) <= 0.30, (
        est.usd,
        actual,
        est.to_dict()["stages"],
        schema.meter.summary(),
    )


def test_the_dry_run_breaks_the_estimate_down_by_stage_and_counts_the_audit():
    kw = dict(adapter=adapter(), dry_run=True, prices={"gpt-5.4": (1.0, 8.0)})
    full = tabagent.onboard(runs(), TARGET, **kw)
    assert set(full.stages) >= {"analyzer", "extractor", "validator", "aggregator", "meta_judge", "auditor"}
    assert full.est_cost_usd == pytest.approx(sum(s["usd"] for s in full.stages.values()))
    text = str(full)
    assert "auditor" in text and "meta_judge" in text and "est. cost $" in text
    from tabagent.tabschema import TabSchemaConfig

    quiet = tabagent.onboard(
        runs(),
        TARGET,
        adapter=adapter(),
        dry_run=True,
        config=TabSchemaConfig(audit=False, prices=kw["prices"]),
    )
    assert "auditor" not in quiet.stages and quiet.est_cost_usd < full.est_cost_usd


def test_the_candidate_sections_and_the_repair_round_are_counted():
    """A decision with candidates adds the per-candidate sections (and three example candidates) to the
    CodeExtractor and CodeValidator prompts; the compile step's repair round is one more call of each, unless
    ``compile_repair=False``."""
    plain = DataSummary([20_000] * 4, [3_000] * 4, 4_000, [], 200)
    with_cands = DataSummary([20_000] * 4, [3_000] * 4, 4_000, [], 200, candidate_chars=1_500)
    base, more = estimate(plain, model=CLAUDE), estimate(with_cands, model=CLAUDE)
    for stage in ("extractor", "validator"):
        assert more.stages[stage].input_tokens > base.stages[stage].input_tokens, stage
        assert more.stages[stage].calls == base.stages[stage].calls, stage
    lean = estimate(plain, model=CLAUDE, compile_repair=False)
    for stage in ("extractor", "validator"):
        assert lean.stages[stage].calls == pytest.approx(base.stages[stage].calls - P.compile_repair_calls)
    dps = [dp for t in runs()[:2] for dp in adapter().decision_points(t, TARGET)]
    data = summarize(dps[:1], dps, TARGET, candidates_of=adapter().candidates)
    assert data.candidate_chars > 0 and summarize(dps[:1], dps, TARGET).candidate_chars == 0
