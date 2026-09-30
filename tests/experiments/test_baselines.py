"""Learned baselines (experiments/baselines): DSR, LLM prompt construction, fold hygiene, the CLI.

Everything runs on CPU in seconds with fabricated tables and tiny randomly initialized models built in
memory (no download). LLM prompts are checked through the dry-run recorder.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import torch  # noqa: E402

from experiments.baselines import dsr as D  # noqa: E402
from experiments.baselines import llm as L  # noqa: E402
from experiments.baselines import presets as P  # noqa: E402
from experiments.baselines.common import fold_array, text_view  # noqa: E402
from experiments.baselines.lora import LoRALinear, apply_lora  # noqa: E402
from tabagent.table import DecisionSpec, DecisionTable  # noqa: E402

# --------------------------------------------------------------------------- fabricated data
TOOLS = {
    "search_flights": "search flights between two airports",
    "book_flight": "book a seat on a flight",
    "get_weather": "get the weather forecast for a city",
    "send_email": "send an email message",
    "create_note": "create a note with text",
    "play_song": "play a song in the music app",
}
INTENTS = [
    (("search_flights", "book_flight"), "find a flight to {c} and book a seat"),
    (("get_weather",), "what is the weather in {c}"),
    (("send_email",), "send an email about {c}"),
    (("create_note",), "make a note about {c}"),
    (("play_song",), "play music from {c}"),
]
CITIES = ["paris", "rome", "tokyo", "lima", "oslo", "cairo"]


def ranking_table(n_tasks: int = 20, seed: int = 0) -> DecisionTable:
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        used, tpl = INTENTS[t % len(INTENTS)]
        text = tpl.format(c=CITIES[rng.integers(len(CITIES))])
        for tool, desc in TOOLS.items():
            rows.append(
                {"task": f"t{t:02d}", "intent": text, "tool": tool, "desc": desc, "used": int(tool in used)}
            )
    return DecisionTable(
        pd.DataFrame(rows), DecisionSpec("ranking", group="task", candidate="tool", label="used")
    )


def folds_of(table: DecisionTable, k: int = 4) -> dict:
    return {g: i % k for i, g in enumerate(table.groups)}


def op_table() -> DecisionTable:
    rows = []
    for t in range(12):
        for s, (op, tag) in enumerate([("CLICK", "button"), ("TYPE", "input"), ("SELECT", "select")]):
            rows.append(
                {
                    "task_uid": f"t{t:02d}",
                    "step_index": s,
                    "instruction": f"book a table number {t}",
                    "target_tag": tag,
                    "target_type": "text" if op == "TYPE" else "",
                    "history_action_reprs": "",
                    "op": op,
                    "eval_original_op": op,
                }
            )
    spec = DecisionSpec("single", group="task_uid", label="op", decision=("task_uid", "step_index"))
    return DecisionTable(pd.DataFrame(rows), spec)


def gate_table() -> DecisionTable:
    rows = []
    for t in range(6):
        for mode in ("without", "with"):
            rows.append(
                {
                    "task_uid": f"t{t}",
                    "task_text": f"task number {t}",
                    "supplied_skills_text": "use the pdf skill" if mode == "with" else "",
                    "mode": mode,
                    "passed": int((mode == "with") == (t % 2 == 0)),
                    "reward": float((mode == "with") == (t % 2 == 0)),
                }
            )
    spec = DecisionSpec("policy", group="task_uid", label="passed", candidate="mode", reward="reward")
    return DecisionTable(pd.DataFrame(rows), spec)


# --------------------------------------------------------------------------- tiny in-memory models
def word_tokenizer(texts, chat: bool = False):
    """A word-level tokenizer over the fabricated vocabulary (optionally with a chat template)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = {w for t in texts for w in t.lower().replace(":", " : ").split()}
    words |= {str(i) for i in range(25)} | {"yes", "no", ",", ":", "?", "query", "passage", "none"}
    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "<s>", "</s>", *sorted(words)])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
        model_input_names=["input_ids", "attention_mask"],
    )
    if chat:
        tok.chat_template = (
            "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}assistant : {% endif %}"
        )
    return tok


ALL_TEXT = [*TOOLS, *TOOLS.values(), *(tpl.format(c=c) for _, tpl in INTENTS for c in CITIES)]


@pytest.fixture(scope="module")
def tiny_loader():
    from transformers import BertConfig, BertModel

    tok = word_tokenizer([t.replace("_", " ") for t in ALL_TEXT] + list(TOOLS))
    cfg = BertConfig(
        vocab_size=len(tok),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )

    def load():
        torch.manual_seed(0)
        return tok, BertModel(cfg)

    return load


FAST = D.DSRConfig(epochs=2, batch_size=8, lr=5e-3, max_length=32, encode_batch_size=16)


# --------------------------------------------------------------------------- helpers
def test_text_view_caps_columns_then_the_whole_text():
    df = pd.DataFrame({"a": ["alpha", None], "b": ["0123456789", "xyz"]})
    assert list(text_view(df, ["a", ("b", 4)])) == ["alpha 0123", " xyz"]
    assert list(text_view(df, ["a", "b"], max_chars=7)) == ["alpha 0", " xyz"]
    assert list(text_view(df, "a")) == ["alpha", ""]


def test_fold_array_rejects_groups_without_a_fold():
    t = ranking_table(4)
    assert set(fold_array(t, folds_of(t, 2))) == {0, 1}
    with pytest.raises(ValueError, match="without a fold"):
        fold_array(t, {"t00": 0})


def test_lora_starts_as_identity_and_trains_only_adapters():
    torch.manual_seed(0)
    net = torch.nn.Sequential()
    net.add_module("query", torch.nn.Linear(4, 4))
    net.add_module("other", torch.nn.Linear(4, 4))
    x = torch.randn(3, 4)
    before = net(x)
    params = apply_lora(net, ["query"], r=2, alpha=4, dropout=0.0)
    assert isinstance(net.query, LoRALinear) and not isinstance(net.other, LoRALinear)
    assert torch.allclose(net(x), before)  # B = 0 at init
    assert {id(p) for p in params} == {id(net.query.lora_A), id(net.query.lora_B)}
    with pytest.raises(ValueError, match="no nn.Linear"):
        apply_lora(torch.nn.Sequential(torch.nn.ReLU()), ["query"])


# --------------------------------------------------------------------------- DSR
def test_dsr_zero_shot_is_the_row_inner_product(tiny_loader):
    t = ranking_table(6)
    s = D.dsr_scores(t, ["intent"], ["tool", "desc"], config=FAST, loader=tiny_loader, device="cpu")
    tok, model = tiny_loader()
    q = D.encode(model.eval(), tok, text_view(t.df, ["intent"], 256), D.QUERY, FAST, "cpu")
    d = D.encode(model, tok, text_view(t.df, ["tool", "desc"], 256), D.PASSAGE, FAST, "cpu")
    row = 7
    expect = q[t.df["intent"][row]] @ d[f"{t.df['tool'][row]} {t.df['desc'][row]}"]
    assert s.shape == (len(t),) and np.isfinite(s).all()
    assert s[row] == pytest.approx(float(expect), abs=1e-5)


def test_dsr_finetune_needs_folds(tiny_loader):
    with pytest.raises(ValueError, match="folds"):
        D.dsr_scores(ranking_table(4), ["intent"], ["tool"], finetune=True, loader=tiny_loader, device="cpu")


def test_dsr_finetune_trains_on_training_folds_only(tiny_loader, monkeypatch):
    t = ranking_table(12)
    folds = folds_of(t, 3)
    seen = []
    real = D.contrastive_finetune

    def spy(model, tok, pairs, relevant, cfg, seed, device):
        seen.append({p.decision for p in pairs} | set(relevant))
        return real(model, tok, pairs, relevant, cfg, seed, device)

    monkeypatch.setattr(D, "contrastive_finetune", spy)
    s = D.dsr_scores(
        t, ["intent"], ["tool", "desc"], folds, True, config=FAST, loader=tiny_loader, device="cpu"
    )
    assert np.isfinite(s).all() and len(seen) == 3
    for f, decisions in enumerate(seen):
        test_groups = {g for g, k in folds.items() if k == f}
        assert decisions and not decisions & test_groups


def test_contrastive_finetune_lowers_the_loss_and_moves_only_adapters(tiny_loader):
    t = ranking_table(20)
    q, d = text_view(t.df, ["intent"]), text_view(t.df, ["tool", "desc"])
    zs = D.dsr_scores(t, ["intent"], ["tool", "desc"], config=FAST, loader=tiny_loader, device="cpu")
    cfg = replace(FAST, hard_negatives=1, mask_false_negatives=True, epochs=6, lr=2e-2)
    pairs, rel = D.training_pairs(q, d, t.labels, t.decision_ids.to_numpy(), np.ones(len(t), bool), zs, cfg)
    assert all(len(p.hard) > 0 for p in pairs)
    tok, model = tiny_loader()
    frozen = model.embeddings.word_embeddings.weight.detach().clone()
    losses = D.contrastive_finetune(model, tok, pairs, rel, cfg, seed=0, device="cpu")
    assert np.mean(losses[-3:]) < np.mean(losses[:3]) - 0.3
    assert torch.equal(model.embeddings.word_embeddings.weight, frozen)
    lora_b = [m.lora_B for m in model.modules() if isinstance(m, LoRALinear)]
    assert lora_b and all(b.abs().sum() > 0 for b in lora_b)


def test_recipes_in_batch_is_the_default():
    assert D.DSRConfig() == D.RECIPES["in-batch"]
    assert (D.DSRConfig().hard_negatives, D.DSRConfig().mask_false_negatives) == (0, False)
    hard = D.RECIPES["hard-negatives"]
    assert (hard.hard_negatives, hard.mask_false_negatives) == (1, True)


def test_in_batch_recipe_mines_no_hard_negatives():
    q = np.array(["q1", "q1", "q2", "q2"], dtype=object)
    d = np.array(["a", "b", "a", "b"], dtype=object)
    pairs, _ = D.training_pairs(q, d, np.array([1, 0, 0, 1]), q, np.ones(4, bool), np.ones(4), D.DSRConfig())
    assert [p.hard for p in pairs] == [(), ()]


def test_hard_negative_pools_come_from_the_decision_and_the_training_rows():
    q = np.array(["q1"] * 4 + ["q2"] * 4, dtype=object)
    d = np.array(["a", "b", "c", "a", "a", "b", "c", "d"], dtype=object)  # row 3 repeats q1's positive text
    y = np.array([1, 0, 0, 0, 0, 1, 0, 0])
    dec = np.array(["q1"] * 4 + ["q2"] * 4, dtype=object)
    zs = np.array([0.9, 0.1, 0.5, 0.95, 0.9, 0.2, 0.3, 0.8])
    cfg = replace(D.RECIPES["hard-negatives"], hard_pool=2)
    pairs, rel = D.training_pairs(q, d, y, dec, np.ones(8, bool), zs, cfg)
    by = {p.decision: p for p in pairs}
    assert by["q1"].hard == ("c", "b")  # "a" is q1's positive text, never its negative
    assert by["q2"].hard == ("a", "d")
    assert rel == {"q1": {"a"}, "q2": {"b"}}
    pairs, rel = D.training_pairs(q, d, y, dec, np.arange(8) < 4, zs, cfg)  # q2 held out
    assert [p.decision for p in pairs] == ["q1"] and set(rel) == {"q1"}


def test_false_negative_mask():
    batch = [D.Pair("x", "a", "d1"), D.Pair("y", "b", "d1"), D.Pair("z", "a", "d2")]
    passages = ["a", "b", "a", "c"]
    rel = {"d1": {"a", "b"}, "d2": {"a"}}
    assert D.false_negatives(batch, passages, rel) == [
        [False, True, True, False],
        [True, False, True, False],
        [True, False, False, False],
    ]


def test_synth_pairs_paraphrase_training_contexts_only():
    t = ranking_table(8)
    train = fold_array(t, folds_of(t, 2)) == 0
    q, d = text_view(t.df, ["intent"]), text_view(t.df, ["tool"])
    pairs, _ = D.training_pairs(q, d, t.labels, t.decision_ids.to_numpy(), train, None, D.DSRConfig())
    syn = D.synth_pairs(t, ["intent"], pairs, train, 2, D.DSRConfig())
    train_decisions = set(t.decision_ids[train])
    assert syn and {p.decision for p in syn} <= train_decisions
    assert all(p.query not in set(q) for p in syn)
    assert {p.positive for p in syn} <= {p.positive for p in pairs}


def test_expand_classes_and_argmax_roundtrip():
    t = op_table()
    x = D.expand_classes(t, ("CLICK", "TYPE", "SELECT"))
    assert len(x) == 3 * len(t) and x.spec.kind == "ranking"
    assert x.df.groupby(x.decision_ids.to_numpy())["op"].sum().eq(1).all()
    oracle = x.labels.astype(float)
    assert (D.argmax_class(x, oracle, t) == t.df["op"].to_numpy()).all()
    with pytest.raises(ValueError):
        D.expand_classes(ranking_table(2))


# --------------------------------------------------------------------------- LLM prompts (pure)
LISTWISE = L.ListwiseTask(
    task_cols=("intent",),
    cand_cols=("tool",),
    instruction="Select which tools are relevant. Output ONLY numbers.\n\n",
    task_chars=10,
    cand_chars=6,
)


def test_listwise_prompt_layout():
    msgs = L.listwise_messages(
        LISTWISE, "find a flight to rome", ["search_flights", "book"], [("hi", ["a", "b"], [0, 1])]
    )
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == (
        "Select which tools are relevant. Output ONLY numbers.\n\nTASK: hi\nCANDIDATES:\n1. a\n2. b\n\n"
        "Relevant numbers:"
    )
    assert msgs[1]["content"] == "2"
    assert msgs[2]["content"] == "TASK: find a fli\nCANDIDATES:\n1. search\n2. book\n\nRelevant numbers:"
    assert L.listwise_target([0, 0]) == "none"
    assert L.listwise_target([1] * 20, 15) == ", ".join(str(i) for i in range(1, 16))


def test_parse_ranking_lists_parsed_numbers_first_then_a_seeded_order():
    def reference(text, y):  # parsed 1-based numbers in order, deduplicated, then the rest shuffled (seed 0)
        seen, order = set(), []
        for k in [int(x) for x in __import__("re").findall(r"\d+", text)]:
            if 1 <= k <= len(y) and k not in seen:
                seen.add(k)
                order.append(k - 1)
        rest = [i for i in range(len(y)) if i not in order]
        np.random.default_rng(0).shuffle(rest)
        return order + rest

    for text in ["3, 1, 3, 99, 0", "", "none", "7 2 5"]:
        assert list(L.parse_ranking(text, 8)) == reference(text, list(range(8)))
    s = L.order_scores(np.array([2, 0, 1]))
    assert list(np.argsort(-s)) == [2, 0, 1]


def test_parse_class_and_yes_no():
    task = P.OPERATION_CLASSIFY
    assert L.parse_class(task, " Select") == "SELECT"
    assert L.parse_class(task, "type it") == "TYPE"
    assert L.parse_class(task, "???") == "CLICK"
    assert L.parse_yes_no("No... on reflection, yes.") is True
    assert L.parse_yes_no("yes? no") is False
    assert L.parse_yes_no("maybe") is None


# --------------------------------------------------------------------------- LLM methods in dry-run mode
@pytest.fixture
def dry():
    return L.DryRunLM("not-a-real-model", tok=None)


def test_fewshot_listwise_holds_out_the_demonstrations(dry):
    t = ranking_table(10)
    s = L.fewshot_listwise(t, LISTWISE, dry, log=None)
    shots = {"t00", "t01"}  # "first": the first two decisions by id
    held = t.df["task"].isin(shots).to_numpy()
    assert np.isnan(s[held]).all() and np.isfinite(s[~held]).all()
    prompts = [r["prompt"] for r in dry.records]
    assert len(prompts) == 8 and all(p.count("TASK:") == 3 for p in prompts)
    assert not dry.templated and prompts[0].endswith("<|assistant|>\n")


def test_random_shots_come_from_distinct_groups():
    items = [L.Decision(str(i), i // 3, np.array([i]), "", [], np.array([1])) for i in range(9)]
    shots = L.select_shots(items, "random", 2, seed=1)
    assert len({s.group for s in shots}) == 2


def test_sft_listwise_trains_on_training_folds_only(dry):
    t = ranking_table(12)
    folds = folds_of(t, 3)
    task = L.ListwiseTask(**{**LISTWISE.__dict__, "task_chars": 200})
    t.df["intent"] = t.df["intent"] + " " + t.df["task"]  # unique task texts
    L.sft_listwise(t, task, dry, folds, log=None)
    rec = pd.DataFrame(dry.records)
    text_fold = {r.intent: folds[r.task] for r in t.df.itertuples()}
    for r in rec.itertuples():
        task_text = r.prompt.split("TASK: ")[1].split("\n")[0]
        if r.kind == "train":
            assert r.target.endswith(dry.eos) and text_fold[task_text] != r.fold
        else:
            assert text_fold[task_text] == r.fold
    assert (rec["kind"] == "generate").sum() == 12


def test_cross_encoder_fits_on_training_folds_only(dry):
    t = ranking_table(8)
    folds = folds_of(t, 2)
    ce = L.CrossEncoderTask(("intent",), ("tool",), "T: {task} C: {cand}", task_chars=50, cand_chars=20)
    s = L.cross_encoder_scores(t, ce, dry, folds, log=None)
    rec = pd.DataFrame(dry.records)
    assert np.isfinite(s).all()
    assert (rec["kind"] == "train").sum() == len(t) and (rec["kind"] == "score").sum() == len(t)
    fold = fold_array(t, folds)
    for f in (0, 1):
        trained = set(rec[(rec.kind == "train") & (rec.fold == f)]["prompt"])
        scored = {f"T: {a} C: {b}" for a, b in zip(t.df.intent[fold == f], t.df.tool[fold == f], strict=True)}
        assert trained.isdisjoint(scored - trained) and len(trained) > 0


def test_fewshot_classify_draws_demonstrations_from_training_folds(dry):
    t = op_table()
    folds = {g: i % 3 for i, g in enumerate(t.groups)}
    preds = L.fewshot_classify(t, P.OPERATION_CLASSIFY, dry, folds, log=None)
    assert set(preds) == {"CLICK"}  # empty replies fall back to CLICK
    fold_of_task = {int(g[1:]): k for g, k in folds.items()}
    first_of_fold = {}
    for r in dry.records:
        first_of_fold.setdefault(r["fold"], r["prompt"])
    for f, prompt in first_of_fold.items():
        turns = prompt.split("<|user|>\n")[1:]
        assert len(turns) == 7 and turns[0].startswith("Classify the web UI action.")
        demos = [int(x.split("number ")[1].split("\n")[0]) for x in turns[:-1]]
        assert all(fold_of_task[d] != f for d in demos)
        assert [x.split("<|assistant|>\n")[1].strip() for x in turns[:-1]] == ["click"] * 2 + ["type"] * 2 + [
            "select"
        ] * 2


def test_gate_scores_the_chosen_arm_and_excludes_undecidable():
    class Scripted:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt, role=""):
            self.prompts.append(prompt)
            reply = {"task number 0": "Yes", "task number 1": "I cannot tell"}
            text = next((v for k, v in reply.items() if k in prompt), "Let me think. No")
            return type("Reply", (), {"text": text})()

    t = gate_table()
    client = Scripted()
    task = L.GateTask(**{**P.GATE.__dict__, "retries": 2})
    s = L.fewshot_gate(t, task, client, log=None)
    by = t.df.assign(s=s).set_index(["task_uid", "mode"])["s"]
    assert by["t0", "with"] == 1 and by["t0", "without"] == 0
    assert np.isnan(by["t1", "with"]) and np.isnan(by["t1", "without"])
    assert by["t2", "with"] == 0 and by["t2", "without"] == 1
    assert all("SKILL HINTS: use the pdf skill" in p for p in client.prompts)


def test_presets_cover_the_paper_rows():
    for name in ["skillsbench/skill-select", "mind2web/element-150", "mind2web/operation", "tau2/airline"]:
        p = P.PRESETS[name]
        assert p.query and p.doc and p.llm is not None and p.paper and p.source.startswith("Table 1")
    assert P.PRESETS["skillsbench/skill-gating"].query is None
    # Table 1 values (percent / 100)
    assert P.PRESETS["skillsbench/skill-select"].paper == {
        "dsr-zs": 0.51,
        "dsr-ft": 0.74,
        "fewshot": 0.33,
        "ft-slm": 0.52,
    }
    assert [
        P.PRESETS[f"tau2/{d}"].paper["dsr-ft"] for d in ("airline", "retail", "telecom", "all-domains")
    ] == [
        0.44,
        0.42,
        0.28,
        0.42,
    ]
    op = P.operation_context(
        {"instruction": "x" * 400, "target_tag": "a", "target_type": None, "op": "CLICK"}
    )
    assert "CLICK" not in op and op.count("x") == 280


# --------------------------------------------------------------------------- Hugging Face backends (tiny)
@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    path = tmp_path_factory.mktemp("tiny-llama")
    tok = word_tokenizer([*ALL_TEXT, "user assistant task candidates relevant numbers target ?"], chat=True)
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_hf_backends_run_end_to_end_on_a_tiny_model(tiny_llama):
    t = ranking_table(6)
    folds = folds_of(t, 2)
    lm = L.HFCausalLM(tiny_llama, device="cpu")
    task = L.ListwiseTask(**{**LISTWISE.__dict__, "max_new_tokens": 4, "sft_epochs": 1, "max_input": 256})
    assert np.isfinite(L.fewshot_listwise(t, task, lm, log=None)[12:]).all()
    assert np.isfinite(L.sft_listwise(t, task, lm, folds, log=None)).all()
    pw = L.PointwiseTask(("intent",), ("tool",), "task {task} target {cand} ?", max_length=64)
    p = L.fewshot_pointwise(t, pw, lm, log=None)
    assert ((p > 0) & (p < 1)).all()
    ce_task = L.CrossEncoderTask(("intent",), ("tool",), "task {task} target {cand}", max_length=32, epochs=1)
    ce = L.HFCrossEncoder(tiny_llama, device="cpu")
    s = L.cross_encoder_scores(t, ce_task, ce, folds, log=None)
    assert ((s > 0) & (s < 1)).all()


# --------------------------------------------------------------------------- CLI
@pytest.fixture
def toy_dataset(monkeypatch, tiny_loader):
    from experiments.baselines import run as R

    t = ranking_table(12)
    monkeypatch.setattr(R.datasets, "load", lambda name, **kw: t)
    monkeypatch.setattr(R.datasets, "folds", lambda name: folds_of(t, 3))
    preset = P.Preset(
        query=("intent",),
        doc=("tool", "desc"),
        metrics=("p@r", "recall@2"),
        dsr=FAST,
        llm=LISTWISE,
        paper={"dsr-zs": 0.5},
    )
    monkeypatch.setitem(R.PRESETS, "toy/tools", preset)
    monkeypatch.setattr(D, "e5_loader", lambda model_id: tiny_loader)
    return R, t


def test_cli_writes_per_decision_vectors(toy_dataset, tmp_path):
    R, t = toy_dataset
    out = tmp_path / "zs.json"
    R.main(["--dataset", "toy/tools", "--method", "dsr-zs", "--device", "cpu", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["paper"] == 0.5 and res["seeds"] == [None]
    assert len(res["per_decision"]["p@r"]) == 12 and set(res["decision_group"]) == set(
        res["per_decision"]["p@r"]
    )
    assert res["summary"]["p@r"]["value"] == pytest.approx(np.mean(list(res["per_decision"]["p@r"].values())))

    out = tmp_path / "ft.json"
    R.main(
        [
            "--dataset",
            "toy/tools",
            "--method",
            "dsr-ft",
            "--seeds",
            "0",
            "1",
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
    )
    res = json.loads(out.read_text())
    assert res["recipe"] == "in-batch" and set(res["per_seed"]["p@r"]) == {"0", "1"}
    assert res["config"]["hard_negatives"] == 0 and not res["config"]["mask_false_negatives"]

    out = tmp_path / "ft-hard.json"
    R.main(["--dataset", "toy/tools", "--method", "dsr-ft", "--recipe", "hard-negatives", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["recipe"] == "hard-negatives" and res["config"]["hard_negatives"] == 1


def test_cli_dry_run_reports_prompts(toy_dataset, tmp_path):
    R, _ = toy_dataset
    out, prompts = tmp_path / "dry.json", tmp_path / "prompts.jsonl"
    R.main(
        [
            "--dataset",
            "toy/tools",
            "--method",
            "ft-slm",
            "--dry-run",
            "--out",
            str(out),
            "--prompts-out",
            str(prompts),
        ]
    )
    res = json.loads(out.read_text())["dry_run"]
    assert res["by_kind"]["train"]["n"] == 2 * 12 and res["by_kind"]["generate"]["n"] == 12
    assert len(prompts.read_text().splitlines()) == res["n_prompts"]
    with pytest.raises(SystemExit):
        R.main(["--dataset", "toy/tools", "--method", "dsr-zs", "--dry-run"])
