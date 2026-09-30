"""Per-dataset configuration of the learned baselines: which text is the query and which the candidate, the
metrics, the prompt shape, and the paper's Table 1 values for reference.

``source`` names the paper result a preset or prompt reproduces. ``reconstructed`` marks a configuration
written for this release without the settings of the paper's run, so its value is not expected to match
Table 1 exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from .common import _cell
from .dsr import DSRConfig
from .llm import ClassifyTask, CrossEncoderTask, GateTask, ListwiseTask, LLMTask, PointwiseTask


@dataclass(frozen=True)
class Preset:
    query: tuple | None  # DSR query text view (None: DSR not defined for this decision)
    doc: tuple | None  # DSR passage text view
    metrics: tuple[str, ...]
    dsr: DSRConfig = field(default_factory=DSRConfig)
    dsr_seed: int = 0  # default DSR-FT seed (the training permutation stream)
    classes: tuple[str, ...] | None = None  # single-class decisions: DSR ranks these class passages
    class_text: Mapping | None = None  # passage per class (default: the lower-cased class name)
    llm: LLMTask | None = None
    ce: CrossEncoderTask | None = None  # the LoRA cross-encoder (``ft-ce``)
    slice_col: str | None = None  # also report the metrics per value of this column
    paper: Mapping = field(default_factory=dict)  # Table 1 values, {method: value or {slice: value}}
    source: str = ""


# --------------------------------------------------------------------------- SkillsBench
SKILL_LISTWISE = ListwiseTask(
    task_cols=("task_text",),
    cand_cols=("skill_name",),
    instruction=(
        "Select which skills are relevant to the task. "
        "Output ONLY the relevant candidate numbers, most relevant first, comma-separated.\n\n"
    ),
    task_chars=500,
    cand_chars=80,
    shots="first",
    max_input=8192,
    max_new_tokens=60,
    sft_epochs=3,
    source="Table 1, skill selection: Few-shot and FT-SLM; App. F, Fig. 8b (SLM family)",
)
SKILL_CE = CrossEncoderTask(
    task_cols=("task_text",),
    cand_cols=("skill_name",),
    template="Task: {task}\nCandidate: {cand}\nRelevant?",
    task_chars=400,
    cand_chars=100,
    max_length=224,
    source="ft-ce option (not a Table 1 column)",
)

SKILL_SELECT = Preset(
    query=("task_text",),
    doc=("skill_name", "skill_desc"),
    metrics=("p@r", "recall@9"),
    dsr=DSRConfig(
        max_chars=None, max_length=512, epochs=3
    ),  # full task and skill texts, as in the paper's runs
    llm=SKILL_LISTWISE,
    ce=SKILL_CE,
    paper={"dsr-zs": 0.51, "dsr-ft": 0.74, "fewshot": 0.33, "ft-slm": 0.52},
    source="Table 1, skill selection row (DSR-ZS, DSR-FT, Few-shot, FT-SLM)",
)

GATE = GateTask(
    task_col="task_text",
    bundle_col="supplied_skills_text",
    arm_col="mode",
    on_arm="with",
    template=(
        "A task will be attempted by an agent. Optional skill hints are available.\n"
        "TASK: {task}\nSKILL HINTS: {bundle}\n"
        "Would activating these skill hints help the agent complete the task correctly? "
        "End your answer with the single word Yes or No."
    ),
    source="Table 1, skill gating: Few-shot (GPT-5.4)",
)
SKILL_GATING = Preset(
    query=None,
    doc=None,
    metrics=("route@1",),
    llm=GATE,
    ce=CrossEncoderTask(
        task_cols=(("task_text", 400), ("supplied_skills_text", 300)),
        cand_cols=("mode",),
        template="Task and skill hints: {task}\nArm: {cand} skills\nWill this arm succeed?",
        task_chars=800,
        cand_chars=20,
        max_length=320,
        source="reconstructed (Table 1, skill gating: FT-SLM)",
    ),
    paper={"fewshot": 0.43, "ft-slm": 0.43},
    source="Table 1, skill gating row (Few-shot = GPT-5.4 through an OpenAI-compatible endpoint)",
)

# --------------------------------------------------------------------------- Mind2Web
ELEMENT_CE = CrossEncoderTask(
    task_cols=("instruction",),
    cand_cols=("elem_attr_text",),
    template="Instruction: {task}\nElement: {cand}\nTarget?",
    task_chars=200,
    cand_chars=100,
    max_length=160,
    epochs=2,
    batch_size=16,
    source="Table 1, Mind2Web element selection: FT-SLM",
)
ELEMENT_POINTWISE = PointwiseTask(
    task_cols=("instruction",),
    cand_cols=("elem_attr_text",),
    yes_template=(
        "Instruction: {task}\nElement: {cand}\nIs this element the target of the action? Answer yes or no."
    ),
    task_chars=200,
    cand_chars=100,
    max_length=512,
    cross_encoder=ELEMENT_CE,
    source="Table 1, Mind2Web element selection: Few-shot",
)
ELEMENT = Preset(
    query=("instruction",),
    doc=(("elem_attr_text", 200),),
    metrics=("recall@3", "recall@5", "recall@1"),
    dsr=DSRConfig(max_chars=256, max_length=512, epochs=2),
    llm=ELEMENT_POINTWISE,
    ce=ELEMENT_CE,
    slice_col="domain",
    paper={
        "dsr-zs": {"Entertainment": 0.11, "Shopping": 0.13, "Travel": 0.12},
        "dsr-ft": {"Entertainment": 0.15, "Shopping": 0.18, "Travel": 0.23},
        "fewshot": {"Entertainment": 0.15, "Shopping": 0.14, "Travel": 0.20},
        "ft-slm": {"Entertainment": 0.31, "Shopping": 0.48, "Travel": 0.41},
    },
    source="Table 1, Mind2Web element selection rows (DSR-ZS, DSR-FT, Few-shot, FT-SLM)",
)


def operation_context(r: Mapping) -> str:
    """The step context of the operation prompt; the labels' eval_* columns are never shown."""
    return (
        f"Task: {_cell(r.get('instruction'))[:280]}\n"
        f"Target element: <{_cell(r.get('target_tag'))}> type={_cell(r.get('target_type'))}\n"
        f"Recent actions: {_cell(r.get('history_action_reprs'))[:180]}\nOperation?"
    )


OPERATION_CLASSIFY = ClassifyTask(
    context=operation_context,
    classes=("CLICK", "TYPE", "SELECT"),
    instruction="Classify the web UI action. Answer with exactly one word: click, type, or select.\n\n",
    parse_order=("SELECT", "TYPE", "CLICK"),  # rarer classes first; CLICK is the fallback
    shots_per_class=2,
    max_input=1024,
    max_new_tokens=4,
    batch_size=16,
    sft_epochs=2,
    source="Table 1, Mind2Web operation selection: Few-shot and FT-SLM",
)
OPERATION = Preset(
    query=(("instruction", 280), "target_tag", "target_type", ("history_action_reprs", 180)),
    doc=("_class_text",),
    classes=("CLICK", "TYPE", "SELECT"),
    class_text={
        "CLICK": "click a button, link or other clickable element",
        "TYPE": "type text into an input field or text box",
        "SELECT": "select an option from a dropdown menu",
    },
    metrics=("accuracy",),
    dsr=DSRConfig(max_chars=None, max_length=512, epochs=3),
    llm=OPERATION_CLASSIFY,
    paper={"dsr-zs": 0.28, "dsr-ft": 0.84, "fewshot": 0.70, "ft-slm": 0.85},
    source=(
        "Table 1, Mind2Web operation selection row (macro-F1): Few-shot and FT-SLM; DSR reconstructed "
        "(the classes ranked as passages)"
    ),
)

# --------------------------------------------------------------------------- tau2-bench
TAU_AGENT_LISTWISE = ListwiseTask(
    task_cols=("instruction",),
    cand_cols=("tool_name",),
    instruction=(
        "Select which tools are relevant to accomplish the task. "
        "Output ONLY the relevant tool numbers, most relevant first, comma-separated.\n\n"
    ),
    header="TOOLS",
    task_chars=600,
    cand_chars=None,
    shots="first",
    max_input=8192,
    max_new_tokens=60,
    sft_epochs=3,
    source="per-agent τ² tables (not a Table 1 row)",
)
TAU_AGENTS = (
    "tau2/claude-3-7-sonnet-20250219",
    "tau2/gpt-4.1-2025-04-14",
    "tau2/gpt-4.1-mini-2025-04-14",
    "tau2/o4-mini-2025-04-16",
)


def tau_agent(name: str) -> Preset:
    """The per-agent τ² tables of Table 2. Table 1 reports no baselines for them."""
    return Preset(
        query=("instruction",),
        doc=("tool_name",),
        metrics=("p@r", "recall@9"),
        dsr=DSRConfig(max_chars=256, max_length=512, epochs=3),
        llm=TAU_AGENT_LISTWISE,
        slice_col="domain",
        source=f"per-agent τ² table {name} (not a Table 1 row)",
    )


TAU_DOMAIN_LISTWISE = ListwiseTask(
    task_cols=("instruction",),
    cand_cols=("tool_name",),
    instruction=(
        "Select which tools are relevant to the task. Output ONLY the relevant candidate numbers, "
        "comma-separated.\n\n"
    ),
    task_chars=400,
    cand_chars=80,
    shots="random",
    shot_seed=20260727,
    max_input=2048,
    max_new_tokens=40,
    sft_epochs=4,
    source="Table 1, τ²-bench rows: Few-shot and FT-SLM",
)
TAU_DOMAIN_PAPER = {
    "tau2/airline": {"dsr-zs": 0.30, "dsr-ft": 0.44, "fewshot": 0.44, "ft-slm": 0.60},
    "tau2/retail": {"dsr-zs": 0.48, "dsr-ft": 0.42, "fewshot": 0.59, "ft-slm": 0.57},
    "tau2/telecom": {"dsr-zs": 0.46, "dsr-ft": 0.28, "fewshot": 0.55, "ft-slm": 0.71},
    "tau2/all-domains": {"dsr-zs": 0.34, "dsr-ft": 0.42, "fewshot": 0.31, "ft-slm": 0.57},
}
TAU_ROWS = {
    "tau2/airline": "airline",
    "tau2/retail": "retail",
    "tau2/telecom": "telecom",
    "tau2/all-domains": "combined",
}


def tau_domain(name: str) -> Preset:
    return Preset(
        query=("instruction",),
        doc=("tool_name",),
        metrics=("p@r", "recall@9"),
        dsr=DSRConfig(max_chars=200, max_length=96, epochs=2),
        dsr_seed=20260727,
        llm=TAU_DOMAIN_LISTWISE,
        slice_col="domain" if name == "tau2/all-domains" else None,
        paper=TAU_DOMAIN_PAPER[name],
        source=f"Table 1, τ²-bench {TAU_ROWS[name]} row (DSR-ZS, DSR-FT, Few-shot, FT-SLM)",
    )


# --------------------------------------------------------------------------- LLMRouterBench
ROUTER = Preset(
    query=("origin_query",),
    doc=("model_id",),
    metrics=("route@1",),
    dsr=DSRConfig(max_chars=256, max_length=512, epochs=3),
    ce=CrossEncoderTask(
        task_cols=("origin_query",),
        cand_cols=("model_id",),
        template="Task: {task}\nModel: {cand}\nWill this model solve the task?",
        task_chars=600,
        cand_chars=80,
        max_length=256,
        source="reconstructed (Table 1, model routing: FT-SLM)",
    ),
    paper={"dsr-zs": 0.49, "dsr-ft": 0.60, "fewshot": 0.50, "ft-slm": 0.70},
    source="reconstructed (Table 1, model routing row: DSR-ZS, DSR-FT); no few-shot prompt is included",
)

PRESETS: dict[str, Preset] = {
    "skillsbench/skill-select": SKILL_SELECT,
    "skillsbench/skill-gating": SKILL_GATING,
    "mind2web/element-150": ELEMENT,
    "mind2web/element": replace(ELEMENT, paper={}),
    "mind2web/operation": OPERATION,
    **{name: tau_agent(name) for name in TAU_AGENTS},
    **{name: tau_domain(name) for name in TAU_DOMAIN_PAPER},
    "llmrouterbench/tau2": ROUTER,
}
