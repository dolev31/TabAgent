"""Hypothesis strategies for messy, adversarial agent logs in every supported format (fabricated data only).

Every generated run carries sentinels that must never reach a decision-point prefix: ``TASKID-*`` (the task
id), ``OUTCOME-SENTINEL`` (outcome fields) and, on each decision and every later step, ``DECISION-<k>`` /
``FUTURE-<k>``. ``check_pipeline`` runs the whole adapter chain on a folder and checks the invariants.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

import tabagent
from tabagent.adapters import (
    NEXT_TOOL_CALL,
    TOOL_SHORTLIST,
    CugaAdapter,
    OpenAIMessagesAdapter,
    detect_adapter,
    live_prefix,
)
from tabagent.adapters.live import run_before
from tabagent.adapters.logfiles import LogChangedError, MixedFormatsError
from tabagent.scanner import check_logs
from tabagent.tabschema.trajectory import GenericJsonAdapter, json_copy, step_name, steps_of

# Reproducible in CI (derandomized, no example database in the repository); TABAGENT_FUZZ=<examples> explores
# new random examples locally.
_FUZZ = int(os.environ.get("TABAGENT_FUZZ", "0") or 0)
settings.register_profile(
    "tabagent-adapters",
    max_examples=_FUZZ or 40,
    derandomize=not _FUZZ,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,
)
settings.load_profile("tabagent-adapters")

OUTCOME = "OUTCOME-SENTINEL"
TOOL_NAMES = [
    "search_flights",
    "Search_Flights",  # a case variant: another tool
    "book flight",  # spaces
    "mcp.server/get-weather",  # dots and slashes (MCP)
    "noter_été_日本",  # unicode
    "t" * 200,  # very long
    "send_email",
]
OUTCOME_VALUES = [
    True, False, 1, 0, 0.5, -1, "true", "PASS", "1.0", "yes", "no", "FAIL", "", "n/a", "None",
    math.nan, math.inf, {"passed": True}, [1], None,
]  # fmt: skip
TASK_IDS = [None, "", "default", 0, 7, 3.5, True, "dup", "TASKID-{k}"]
TEXTS = st.one_of(
    st.just(""),
    st.text(max_size=30),
    st.sampled_from(["a b", "x\x00y", "😀 emoji", "line\r\nbreak", "\\ud83d", "﻿bom"]),
)


def _task_id(draw, k: int) -> Any:
    tid = draw(st.sampled_from(TASK_IDS))
    return tid.format(k=k) if isinstance(tid, str) and "{k}" in tid else tid


def _outcome(draw) -> dict:
    out = {}
    for key in draw(st.lists(st.sampled_from(["success", "reward", "score"]), max_size=2, unique=True)):
        out[key] = draw(st.sampled_from(OUTCOME_VALUES))
    if draw(st.booleans()):
        out["final_answer"] = f"{OUTCOME} answer"
        out["ground_truth"] = {"expected": OUTCOME}
    return out


# --------------------------------------------------------------------------- chat messages
def _tool_spec(draw, name: str) -> Any:
    style = draw(st.sampled_from(["openai", "bare", "anthropic", "mcp", "broken", "string"]))
    props = {"q": {"type": "string"}}
    if style == "openai":
        return {
            "type": "function",
            "function": {"name": name, "description": "d", "parameters": {"properties": props}},
        }
    if style == "bare":
        return {
            "name": name,
            "description": draw(st.sampled_from(["d", "", None, {"x": 1}])),
            "parameters": props,
        }
    if style == "anthropic":
        return {"name": name, "description": "d", "input_schema": {"type": "object", "properties": props}}
    if style == "mcp":
        return {"name": name, "inputSchema": {"properties": ["not", "a", "dict"]}}
    if style == "broken":
        return draw(st.sampled_from([{"type": "function"}, {"function": "x"}, 5, None, {"name": None}]))
    return name


def _tools(draw, names: list[str]) -> Any:
    style = draw(
        st.sampled_from(["missing", "none", "empty", "list", "mapping", "json-text", "junk", "dupes"])
    )
    if style == "missing":
        return None
    if style == "none":
        return {"tools": None}
    if style == "empty":
        return {"tools": []}
    specs = [_tool_spec(draw, n) for n in names]
    if style == "list":
        return {"tools": specs}
    if style == "mapping":
        return {"tools": {n: "desc" for n in names}}
    if style == "json-text":
        return {"tools": json.dumps(specs)}
    if style == "dupes":
        return {
            "tools": specs + [{"name": names[0], "description": "another description"}] if names else specs
        }
    return {"tools": draw(st.sampled_from(["search, book", "/no/such/catalog.json", 42, {"type": "object"}]))}


def _call(draw, name: str, k: int, j: int, dialect: str) -> dict:
    args = draw(st.sampled_from(['{"q": "x"}', "{not json", {"q": 1}, None, "[1, 2]"]))
    cid = f"c{k}_{j}"
    if dialect == "langchain":
        return {"name": name, "args": args if isinstance(args, dict) else {}, "id": cid}
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


DIALECTS = ("openai", "anthropic", "langchain", "legacy", "responses", "kwargs", "text")


class _Marks:
    """Unique message markers (``M000``, ``M001``, ...) of one generated chat run, and what the oracle
    expects: the markers before each tool-calling turn, the system and user markers before the first one,
    and the tools each turn calls."""

    def __init__(self) -> None:
        self.n = 0
        self.seen: list[tuple[str, str]] = []  # (marker, role) in message order
        self.before: list[list[str]] = []  # per turn: every marker before it
        self.shortlist: list[str] | None = None  # system and user markers before the first turn
        self.tools: list[list[str]] = []  # per turn: the tools it calls

    def mark(self, role: str) -> str:
        m = f"M{self.n:03d}"
        self.n += 1
        self.seen.append((m, role))
        return m

    def turn(self, called: list[str]) -> None:
        markers = [m for m, _ in self.seen]
        if self.shortlist is None:
            self.shortlist = [m for m, r in self.seen if r in ("system", "user")]
        self.before.append(markers)
        self.tools.append(called)


def _assistant_turn(draw, k: int, names: list[str], marks: _Marks, dialects: tuple = DIALECTS) -> list[Any]:
    """An assistant turn that calls tools (parallel, repeated), in one of the dialects, and its results.
    Its text holds DECISION-<k>; every later message holds FUTURE-<k>; tool results hold a marker."""
    called = draw(st.lists(st.sampled_from(names), min_size=1, max_size=3))
    dialect = draw(st.sampled_from(dialects))
    text = draw(st.sampled_from(["", f"DECISION-{k}", f"thinking DECISION-{k}"]))
    marks.turn([called[0]] if dialect == "legacy" else called)
    if dialect == "anthropic":
        blocks = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": f"tu{k}_{j}", "name": n, "input": {"q": 1}}
            for j, n in enumerate(called)
        ]
        results = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"tu{k}_{j}",
                    "content": f"{marks.mark('tool')} FUTURE-{k} ok",
                }
                for j in range(len(called))
            ],
        }
        return [{"role": "assistant", "content": blocks}, results]
    if dialect == "responses":  # a response's own message item: the calls join it (never an earlier reply)
        items: list[Any] = [{"type": "reasoning", "summary": []}]
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text or f"DECISION-{k}"}],
            }
        )
        outputs = []  # a response's call items come together; their outputs follow, as the API logs them
        for j, n in enumerate(called):
            kind = draw(st.sampled_from(["function_call", "custom_tool_call"]))
            items.append({"type": kind, "call_id": f"r{k}_{j}", "name": n, "arguments": "{}", "input": "x"})
            out_kind = "function_call_output" if kind == "function_call" else "custom_tool_call_output"
            outputs.append(
                {"type": out_kind, "call_id": f"r{k}_{j}", "output": f"{marks.mark('tool')} FUTURE-{k}"}
            )
        items += outputs
        return items
    if dialect == "legacy":
        msg = {
            "role": "assistant",
            "content": text or None,
            "function_call": {"name": called[0], "arguments": "{}"},
        }
        return [msg, {"role": "function", "name": called[0], "content": f"{marks.mark('tool')} FUTURE-{k}"}]
    if dialect == "langchain":
        calls = [_call(draw, n, k, j, dialect) for j, n in enumerate(called)]
        return [
            {"type": "ai", "content": text, "tool_calls": calls},
            *(
                {
                    "type": "tool",
                    "content": f"{marks.mark('tool')} FUTURE-{k}",
                    "tool_call_id": c["id"],
                    "name": c["name"],
                }
                for c in calls
            ),
        ]
    calls = [_call(draw, n, k, j, dialect) for j, n in enumerate(called)]
    if dialect == "kwargs":
        msg = {"role": "assistant", "content": text, "additional_kwargs": {"tool_calls": calls}}
    elif dialect == "text":
        msg = {"role": "assistant", "content": text, "tool_calls": json.dumps(calls)}  # stringified calls
    else:
        msg = {"role": "assistant", "content": text, "tool_calls": calls}
    tool_msgs = []
    for c in calls:
        content = draw(st.sampled_from(["result", None, "\x00\x01" * 40]))
        if content == "result":
            content = f"{marks.mark('tool')} FUTURE-{k}"
        tool_msgs.append({"role": "tool", "tool_call_id": c["id"], "content": content})
    return [msg, *tool_msgs]


def _noise(draw, k: int, marks: _Marks) -> list[Any]:
    out: list[Any] = []
    for kind in draw(st.lists(st.sampled_from(["user", "system", "assistant", "junk", "tool"]), max_size=2)):
        if kind == "junk":
            out.append(
                draw(st.sampled_from([5, "str", None, {"no": "role"}, {"role": None}, {"role": "critic"}]))
            )
            continue
        text = draw(TEXTS) if kind == "user" else "noise"
        content = f"{marks.mark(kind)} FUTURE-{k} {text}"
        msg: dict[str, Any] = {"role": kind, "content": content}
        if kind == "tool":
            msg["tool_call_id"] = "nowhere"  # an orphan result
        out.append(msg)
    return out


@st.composite
def chat_run(draw, k: int = 0, dialects: tuple = DIALECTS) -> dict:
    """One chat run with messy fields, dialects and outcomes; ``_oracle`` (never a prefix field) holds what
    its decision points must be: the markers before each turn and the tools each turn calls."""
    names = draw(st.lists(st.sampled_from(TOOL_NAMES), min_size=1, max_size=4, unique=True))
    marks = _Marks()
    run: dict[str, Any] = {}
    if draw(st.booleans()):
        run["system"] = [{"type": "text", "text": f"{marks.mark('system')} anthropic system"}]
    msgs: list[Any] = []
    if draw(st.booleans()):
        role = draw(st.sampled_from(["system", "developer"]))
        msgs.append({"role": role, "content": f"{marks.mark('system')} be helpful"})
    first = draw(st.sampled_from(["text", "blocks", "none"]))
    if first == "text":
        msgs.append({"role": "user", "content": f"{marks.mark('user')} book it"})
    elif first == "blocks":
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"{marks.mark('user')} hi"}]})
    else:
        msgs.append({"role": "user", "content": None})
    msgs += _noise(draw, -1, marks)
    for t in range(draw(st.integers(0, 3))):
        msgs += _assistant_turn(draw, t, names, marks, dialects)
        msgs += _noise(draw, t, marks)
    run["messages"] = msgs
    run["_oracle"] = {"before": marks.before, "shortlist": marks.shortlist or [], "tools": marks.tools}
    tid = _task_id(draw, k)
    if tid is not None or draw(st.booleans()):
        run["task_id"] = tid
    if draw(st.booleans()):
        run["instruction"] = draw(st.sampled_from(["do it", "", "x" * 50]))
    tools = _tools(draw, names)
    if tools:
        run.update(tools)
    run.update(_outcome(draw))
    return run


# --------------------------------------------------------------------------- step logs (CUGA and generic)
CUGA_NAMES = ["TaskAnalyzerAgent", "APIPlannerAgent", "ShortlisterAgent", "CodeAgent", "PlanControllerAgent"]
APIS = {"phone": ["phone_search", "phone_call"], "venmo": ["venmo_pay"], "Mail": ["Mail_send"]}


def _code(draw, k: int) -> tuple[str, list | None]:
    """Generated code calling ``call_api`` in every form, and the ``(app, API)`` calls a reader of the code
    must find (the oracle)."""
    app, api = draw(
        st.sampled_from([(a, x) for a, xs in APIS.items() for x in xs] + [("ghost", "ghost_api")])
    )
    bare = api.split("_", 1)[1]
    forms = {  # form -> (code, calls when the code parses, calls when only the pattern can read it)
        "positional": (f'r = await call_api("{app}", "{api}", {{}})', [(app, api)], [(app, api)]),
        "bare api": (f"r = await call_api('{app}', '{bare}')", [(app, api)], [(app, api)]),
        "keywords": (
            f'r = await call_api(api_name="{api}", app_name="{app}")',  # reversed
            [(app, api)],
            [(app, api)],
        ),
        "comment": (f'# r = await call_api("{app}", "commented_out")', [], []),
        "string": (
            f'print("call_api(\\"{app}\\", \\"in_a_string\\")")',  # a string: no call when parsed
            [],
            [(app, f"{app}_in_a_string")],
        ),
        "f-string": (f'r = await call_api(f"{app}", "{api}")', [(app, api)], []),
        "variable": ('r = await call_api(unknown_app, "x")', [], []),
        "assigned": (f'the_app = "{app}"\nr = await call_api(the_app, "{api}")', [(app, api)], []),
        "fence in string": (
            f'fence = "```"\nr = await call_api("{app}", "{api}")',
            [(app, api)],
            [(app, api)],
        ),
        "cut off": (f'r = await call_api("{app}", "{api}"', None, [(app, api)]),
    }
    chosen = draw(st.lists(st.sampled_from(list(forms)), min_size=1, max_size=3))
    body = "\n".join(forms[f][0] for f in chosen)
    parses = "cut off" not in chosen
    calls = [c for f in chosen for c in (forms[f][1] if parses else forms[f][2])]
    if "assigned" in chosen and chosen.count("assigned") > 1:
        calls = None  # assigned twice: dynamic (not modelled here)
    wrap = draw(st.sampled_from(["plain", "fenced", "unclosed"]))
    if wrap == "fenced":
        body = f"```python\n{body}\n```"
    elif wrap == "unclosed":
        body = f"```\n{body}"
    if wrap != "plain" and "fence in string" in chosen:
        calls = None  # fences inside and around the code: not modelled
    return body, calls


def _step(draw, k: int, cuga: bool) -> Any:
    """A step holding its marker STEP-<k>; malformed in many ways."""
    kind = draw(st.integers(0, 9))
    if kind == 0:
        return draw(st.sampled_from([f"STEP-{k}", k, None, [f"STEP-{k}"]]))  # not an object
    name: Any = draw(st.sampled_from(CUGA_NAMES if cuga else ["router", "check", "lookup", "router"]))
    if kind == 1:
        name = draw(st.sampled_from([None, "", 5, ["ShortlisterAgent"], {"n": 1}]))
    step: dict[str, Any] = {"marker": f"STEP-{k}"}
    if kind != 2:
        step["name"] = name
    prompts_style = draw(st.integers(0, 4))
    if name == "ShortlisterAgent" and prompts_style < 3:
        app = draw(st.sampled_from(list(APIS)))
        block = json.dumps(
            {app: {a: {"api_name": a, "description": "d", "parameters": []} for a in APIS[app]}}
        )
        step["prompts"] = [
            {"role": "system", "value": "select"},
            {"role": "human", "value": f'`Current Application`: {app}\n`Available APIs`:\n"""\n{block}\n"""'},
        ]
        step["data"] = json.dumps({"result": [{"name": a} for a in APIS[app]]})
    elif name == "TaskAnalyzerAgent" and prompts_style < 3:
        apps = [{"name": a, "description": "app"} for a in APIS]
        step["prompts"] = [{"role": "system", "value": f"Human: {{'intent': 'x', 'applications': {apps!r}}}"}]
        step["data"] = draw(st.sampled_from([json.dumps(apps[:1]), apps[:2], "", {"apps": [{"no": "name"}]}]))
    elif name == "CodeAgent":
        code, calls = _code(draw, k)
        wrapper = draw(st.sampled_from(["json", "dict", "raw", "none", "number"]))
        step["data"] = {"json": json.dumps({"code": code}), "dict": {"code": code}, "raw": code}.get(
            wrapper, None if wrapper == "none" else {"code": 5}
        )
        step["oracle_calls"] = (
            None if calls is None else [] if wrapper in ("none", "number") else [list(c) for c in calls]
        )
    else:
        step["prompts"] = draw(
            st.sampled_from([[], None, "str", {"role": "x"}, [5, {"role": "assistant", "value": None}]])
        )
        step["data"] = draw(st.sampled_from(["", "{}", json.dumps({"action": "go"}), None, {"x": [1, 2]}]))
    return step


@st.composite
def step_run(draw, k: int = 0, cuga: bool = True) -> dict:
    """One step log (CUGA-shaped when ``cuga``), with markers on every step."""
    n = draw(st.integers(0, 8))
    steps = [_step(draw, i, cuga) for i in range(n)]
    if draw(st.booleans()):
        steps.append({"name": "EvaluationResult", "marker": OUTCOME, "data": json.dumps({"score": 1})})
    run: dict[str, Any] = {"steps": steps, "intent": "do the task", "dataset_name": "fabricated"}
    tid = _task_id(draw, k)
    if tid is not None or draw(st.booleans()):
        run["task_id"] = tid
    run.update(_outcome(draw))
    if cuga:
        run["score"] = draw(st.sampled_from([1.0, 0.0, "1", "true", None, math.nan, {"v": 1}]))
        run["eval"] = {"passed": OUTCOME}
        run["actions_count"] = n
    return run


# --------------------------------------------------------------------------- tau2-bench results
@st.composite
def tau2_results(draw, k: int = 0) -> dict:
    sims = []
    for i in range(draw(st.integers(0, 3))):
        run = draw(chat_run(k * 10 + i, ("openai", "kwargs", "text")))
        msgs = [
            m
            for m in run["messages"]
            if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")
        ]
        sim: Any = {
            "id": draw(st.sampled_from([f"sim-{k}-{i}", "same-sim", None])),
            "task_id": draw(st.sampled_from([str(i), i, None, "missing-task"])),
            "trial": 0,
            "messages": msgs,
            "reward_info": draw(
                st.sampled_from([{"reward": 1.0}, {"reward": 0.0}, None, 1.0, "1", {"reward": "x"}])
            ),
        }
        sims.append(draw(st.sampled_from([sim, sim, sim, "not a simulation"])))
    info = draw(
        st.sampled_from(
            [
                {
                    "environment_info": {"domain_name": "airline", "tool_defs": None, "policy": "p"},
                    "agent_info": {},
                },
                {
                    "environment_info": {"domain_name": "telecom"},
                    "agent_info": {"implementation": "llm_agent_solo"},
                },
                None,
                "not info",
            ]
        )
    )
    tasks = draw(st.sampled_from([[{"id": "0", "ticket": "the ticket"}], [], None, ["x"], [{"id": 1}]]))
    return {"info": info, "tasks": tasks, "simulations": sims}


# --------------------------------------------------------------------------- the invariants
DOCUMENTED = (ValueError, NotImplementedError, FileNotFoundError, LogChangedError, KeyError)


def _sentinels(raw: Any, group: str | None) -> list[str]:
    out = [OUTCOME]
    tid = raw.get("task_id") if isinstance(raw, dict) else None
    if isinstance(tid, str) and tid.startswith("TASKID-"):
        out.append(tid)
    return out


def _later(prefix_text: str, at: int) -> list[str]:
    """Markers of the decision turn ``at`` or later found in a prefix."""
    found = re.findall(r"(?:DECISION|FUTURE)-(\d+)", prefix_text)
    return [f for f in found if int(f) >= at]


def check_prefix(adapter: Any, traj: Any, dp: Any, decision: str, turn: int | None = None) -> None:
    """The prefix holds only what precedes the decision: no outcome fields, no task id, no decision or later
    step. Chat prefixes hold exactly the generator's markers before the decision (an oracle independent of the
    adapter's normalization: every system and user marker before the first tool call for ``ToolShortlist``,
    every marker before the turn for ``NextToolCall``); step-log prefixes hold exactly the steps before it
    (outcome steps dropped)."""
    text = json.dumps(dp.prefix, ensure_ascii=False, default=str)
    for s in _sentinels(traj.raw, dp.group):
        assert s not in text, (dp.id, s)
    outcome_keys = set(getattr(adapter, "outcome_keys", ()))
    assert not outcome_keys & set(dp.prefix), dp.id
    assert "task_id" not in dp.prefix and "_oracle" not in dp.prefix, dp.id
    if isinstance(adapter, OpenAIMessagesAdapter):
        oracle = traj.raw.get("_oracle")
        if oracle is not None and turn is not None:
            expected = oracle["shortlist"] if decision == TOOL_SHORTLIST else oracle["before"][turn]
            assert re.findall(r"M\d{3}", text) == expected, (dp.id, decision, turn)
        at = 0 if decision == TOOL_SHORTLIST else turn
        if at is not None:
            assert not _later(text, at), (dp.id, _later(text, at))
    else:
        raw_steps = steps_of(traj.raw, adapter.steps_key)
        drop = set(adapter.outcome_steps)
        expected = [s for s in raw_steps[: dp.step_index] if step_name(s, adapter.step_name_key) not in drop]
        assert dp.prefix[adapter.steps_key] == expected, dp.id
        markers = {int(m) for m in re.findall(r"STEP-(\d+)", text)}
        assert all(m < dp.step_index for m in markers), dp.id


def oracle_labels(adapter: Any, traj: Any, dp: Any, decision: str, turn: int) -> set[str] | None:
    """The labels a decision point must have, from the generator's oracle (None when it is not modelled)."""
    if isinstance(adapter, OpenAIMessagesAdapter):
        oracle = traj.raw.get("_oracle")
        if oracle is None:
            return None
        if decision == TOOL_SHORTLIST:
            return {t for tools in oracle["tools"] for t in tools}
        return set(oracle["tools"][turn])
    if isinstance(adapter, CugaAdapter) and adapter.labels_fn is None:
        steps = steps_of(traj.raw, adapter.steps_key)
        name = step_name(dp.target_step, adapter.step_name_key)
        calls: list = []
        for s in steps[dp.step_index + 1 :]:
            if name == "ShortlisterAgent" and step_name(s, adapter.step_name_key) == "ShortlisterAgent":
                break
            if isinstance(s, dict) and step_name(s) == "CodeAgent":
                if s.get("oracle_calls") is None:
                    return None
                calls += s["oracle_calls"]
        if name == "ShortlisterAgent":
            return {api for _, api in calls}
        return {app for app, _ in calls if app}
    return None


def decisions_of(adapter: Any, comps: dict) -> list[str]:
    if isinstance(adapter, OpenAIMessagesAdapter):
        return [TOOL_SHORTLIST, NEXT_TOOL_CALL] if comps else []
    if isinstance(adapter, CugaAdapter):
        return [d for d in ("ShortlisterAgent", "TaskAnalyzerAgent") if d in comps]
    return list(comps)[:2]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def check_pipeline(path: Any, adapter: Any = None, onboard: bool = False) -> dict:
    """Run detect -> load -> components -> decision points -> candidates and labels -> live prefixes -> scan
    (-> onboard dry run) on ``path``, assert the invariants, and return everything as plain data."""
    try:
        ad = adapter if adapter is not None else detect_adapter(path)
        trajs = ad.load(path)
    except MixedFormatsError as e:
        assert "more than one log format" in str(e) and "point at a folder" in str(e)
        return {"error": "mixed"}
    except (ValueError, FileNotFoundError) as e:
        assert str(e)
        return {"error": type(e).__name__}
    good = [t for t in trajs if t.success]
    assert len({t.id for t in trajs}) == len(trajs), "run ids must be unique"
    for w in getattr(ad, "warnings", []):
        assert isinstance(w, str) and w
    comps = ad.components(good)
    assert list(comps) == sorted(comps, key=lambda c: (-comps[c], c)), (
        "components: most frequent, then by name"
    )
    out: dict[str, Any] = {
        "runs": [(t.id, t.success, t.group, t.source) for t in trajs],
        "components": comps,
        "dps": {},
    }
    outside: dict[str, int] = {}  # decision points with a label outside their candidates, per decision
    for decision in decisions_of(ad, comps):
        for t in good:
            for turn, dp in enumerate(ad.decision_points(t, decision)):
                check_prefix(ad, t, dp, decision, turn)
                assert live_prefix(ad, run_before(ad, t, dp), decision) == dp.prefix, ("parity", dp.id)
                try:
                    cands = sorted(c.id for c in ad.candidates(dp))
                    labels = sorted(ad.labels(dp))
                except NotImplementedError:
                    cands = labels = None
                if cands is not None:
                    assert all(isinstance(c, str) and c for c in cands)
                    assert len(cands) == len(set(cands)), "candidates are unique"
                    expected = oracle_labels(ad, t, dp, decision, turn)
                    if expected is not None:
                        assert set(labels) == expected, (dp.id, decision, sorted(labels), sorted(expected))
                    if cands and set(labels) - set(cands):
                        outside[decision] = outside.get(decision, 0) + 1
                out["dps"][f"{decision}:{dp.id}"] = (dp.group, canonical(dp.prefix), cands, labels)
    if good and out["dps"]:  # check_logs reports exactly the labels that are not among the candidates
        found = {d.name: d.labels_outside for d in check_logs(path, adapter=_fresh(ad), sample=0).decisions}
        for decision in decisions_of(ad, comps):
            if decision in found:
                assert found[decision] == outside.get(decision, 0), (decision, found, outside)
    if trajs:
        try:
            report = tabagent.scan(path, adapter=_fresh(ad))
            report.to_json()
            report.summary()
        except ValueError as e:
            assert str(e)
    if onboard and good:
        for decision in decisions_of(ad, comps)[:1]:
            try:
                tabagent.onboard(path, decision, adapter=_fresh(ad), dry_run=True)
            except (ValueError, NotImplementedError) as e:
                assert str(e)
    return out


def _fresh(adapter: Any) -> Any:
    """A new adapter like ``adapter`` (scan and onboard load the logs again)."""
    if isinstance(adapter, GenericJsonAdapter) and not isinstance(adapter, CugaAdapter):
        return GenericJsonAdapter(candidates_fn=adapter.candidates_fn, labels_fn=adapter.labels_fn)
    return type(adapter)()


MUTATIONS = ("none", "bom", "utf16", "cp1252", "crlf", "trailing commas", "lone surrogate", "big", "deep")


def _trailing(value: Any) -> str:
    """JSON with a trailing comma in every non-empty object and array (which JSON does not allow)."""
    if isinstance(value, dict) and value:
        return "{" + ", ".join(f"{json.dumps(str(k))}: {_trailing(v)}" for k, v in value.items()) + ",}"
    if isinstance(value, list) and value:
        return "[" + ", ".join(_trailing(v) for v in value) + ",]"
    return json.dumps(value)


def _mutated(value: Any, mutation: str) -> Any:
    """A run with a value the reader must survive: a lone surrogate, a large string, a deep value."""
    if not isinstance(value, dict):
        return value
    if mutation == "lone surrogate":
        return {**value, "note": "cut " + chr(0xD83D)}
    if mutation == "big":
        return {**value, "payload": "words, " * 50_000}
    if mutation == "deep":
        deep: Any = "leaf"
        for _ in range(250):
            deep = {"d": deep}
        return {**value, "deep": deep}
    return value


def _encode(text: str, mutation: str) -> bytes:
    if mutation == "bom":
        return b"\xef\xbb\xbf" + text.encode("utf-8", "surrogatepass")
    if mutation == "utf16":
        return text.encode("utf-16", "surrogatepass")
    if mutation == "cp1252":
        try:
            return text.encode("cp1252")
        except UnicodeEncodeError:
            pass
    return text.encode("utf-8", "surrogatepass")


def write(folder: Any, files: dict[str, Any], mutation: str = "none") -> Any:
    """Write ``{name: value}`` as JSON (``.jsonl``: a list of values, one per line; bytes as they are), the
    way ``mutation`` says (an encoding, CRLF, trailing commas, or a lone surrogate, a large string or a
    deep value added to each run)."""
    folder.mkdir(parents=True, exist_ok=True)

    def dump(v: Any, pretty: bool) -> str:
        v = _mutated(v, mutation)
        if mutation == "trailing commas":
            return _trailing(v)
        ascii_only = mutation not in ("cp1252", "lone surrogate")
        text = json.dumps(v, ensure_ascii=ascii_only, indent=1 if pretty and mutation == "crlf" else None)
        return text.replace("\n", "\r\n") if mutation == "crlf" else text

    for name, value in files.items():
        f = folder / name
        f.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            f.write_bytes(value)
        elif name.endswith(".jsonl"):
            newline = "\r\n" if mutation == "crlf" else "\n"
            f.write_bytes(_encode("".join(dump(v, False) + newline for v in value), mutation))
        else:
            f.write_bytes(_encode(dump(value, True), mutation))
    return folder


def copy_runs(runs: list[Any]) -> list[Any]:
    return [json_copy(r) for r in runs]
