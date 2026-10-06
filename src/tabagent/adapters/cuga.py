"""CUGA agent trajectories (github.com/cuga-project/cuga-agent, Apache-2.0).

A CUGA trajectory file is ``{"intent", "task_id", "steps", "score", "eval", ...}``; each step names the agent
node that produced it (``TaskAnalyzerAgent``, ``PlanControllerAgent``, ``APIPlannerAgent``,
``ShortlisterAgent``, ``CodeAgent``, ``FinalAnswerAgent``, ...) and carries its ``prompts`` (role/value) and
``data`` (the node's output, usually JSON). Evaluated runs end with an ``EvaluationResult`` step.

Two decisions are built in, the two CUGA components the paper replaces. The API shortlister (the paper's CUGA
tool shortlisting, a ranking): at each
``ShortlisterAgent`` step the candidates are the APIs offered to it
(the ``Available APIs`` block of its prompt, else a catalog given here, e.g. CUGA API docs, else the APIs
seen for that app across the loaded runs), and the labels are the APIs the ``CodeAgent`` then called
(``call_api(app, api)``, app-prefixed as ``<app>_<api>``) before the next shortlisting. The prefix is the
trajectory truncated before the step, with only the ``intent`` and ``dataset_name`` fields (``task_keys``;
never the ``task_id``, the cross-validation group, nor ``score``, ``eval`` or ``actions_count``, a post-hoc
count) and without the ``EvaluationResult`` step.

The app analyzer (``TaskAnalyzerAgent``, the paper's app selection, a multi-label decision: onboard it
with ``kind="multilabel"``, which ``onboard`` and ``scan`` pick by themselves): the candidates are the apps
listed in its prompt's input (``applications``, else the apps seen across the loaded runs) and the labels
are the apps the successful run then called (the paper's G(t), the candidates the run used).
``app_labels="offered"`` labels the apps the analyzer itself chose instead, as the paper's legacy AppWorld
data did; on the successful CUGA AppWorld runs checked, the two agree on nearly all.

Other components work with ``candidates_fn`` and ``labels_fn``.

For ``tabagent.scan``, every step of an agent node (or with recorded prompts) is an LLM call, one per recorded
generation: the shortlister's choices are the APIs it returned, other nodes report the categorical fields of
their JSON output, and the code, answer and chat nodes generate free text. CUGA records no tokens, cost or
latency per step (its ``results.json`` holds per-task totals only), so a scan reports them as not recorded.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tabagent.adapters.common import parse_catalog
from tabagent.tabschema.trajectory import (
    Candidate,
    DecisionPoint,
    GenericJsonAdapter,
    LLMCall,
    Trajectory,
    output_fields,
    parse_json,
    read_outcome,
    read_usage,
    step_name,
    steps_of,
)

SHORTLISTER = "ShortlisterAgent"
ANALYZER = "TaskAnalyzerAgent"
#: Top-level fields of a CUGA trajectory that its prefixes keep: what describes the task, never the task id.
TASK_KEYS = ("intent", "dataset_name")
CUGA_STEPS = frozenset(
    {
        "TaskAnalyzerAgent",
        "TaskDecompositionAgent",
        "PlanControllerAgent",
        "APIPlannerAgent",
        "APICodePlannerAgent",
        "ShortlisterAgent",
        "CodeAgent",
        "FinalAnswerAgent",
        "BrowserPlannerAgent",
        "ActionAgent",
        "QaAgent",
        "ChatAgent",
        "ReuseAgent",
        "Assistant_code",  # CUGA's CodeAct mode
        "Raw_Assistant_Response",
    }
)
#: Nodes whose output is free text (code, answers, chat), whatever its shape.
FREE_TEXT_STEPS = frozenset(
    {"CodeAgent", "FinalAnswerAgent", "QaAgent", "ChatAgent", "Assistant_code", "Raw_Assistant_Response"}
)
_GENERATION_ROLES = frozenset({"assistant", "generation", "ai"})
_AVAILABLE = re.compile(r'`Available APIs`:\s*"""(.*?)"""', re.S)
_APP = re.compile(r"`Current Application`:\s*([^\s`]+)")
_APPS = re.compile(r"""['"](?:applications|available_apps)['"]\s*:\s*\[""")
_Q = r"""\\?["']"""  # a quote, possibly escaped when the code sits inside a JSON string
_CALL = re.compile(
    rf"call_api\(\s*(?:(\w+)\s*=\s*)?{_Q}([^\"'\\]+){_Q}\s*,\s*(?:(\w+)\s*=\s*)?{_Q}([^\"'\\]+){_Q}"
)
_COMMENT = re.compile(r"(?m)^\s*#.*$|\s#[^\"'\n]*$")
_CODE_FENCE = re.compile(r"```[\w+.-]*[^\S\n]*\n?(.*?)(?:```|\Z)", re.S)
_APP_KEYWORDS = ("app_name", "app")
_API_KEYWORDS = ("api_name", "api", "function_name")


def _prefixed(app: str, api: str) -> str:
    return api if not app or api.startswith(f"{app}_") else f"{app}_{api}"


def _prompts(step: Any) -> list[dict]:
    """A step's recorded prompts (``{"role", "value"}`` objects); anything malformed counts as none."""
    prompts = step.get("prompts") if isinstance(step, dict) else None
    return [p for p in prompts if isinstance(p, dict)] if isinstance(prompts, list) else []


def _value(prompt: dict) -> str:
    v = prompt.get("value", "")
    return v if isinstance(v, str) else json.dumps(v, default=str) if v is not None else ""


def _clean(cands: dict[str, Candidate]) -> dict[str, Candidate]:
    """CUGA API descriptions read ``description: <text>\\nmodel: ...\\n---\\n<schema>``; keep the text."""
    out = {}
    for k, c in cands.items():
        head = [ln for ln in c.description.split("\n---")[0].splitlines() if not ln.startswith("model:")]
        d = re.sub(r"^description:\s*", "", "\n".join(head)).strip()
        out[k] = dataclasses.replace(c, description=d)
    return out


def available_apis(step: dict[str, Any]) -> dict[str, Candidate]:
    """The APIs a shortlister step was offered: the ``Available APIs`` JSON block of its prompt."""
    for p in _prompts(step):
        m = _AVAILABLE.search(_value(p))
        if m:
            data = parse_json(m.group(1))
            if isinstance(data, (dict, list)):
                return _clean(parse_catalog(data))
    return {}


def _literal_list(text: str, start: int) -> list | None:
    """The JSON or Python-literal list that opens at ``text[start]`` (a ``[``), or None."""
    end = start
    for _ in range(64):  # the closing bracket is one of the next ``]``: try them in order
        end = text.find("]", end + 1)
        if end < 0:
            return None
        chunk = text[start : end + 1]
        for parse in (json.loads, ast.literal_eval):
            try:
                value = parse(chunk)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if isinstance(value, list):
                return value
    return None


def available_apps(step: dict[str, Any]) -> dict[str, Candidate]:
    """The apps an app-analyzer step could choose: the ``applications`` (or ``available_apps``) list of the
    input in its prompt, which CUGA writes as a Python literal after ``Human:``, as candidates with their
    descriptions. The instructions before the input show example lists under the same names; the last list is
    the input."""
    for p in reversed(_prompts(step)):
        text = _value(p)
        human = text.rfind("Human:")
        text = text[human:] if human >= 0 and _APPS.search(text, human) else text
        items = None
        for m in reversed(list(_APPS.finditer(text))):
            items = _literal_list(text, m.end() - 1)
            if items:
                break
        if items:
            out = {}
            for a in items:
                name = a.get("name") if isinstance(a, dict) else a
                if isinstance(name, str) and name.strip():
                    desc = a.get("description", "") if isinstance(a, dict) else ""
                    out[name.strip()] = Candidate(name.strip(), name.strip(), str(desc or ""))
            if out:
                return out
    return {}


def chosen_apps(step: dict[str, Any]) -> set[str]:
    """The apps an app-analyzer step chose: the names in its output, a list of app objects (or
    ``{"apps": [...]}``); items without a name are left out."""
    out = step.get("data")
    if not out:
        gens = [p.get("value") for p in _prompts(step) if str(p.get("role", "")).lower() in _GENERATION_ROLES]
        out = gens[-1] if gens else None
    d = parse_json(out) if isinstance(out, str) else out
    if isinstance(d, dict):
        d = d.get("apps") or d.get("result") or []
    if not isinstance(d, list):
        return set()
    names = (a.get("name") if isinstance(a, dict) else a for a in d)
    return {n.strip() for n in names if isinstance(n, str) and n.strip()}


def _constant(node: ast.AST) -> str | None:
    """The text of a string literal (or an f-string without placeholders), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and all(
        isinstance(v, ast.Constant) and isinstance(v.value, str) for v in node.values
    ):
        return "".join(v.value for v in node.values)  # type: ignore[attr-defined]
    return None


def _code_text(code: str) -> str:
    """Generated code without its Markdown fences (CUGA's code node writes them): the code between the opening
    fence and the last one when that parses (a fence inside a string of the code stays), else every fenced
    block, joined."""
    if "```" not in code:
        return code
    text = code.strip()
    if text.startswith("```"):
        body = text.partition("\n")[2]
        end = body.rfind("```")
        whole = body[:end] if end >= 0 else body
        if _parse_code(whole) is not None:
            return whole
    return "\n".join(m.group(1) for m in _CODE_FENCE.finditer(code))


def _parse_code(code: str) -> ast.AST | None:
    flags = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    for src in (code, "async def _f():\n" + "\n".join(f"    {ln}" for ln in code.splitlines())):
        try:
            return compile(src, "<code>", "exec", flags=flags, dont_inherit=True)
        except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
            continue
    return None


def _literal_names(tree: ast.AST) -> dict[str, str]:
    """The names the code binds exactly once, by assigning them a string literal (``app = "phone"``). A name
    bound in any other way as well (assigned twice, a loop variable, a parameter, augmented, imported) is
    dynamic and left out."""
    bound: Counter = Counter()
    values: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound[node.id] += 1
        elif isinstance(node, ast.arg):
            bound[node.arg] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound[node.name] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update({n: 2 for n in node.names})
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound[node.name] += 1
        target = (
            node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else node.target
            if isinstance(node, ast.AnnAssign) and node.value is not None
            else None
        )
        if isinstance(target, ast.Name):
            value = _constant(node.value)  # type: ignore[attr-defined]
            if value is not None:
                values[target.id] = value
    return {k: v for k, v in values.items() if bound[k] == 1}


def _resolved(node: ast.AST | None, names: dict[str, str]) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return names.get(node.id)
    return _constant(node)


def code_api_calls_counted(code: str) -> tuple[set[tuple[str, str]], int]:
    """``(app, api)`` pairs of the ``call_api(app_name, api_name, ...)`` calls in generated code, and how many
    calls name their app or API dynamically (left out). Keyword arguments count in any order; an app or API
    given as a name the code assigns a string literal once (``app = "phone"``) is resolved. Code that parses
    is read as Python, so calls in comments and strings do not count; Markdown code fences are removed only
    when the code does not parse with them. Code that never parses (cut off, escaped inside a JSON string)
    is searched with a pattern, comments removed (no call is then counted as dynamic)."""
    tree = _parse_code(code)
    if tree is None and "```" in code:
        code = _code_text(code)
        tree = _parse_code(code)
    if tree is None:
        out = set()
        for kw1, v1, kw2, v2 in _CALL.findall(_COMMENT.sub("", code)):
            app, api = (v2, v1) if kw1 in _API_KEYWORDS or kw2 in _APP_KEYWORDS else (v1, v2)
            out.add((app.strip(), api.strip()))
        return {(app, api) for app, api in out if app and api}, 0
    names = _literal_names(tree)
    out = set()
    dynamic = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name != "call_api":
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        app_node = node.args[0] if node.args else next((kw[k] for k in _APP_KEYWORDS if k in kw), None)
        if len(node.args) > 1:
            api_node = node.args[1]
        else:
            api_node = next((kw[k] for k in _API_KEYWORDS if k in kw), None)
        app, api = _resolved(app_node, names), _resolved(api_node, names)
        if app and api and app.strip() and api.strip():
            out.add((app.strip(), api.strip()))
        else:
            dynamic += 1
    return out, dynamic


def code_api_calls(code: str) -> set[tuple[str, str]]:
    """The ``(app, api)`` pairs of :func:`code_api_calls_counted`."""
    return code_api_calls_counted(code)[0]


def _step_calls(step: Any) -> tuple[set[tuple[str, str]], int]:
    """``(app, api)`` pairs a step called (``call_api`` in CodeAgent code, or an ``api_call`` step), and the
    calls with dynamic names it made."""
    if not isinstance(step, dict):
        return set(), 0
    data = step.get("data") or ""
    if step.get("name") == "api_call":
        d = parse_json(data)
        fn = d.get("function_name") if isinstance(d, dict) else None
        if isinstance(fn, str) and fn.strip():
            app = d.get("app_name")
            app = app.strip() if isinstance(app, str) else ""
            return {(app, _prefixed(app, fn.strip()))}, 0
        return set(), 0
    if step.get("name") != "CodeAgent":
        return set(), 0
    d = parse_json(data)
    code = d.get("code", "") if isinstance(d, dict) else data if isinstance(data, str) else ""
    if not isinstance(code, str):
        code = ""
    calls, dynamic = code_api_calls_counted(code)
    return {(app, _prefixed(app, api)) for app, api in calls}, dynamic


def called_apis(step: dict[str, Any]) -> set[tuple[str, str]]:
    """``(app, api)`` pairs a step called: ``call_api(app, api)`` in CodeAgent code (see
    :func:`code_api_calls_counted`), or an ``api_call`` step."""
    return _step_calls(step)[0]


def decision_app(dp: DecisionPoint) -> str | None:
    """The app a shortlisting call is about: ``Current Application`` in the step's prompt, else the
    ``action_input_shortlisting_agent`` of the last ``APIPlannerAgent`` step before it."""
    for p in _prompts(dp.target_step):
        m = _APP.search(_value(p))
        if m:
            return m.group(1)
    for s in reversed(steps_of(dp.prefix, "steps")):
        if step_name(s) != "APIPlannerAgent":
            continue
        for text in (s.get("data"), *(p.get("value") for p in reversed(_prompts(s)))):
            d = parse_json(text)
            short = d.get("action_input_shortlisting_agent") if isinstance(d, dict) else None
            if isinstance(short, dict) and isinstance(short.get("app_name"), str):
                return short["app_name"]
        break
    return None


def _shortlisted(output: Any, offered: dict[str, Candidate]) -> tuple[str, ...] | None:
    """The APIs a shortlister returned (``{"result": [{"name": ...}]}``), named as in its catalog."""
    items = output.get("result") if isinstance(output, dict) else output
    if not isinstance(items, list):
        return None
    names = [str(i.get("name") or i.get("api_name") or "") if isinstance(i, dict) else str(i) for i in items]
    return tuple(
        n if n in offered or not offered else next((k for k in offered if k.endswith(f"_{n}")), n)
        for n in names
        if n
    )


def _score_positive(raw: dict) -> bool:
    """CUGA's success: ``score > 0`` (see ``read_outcome``; a missing or unreadable score is a failure)."""
    return read_outcome(raw.get("score")) is True


def _see(pool: dict[str, Counter], key: str, entry: Candidate, n: int = 1) -> None:
    pool.setdefault(key, Counter())[entry] += n


def _best(entries: Counter) -> Candidate:
    """The description most runs gave (ties: the longest), the same whatever order the runs were read in."""
    return max(entries.items(), key=lambda kv: (kv[1], len(kv[0].description), kv[0].description))[0]


class CugaAdapter(GenericJsonAdapter):
    """CUGA trajectories -> decision points for any agent node; candidates and labels built in for the
    ``ShortlisterAgent`` and the ``TaskAnalyzerAgent`` (see the module docstring). A run succeeds when
    ``score > 0``; a missing or unreadable score counts as a failure and is reported in ``warnings``.
    ``app_labels`` is ``"used"`` (the apps the run called) or ``"offered"`` (the apps the analyzer chose)."""

    #: The decision pattern of each built-in decision, for ``onboard`` and ``scan`` when none is given.
    decision_kinds = {SHORTLISTER: "ranking", ANALYZER: "multilabel"}
    format_name = "CUGA trajectories"

    scan_notes = (
        "CUGA trajectories record no tokens, cost or latency per step (results.json holds per-task "
        "totals), so this scan ranks components by their LLM calls",
    )

    def __init__(
        self,
        catalog: Any = None,
        success: Callable[[dict], bool] | None = None,
        outcome_keys: Iterable[str] = ("score", "eval", "actions_count"),
        outcome_steps: Iterable[str] = ("EvaluationResult",),
        candidates_fn: Callable[[DecisionPoint], list[Candidate]] | None = None,
        labels_fn: Callable[[DecisionPoint, dict], set[str]] | None = None,
        task_keys: Iterable[str] | None = TASK_KEYS,
        app_labels: str = "used",
    ):
        if app_labels not in ("used", "offered"):
            raise ValueError(f"app_labels must be 'used' or 'offered', not {app_labels!r}")
        super().__init__(
            success=success or self._score,
            group_key="task_id",
            outcome_keys=outcome_keys,
            outcome_steps=outcome_steps,
            candidates_fn=candidates_fn,
            labels_fn=labels_fn,
            task_keys=task_keys,
        )
        self.target = SHORTLISTER
        self.catalog = _clean(parse_catalog(catalog))
        self.app_labels = app_labels

    def _reset(self) -> None:
        super()._reset()
        self._observed_apis: dict[str, Counter] = {}  # API -> its catalog entries in the loaded runs
        self._apps_seen: dict[str, Counter] = {}  # app -> its entries offered to the analyzer or called
        self._labels: dict[tuple[str, int], frozenset[str]] = {}  # (trajectory id, step) -> built-in labels
        self._step_calls: dict[str, list[frozenset]] = {}  # run id -> the API calls of each step, from load

    @property
    def _observed(self) -> dict[str, Candidate]:
        """The APIs offered to or called in the loaded runs."""
        return {k: _best(v) for k, v in sorted(self._observed_apis.items())}

    @property
    def _apps(self) -> dict[str, Candidate]:
        """The apps offered to the app analyzer or called in the loaded runs."""
        return {k: _best(v) for k, v in sorted(self._apps_seen.items())}

    # ------------------------------------------------------------------ loading
    _missing_outcome = "failed"

    @property
    def _outcome_fields(self) -> str:
        return "score"

    def _score(self, raw: dict) -> bool:
        """CUGA's success, ``score > 0``; a missing or unreadable score is a failure (see ``warnings``)."""
        if raw.get("score") is None:
            self._no_outcome += 1
            return False
        return self._read_outcome("score", raw["score"])

    def _is_run(self, record: Any) -> bool:
        """A CUGA run: a ``steps`` list with an agent node, or the fields of a CUGA trajectory (``intent``,
        ``task_id`` or ``score``; a run that stopped before any agent step). Another log with named steps
        (a pipeline configuration next to the runs, say) is not one."""
        if not (isinstance(record, dict) and isinstance(record.get(self.steps_key), list)):
            return False
        names = {step_name(s, self.step_name_key) for s in record[self.steps_key]}
        return bool(names & CUGA_STEPS) or any(k in record for k in ("intent", "task_id", "score"))

    def _group(self, raw: dict, rid: str, f: Path, single: bool) -> str:
        """The ``task_id``; without one (or ``"default"``), the file name (the run id for a run that shares
        its file)."""
        tid = str(raw.get(self.group_key) or "")
        if tid and tid != "default":
            return tid
        return f.stem if single else rid

    def _on_run(self, raw: dict, rid: str) -> None:
        self._step_calls[rid] = self._observe(raw, rid)

    def load(self, path: str | Path, *, lazy: bool | None = None) -> list[Trajectory]:
        """Every trajectory under ``path`` (recursive; ``results.json``/``metadata.json`` skipped); see
        ``GenericJsonAdapter.load``. A run is an object with a ``steps`` list."""
        return super().load(path, lazy=lazy)

    def _observe(self, raw: dict, rid: str = "") -> list[frozenset[tuple[str, str]]]:
        """Count the APIs and apps a run was offered (with their descriptions) or called (names only, counted
        0 times, so that any described entry wins); returns the ``(app, api)`` calls of each step. Calls
        whose app or API name is dynamic are counted in the load's notes."""
        out = []
        for s in steps_of(raw, self.steps_key):
            found, dynamic = _step_calls(s)
            calls = frozenset(found)
            out.append(calls)
            if dynamic and self.notes is not None:
                self.notes.add(
                    "dynamic_calls",
                    rid,
                    "calls",
                    n=dynamic,
                    level="info",
                    found="call_api calls whose app or API name is not a literal (nor a name assigned a "
                    "literal once), not counted in the labels",
                )
            for app, api in calls:
                _see(self._observed_apis, api, Candidate(api, api), 0)
                if app:
                    _see(self._apps_seen, app, Candidate(app, app), 0)
            name = step_name(s, self.step_name_key)
            if name == SHORTLISTER:
                for k, c in available_apis(s).items():
                    _see(self._observed_apis, k, c)
            elif name == ANALYZER:
                for k, c in available_apps(s).items():
                    _see(self._apps_seen, k, c)
        return out

    # ------------------------------------------------------------------ scan
    def llm_calls(self, traj: Trajectory) -> list[LLMCall]:
        """The LLM calls of a run, for ``tabagent.scan``: each step of an agent node (or with recorded
        prompts), one call per recorded generation (see the module docstring)."""
        out: list[LLMCall] = []
        for s in steps_of(traj.raw, self.steps_key):
            name = step_name(s, self.step_name_key)
            prompts = _prompts(s)
            if name is None or name in self.outcome_steps or not (name in CUGA_STEPS or prompts):
                continue
            gens = [p.get("value") for p in prompts if str(p.get("role", "")).lower() in _GENERATION_ROLES]
            raw_output = s.get("data") or (gens[-1] if gens else "")
            output = parse_json(raw_output)
            out.extend(LLMCall(name) for _ in gens[1:])  # earlier generations of the step (retries)
            usage = read_usage(s)
            if name == SHORTLISTER:
                offered = available_apis(s)
                choices = _shortlisted(output, offered)
                out.append(LLMCall(name, choices=choices, catalog=tuple(offered) or None, **usage))
            elif name == ANALYZER and (apps := available_apps(s)):
                chosen = tuple(sorted(chosen_apps(s))) or None
                out.append(LLMCall(name, choices=chosen, catalog=tuple(apps), **usage))
            elif name in FREE_TEXT_STEPS:
                text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, default=str)
                out.append(LLMCall(name, text_chars=len(text), **usage))
            else:
                fields, chars = output_fields(raw_output if output is None else output)
                out.append(LLMCall(name, fields=fields, text_chars=chars, **usage))
        return out

    # ------------------------------------------------------------------ the adapter protocol
    def decision_points(self, traj: Trajectory, target: str) -> list[DecisionPoint]:
        """See ``GenericJsonAdapter.decision_points``. For the built-in decisions the labels are read from
        the full run now and kept (a few names per decision point), so they need not read the run again."""
        dps = super().decision_points(traj, target)
        if dps and self.labels_fn is None and target in (SHORTLISTER, ANALYZER):
            steps = steps_of(traj.raw, self.steps_key)
            loaded = self._names.get(traj.id)  # the calls read at load, if this is the loaded run
            calls = self._step_calls.get(traj.id) if loaded is not None and loaded[0] is traj else None
            for dp in dps:
                self._labels[(traj.id, dp.step_index)] = frozenset(self._labels_in(steps, dp, calls))
        return dps

    def _require_builtin(self, dp: DecisionPoint) -> str:
        name = step_name(dp.target_step, self.step_name_key)
        if name not in (SHORTLISTER, ANALYZER):
            raise NotImplementedError(
                f"CugaAdapter builds candidates and labels for {SHORTLISTER} and {ANALYZER}; "
                f"pass candidates_fn= and labels_fn= to replace {name}"
            )
        return name

    def candidates(self, dp: DecisionPoint) -> list[Candidate]:
        if self.candidates_fn is not None:
            return self.candidates_fn(dp)
        if self._require_builtin(dp) == ANALYZER:
            apps = available_apps(dp.target_step) or self._apps
            return sorted(apps.values(), key=lambda c: c.id)
        offered = available_apis(dp.target_step)
        if not offered:  # CUGA API names are app-prefixed (<app>_<api>)
            pool, app = self.catalog or self._observed, decision_app(dp)
            offered = {k: c for k, c in pool.items() if app and k.startswith(f"{app}_")} or pool
        return sorted(offered.values(), key=lambda c: c.id)

    def _labels_in(self, steps: list[Any], dp: DecisionPoint, calls: list | None = None) -> set[str]:
        """The built-in labels of ``dp`` from its run's steps (``calls``: each step's API calls, if known)."""
        start = dp.step_index + 1

        def called(i: int) -> Any:
            return calls[i] if calls is not None and len(calls) == len(steps) else called_apis(steps[i])

        if self._require_builtin(dp) == ANALYZER:
            return {app for i in range(start, len(steps)) for app, _ in called(i) if app}
        out: set[str] = set()
        for i in range(start, len(steps)):
            if step_name(steps[i], self.step_name_key) == SHORTLISTER:
                break
            out |= {api for _, api in called(i)}
        return out

    def labels(self, dp: DecisionPoint) -> set[str]:
        """Shortlister: the APIs the CodeAgent called after this shortlisting and before the next one. App
        analyzer: the apps the run called after it (``app_labels="used"``), or the apps it chose
        (``"offered"``). Read from the full trajectory."""
        if self.labels_fn is not None:
            return super().labels(dp)
        if self._require_builtin(dp) == ANALYZER and self.app_labels == "offered":
            return chosen_apps(dp.target_step)
        kept = self._labels.get((dp.trajectory_id, dp.step_index))
        if kept is not None:
            return set(kept)
        return self._labels_in(steps_of(self._run_of(dp), self.steps_key), dp)
