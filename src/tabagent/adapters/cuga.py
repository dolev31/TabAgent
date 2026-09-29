"""CUGA agent trajectories (github.com/cuga-project/cuga-agent, Apache-2.0).

A CUGA trajectory file is ``{"intent", "task_id", "steps", "score", "eval", ...}``; each step names the agent
node that produced it (``TaskAnalyzerAgent``, ``PlanControllerAgent``, ``APIPlannerAgent``,
``ShortlisterAgent``, ``CodeAgent``, ``FinalAnswerAgent``, ...) and carries its ``prompts`` (role/value) and
``data`` (the node's output, usually JSON). Evaluated runs end with an ``EvaluationResult`` step.

The built-in decision is the API shortlister (the paper's CUGA tool shortlisting), ported from the original
``convert2tabular`` pipeline: at each ``ShortlisterAgent`` step the candidates are the APIs offered to it
(the ``Available APIs`` block of its prompt, else a catalog given here, e.g. CUGA API docs, else the APIs
seen for that app across the loaded runs), and the labels are the APIs the ``CodeAgent`` then called
(``call_api(app, api)``, app-prefixed as ``<app>_<api>``) before the next shortlisting. The prefix is the
trajectory truncated before the step, without ``score``, ``eval``, ``actions_count`` (a post-hoc count) and
the ``EvaluationResult`` step. Other components work with ``candidates_fn`` and ``labels_fn``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tabagent.adapters.common import SKIP_FILES, iter_files, parse_catalog, run_id
from tabagent.tabschema.trajectory import Candidate, DecisionPoint, GenericJsonAdapter, Trajectory, redact

SHORTLISTER = "ShortlisterAgent"
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
_AVAILABLE = re.compile(r'`Available APIs`:\s*"""(.*?)"""', re.S)
_APP = re.compile(r"`Current Application`:\s*([^\s`]+)")
_Q = r"""\\?["']"""  # a quote, possibly escaped when the code sits inside a JSON string
_CALL = re.compile(
    rf"call_api\(\s*(?:\w+\s*=\s*)?{_Q}([^\"'\\]+){_Q}\s*,\s*(?:\w+\s*=\s*)?{_Q}([^\"'\\]+){_Q}"
)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _json(text: Any) -> Any:
    """A step's ``data`` or prompt value as JSON (tolerating Markdown fences), or None."""
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    for s in (text, *_FENCE.findall(text)):
        try:
            return json.loads(s)
        except ValueError:
            continue
    return None


def _prefixed(app: str, api: str) -> str:
    return api if not app or api.startswith(f"{app}_") else f"{app}_{api}"


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
    for p in step.get("prompts") or []:
        m = _AVAILABLE.search(str(p.get("value", "")) if isinstance(p, dict) else "")
        if m:
            data = _json(m.group(1))
            if isinstance(data, (dict, list)):
                return _clean(parse_catalog(data))
    return {}


def called_apis(step: dict[str, Any]) -> set[tuple[str, str]]:
    """``(app, api)`` pairs a step called: ``call_api(app, api)`` in CodeAgent code, or an ``api_call``."""
    data = step.get("data") or ""
    if step.get("name") == "api_call":
        d = _json(data)
        if isinstance(d, dict) and d.get("function_name"):
            app = str(d.get("app_name") or "")
            return {(app, _prefixed(app, str(d["function_name"])))}
        return set()
    if step.get("name") != "CodeAgent":
        return set()
    d = _json(data)
    code = d.get("code", "") if isinstance(d, dict) else str(data)
    return {(app.strip(), _prefixed(app.strip(), api.strip())) for app, api in _CALL.findall(str(code))}


def decision_app(dp: DecisionPoint) -> str | None:
    """The app a shortlisting call is about: ``Current Application`` in the step's prompt, else the
    ``action_input_shortlisting_agent`` of the last ``APIPlannerAgent`` step before it."""
    for p in dp.target_step.get("prompts") or []:
        m = _APP.search(str(p.get("value", "")) if isinstance(p, dict) else "")
        if m:
            return m.group(1)
    for s in reversed(dp.prefix.get("steps", [])):
        if s.get("name") != "APIPlannerAgent":
            continue
        for text in (s.get("data"), *(p.get("value") for p in reversed(s.get("prompts") or []))):
            d = _json(text)
            if isinstance(d, dict) and isinstance(d.get("action_input_shortlisting_agent"), dict):
                return d["action_input_shortlisting_agent"].get("app_name")
        break
    return None


def _score_positive(raw: dict) -> bool:
    try:
        return float(raw.get("score") or 0) > 0
    except (TypeError, ValueError):
        return False


class CugaAdapter(GenericJsonAdapter):
    """CUGA trajectories -> decision points for any agent node; candidates and labels built in for the
    ``ShortlisterAgent`` (see the module docstring). A run succeeds when ``score > 0``."""

    def __init__(
        self,
        catalog: Any = None,
        success: Callable[[dict], bool] | None = None,
        outcome_keys: Iterable[str] = ("score", "eval", "actions_count"),
        outcome_steps: Iterable[str] = ("EvaluationResult",),
        candidates_fn: Callable[[DecisionPoint], list[Candidate]] | None = None,
        labels_fn: Callable[[DecisionPoint, dict], set[str]] | None = None,
    ):
        super().__init__(
            success=success or _score_positive,
            group_key="task_id",
            outcome_keys=outcome_keys,
            outcome_steps=outcome_steps,
            candidates_fn=candidates_fn,
            labels_fn=labels_fn,
        )
        self.target = SHORTLISTER
        self.catalog = _clean(parse_catalog(catalog))
        self._observed: dict[str, Candidate] = {}  # APIs offered to or called in the loaded runs

    def load(self, path: str | Path) -> list[Trajectory]:
        """Every trajectory file under ``path`` (recursive; ``results.json``/``metadata.json`` skipped)."""
        root = Path(path)
        out = []
        for f in iter_files(root, (".json",)):
            if f.name in SKIP_FILES:
                continue
            try:
                raw = json.loads(f.read_text())
            except ValueError:
                continue
            if not isinstance(raw, dict) or not isinstance(raw.get(self.steps_key), list):
                continue
            tid = str(raw.get(self.group_key) or "")
            group = tid if tid and tid != "default" else f.stem
            self._observe(raw)
            out.append(Trajectory(run_id(root, f), raw, bool(self.success(raw)), group, f.name))
        return out

    def _observe(self, raw: dict) -> None:
        for s in raw.get(self.steps_key, []):
            if not isinstance(s, dict):
                continue
            for _, api in called_apis(s):
                self._observed.setdefault(api, Candidate(api, api))
            if s.get(self.step_name_key) == SHORTLISTER:
                self._observed.update(available_apis(s))

    def _require_shortlister(self, dp: DecisionPoint) -> None:
        name = dp.target_step.get(self.step_name_key)
        if name != SHORTLISTER:
            raise NotImplementedError(
                f"CugaAdapter builds candidates and labels for {SHORTLISTER}; "
                f"pass candidates_fn= and labels_fn= to replace {name}"
            )

    def candidates(self, dp: DecisionPoint) -> list[Candidate]:
        if self.candidates_fn is not None:
            return self.candidates_fn(dp)
        self._require_shortlister(dp)
        offered = available_apis(dp.target_step)
        if not offered:  # CUGA API names are app-prefixed (<app>_<api>)
            pool, app = self.catalog or self._observed, decision_app(dp)
            offered = {k: c for k, c in pool.items() if app and k.startswith(f"{app}_")} or pool
        return sorted(offered.values(), key=lambda c: c.id)

    def labels(self, dp: DecisionPoint) -> set[str]:
        """APIs the CodeAgent called after this shortlisting and before the next one (full trajectory)."""
        if self.labels_fn is not None:
            return super().labels(dp)
        self._require_shortlister(dp)
        out: set[str] = set()
        for s in self._raw[dp.trajectory_id].get(self.steps_key, [])[dp.step_index + 1 :]:
            if s.get(self.step_name_key) == SHORTLISTER:
                break
            out |= {api for _, api in called_apis(s)}
        return out

    def live_prefix(self, raw: dict) -> dict[str, Any]:
        """The prefix of a live decision: the trajectory so far, redacted exactly as in training."""
        n = len(raw.get(self.steps_key, []))
        return redact(raw, self.steps_key, n, self.outcome_keys, self.outcome_steps, self.step_name_key)
