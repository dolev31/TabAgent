"""Replace CUGA's API shortlister and app matcher with onboarded heads (the paper's in-loop hot swap).

Onboard the heads from CUGA trajectories (``tabagent onboard logs/ --decision ShortlisterAgent`` and
``--decision TaskAnalyzerAgent``), then call them inside CUGA where the LLM was called. Both helpers take the
run so far as CUGA's ``ActivityTracker`` holds it (:func:`run_so_far`), which is the shape of the trajectory
files the heads were trained on, so the live prefix equals the training prefix. When a head defers (its
confidence gate, an extractor failure, an unreachable service), the helper returns None and CUGA keeps its
own LLM call::

    from tabagent.integrations.cuga import CugaAppMatcher, CugaShortlister, run_so_far

    shortlister = CugaShortlister("shortlister_head", k=9)  # a directory, hf://OWNER/REPO, or a serve URL
    app_matcher = CugaAppMatcher("app_head")

    # cuga/backend/cuga_graph/nodes/api/shortlister_agent/shortlister_agent.py, in ShortlisterAgent.run:
    result = shortlister.shortlist(run_so_far(tracker), input_variables.api_shortlister_all_filtered_apis)
    if result is not None:
        return AIMessage(content=ShortListerOutput(thoughts=[], result=result).model_dump_json())
    # ... CUGA's own LLM shortlisting, unchanged

    # cuga/backend/cuga_graph/nodes/task_decomposition_planning/analyze_task.py, in TaskAnalyzer.match_apps,
    # before match_apps_task.ainvoke:
    offered = [{"name": a.name, "description": a.description} for a in apps]
    names = app_matcher.match(run_so_far(tracker), offered)
    if names is not None:
        res = AppMatch(thoughts="", relevant_apps=names)

After a swap, CUGA records the shortlister's step without the LLM's prompts, so the prefixes of later
decisions in the same run differ from the training prefixes in those steps. The head's features read the
earlier steps' content; measure the swapped agent on held-out tasks before relying on it (the paper's
hot-swap protocol).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tabagent.adapters.common import parse_catalog
from tabagent.adapters.cuga import _clean
from tabagent.tabschema.trajectory import Candidate

__all__ = ["CugaAppMatcher", "CugaShortlister", "run_so_far"]


def run_so_far(tracker: Any) -> dict[str, Any]:
    """The run so far as CUGA's ``ActivityTracker`` holds it, in the shape of its trajectory files: the
    ``intent``, the ``dataset_name`` and the steps recorded up to now (pydantic ``Step`` objects or dicts)."""
    steps = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in getattr(tracker, "steps", [])]
    return {
        "intent": getattr(tracker, "intent", "") or "",
        "dataset_name": getattr(tracker, "dataset_name", "") or "",
        "steps": steps,
    }


def _load(head: Any, **kw: Any) -> Any:
    if isinstance(head, str) and head.startswith(("http://", "https://")):
        if any(v not in (None, False) for v in kw.values()):
            raise ValueError("a served head runs its extractor where it is served; set these on `serve`")
        from tabagent.serve import RemoteHead

        return RemoteHead(head)
    if isinstance(head, (str, Path)):
        from tabagent.api import TabAgentHead

        return TabAgentHead.load(head, **{k: v for k, v in kw.items() if v is not None})
    return head


class _Swap:
    def __init__(self, head: Any, defer_below: float | None, **load_kw: Any):
        self.head = _load(head, **load_kw)
        self.defer_below = defer_below
        self.last: dict[str, Any] | None = None  # the head's latest answer, for logging and monitoring

    def _decide(
        self, run: dict[str, Any], candidates: list[Candidate], k: int | None
    ) -> dict[str, Any] | None:
        self.last = self.head.decide(run=run, candidates=candidates, k=k, defer_below=self.defer_below)
        return None if self.last.get("deferred") else self.last


class CugaShortlister(_Swap):
    """CUGA's API shortlister as a head: the top ``k`` of the current app's APIs, as CUGA's ``APIDetails``.

    Parameters
    ----------
    head: a ``TabAgentHead`` onboarded on ``ShortlisterAgent``, its directory, ``hf://OWNER/REPO``, or the URL
        of a ``tabagent serve`` service.
    k: APIs to return (CUGA's shortlister usually returns 3 to 9).
    defer_below: the confidence gate (default: the head's calibrated threshold, if any).
    trust_extractor, sandbox, redact, allow_unpinned: when loading ``head`` from a directory, see
        ``TabAgentHead.load``.
    """

    def __init__(
        self,
        head: Any,
        k: int = 9,
        defer_below: float | None = None,
        trust_extractor: bool = False,
        sandbox: Any = None,
        redact: Any = None,
        allow_unpinned: bool = False,
    ):
        super().__init__(
            head,
            defer_below,
            trust_extractor=trust_extractor,
            sandbox=sandbox,
            redact=redact,
            allow_unpinned=allow_unpinned,
        )
        self.k = k

    @staticmethod
    def candidates(apis: Any) -> list[Candidate]:
        """The APIs offered to the shortlister, read as the adapter read them in training (the ``Available
        APIs`` block of the shortlister's prompt is this mapping as JSON)."""
        return sorted(_clean(parse_catalog(apis)).values(), key=lambda c: c.id)

    def shortlist(self, run: dict[str, Any], apis: Any) -> list[dict[str, Any]] | None:
        """``[{"name", "relevance_score", "reasoning"}]`` for CUGA's ``ShortListerOutput.result``, best first;
        None when the head defers (keep CUGA's LLM call)."""
        answer = self._decide(run, self.candidates(apis), self.k)
        if answer is None:
            return None
        return [
            {
                "name": c,
                "relevance_score": round(float(answer["scores"].get(c, 0.0)), 4),
                "reasoning": "ranked by the TabAgent head",
            }
            for c in answer["candidates"]
        ]


class CugaAppMatcher(_Swap):
    """CUGA's app matcher as a multi-label head: the apps a task needs, among those CUGA offers.

    Parameters
    ----------
    head: a ``TabAgentHead`` onboarded on ``TaskAnalyzerAgent`` (``kind="multilabel"``), its directory,
        ``hf://OWNER/REPO``, or the URL of a ``tabagent serve`` service.
    defer_below: the confidence gate (default: the head's calibrated threshold, if any).
    trust_extractor, sandbox, redact, allow_unpinned: when loading ``head`` from a directory.
    """

    def __init__(
        self,
        head: Any,
        defer_below: float | None = None,
        trust_extractor: bool = False,
        sandbox: Any = None,
        redact: Any = None,
        allow_unpinned: bool = False,
    ):
        super().__init__(
            head,
            defer_below,
            trust_extractor=trust_extractor,
            sandbox=sandbox,
            redact=redact,
            allow_unpinned=allow_unpinned,
        )

    @staticmethod
    def candidates(apps: list[Any]) -> list[Candidate]:
        """The apps CUGA offers (``{"name", "description"}`` dicts, or names), as in training."""
        out = {}
        for a in apps:
            name = a.get("name") if isinstance(a, dict) else str(a)
            if name:
                desc = a.get("description", "") if isinstance(a, dict) else ""
                out[str(name)] = Candidate(str(name), str(name), str(desc or ""))
        return sorted(out.values(), key=lambda c: c.id)

    def match(self, run: dict[str, Any], apps: list[Any]) -> list[str] | None:
        """The names of the apps the task needs (the head's multi-label decision, at least one); None when the
        head defers (keep CUGA's LLM call)."""
        answer = self._decide(run, self.candidates(apps), None)
        return None if answer is None else list(answer["candidates"])
