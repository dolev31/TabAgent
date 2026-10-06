"""Live prefixes: build a live decision's prefix exactly as an adapter built the training prefixes.

A head scores the prefix its adapter produced in training: for chat-message logs, the run-level task fields
(``instruction``, ``domain``, ...) and the normalized messages; for CUGA and generic step logs, the run so far
without its outcome fields. A live decision must reach the extractor in the same shape, or the extractor
reads a field that is not there. ``describe`` records which adapter, with which prefix settings, a head was
trained with (``head.json``); ``from_description`` rebuilds it when the head is loaded, and ``live_prefix``
applies it to a live run.
"""

from __future__ import annotations

from typing import Any

from tabagent.adapters.cuga import CugaAdapter
from tabagent.adapters.openai_messages import OpenAIMessagesAdapter
from tabagent.adapters.tau2 import Tau2Adapter
from tabagent.tabschema.trajectory import DecisionPoint, GenericJsonAdapter, Trajectory, redact, steps_of

_STEP_KEYS = ("steps_key", "step_name_key", "outcome_keys", "outcome_steps")


def describe(adapter: Any) -> dict[str, Any] | None:
    """A JSON description of ``adapter``: its class and the settings that shape a prefix (None for None; a
    description passes through unchanged).

    The built-in adapters are described fully; another class is named, but :func:`from_description` cannot
    rebuild it, so a head trained with it needs the adapter passed again when it is loaded.
    """
    if adapter is None or isinstance(adapter, dict):
        return adapter
    cls = type(adapter)
    out: dict[str, Any] = {"class": cls.__name__, "module": cls.__module__}
    if cls is Tau2Adapter:
        out["config"] = {
            "granularity": adapter.granularity,
            "include_policy": adapter.include_policy,
            "max_chars": adapter.max_chars,
        }
    elif cls is OpenAIMessagesAdapter:
        out["config"] = {
            "granularity": adapter.granularity,
            "task_id_key": adapter.task_id_key,
            "task_keys": list(adapter.task_keys),
            "outcome_keys": list(adapter.outcome_keys),
            "max_chars": adapter.max_chars,
        }
    elif cls in (CugaAdapter, GenericJsonAdapter):
        cfg = {k: getattr(adapter, k) for k in (*_STEP_KEYS, "group_key", "task_keys")}
        out["config"] = {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}
        if cls is CugaAdapter:  # fixed by CugaAdapter
            del out["config"]["steps_key"], out["config"]["step_name_key"], out["config"]["group_key"]
    return out


def from_description(desc: dict[str, Any] | None) -> Any:
    """The adapter a :func:`describe` output names, or None when it is not a built-in adapter."""
    if not desc or "config" not in desc:
        return None
    classes = {c.__name__: c for c in (OpenAIMessagesAdapter, Tau2Adapter, CugaAdapter, GenericJsonAdapter)}
    cls = classes.get(desc.get("class", ""))
    return None if cls is None else cls(**desc["config"])


def live_prefix(adapter: Any, run: dict[str, Any], target: str | None = None) -> dict[str, Any]:
    """The prefix ``adapter`` builds for a live ``run`` (the run so far, in the logged format) before the
    decision ``target``.

    Chat-message adapters take ``{<task fields>, "messages": [...], "tools": [...]}``; CUGA and generic step
    adapters take the trajectory so far (``{<task fields>, "steps": [...]}``), redacted as in training.
    """
    if not isinstance(run, dict):
        raise TypeError(f"a live run is a dict in the format of the training logs, not {type(run).__name__}")
    if isinstance(adapter, OpenAIMessagesAdapter):
        return adapter.live_prefix(run, target if target in adapter.targets else None)
    if hasattr(adapter, "live_prefix"):
        return adapter.live_prefix(run)
    if isinstance(adapter, GenericJsonAdapter):
        a = adapter
        n = len(steps_of(run, a.steps_key))
        return redact(run, a.steps_key, n, a.outcome_keys, a.outcome_steps, a.step_name_key)
    raise TypeError(
        f"{type(adapter).__name__} has no live_prefix(run): build the prefix as in training and pass it to "
        "decide(prefix, ...)"
    )


def run_before(adapter: Any, traj: Trajectory, dp: DecisionPoint) -> dict[str, Any]:
    """The run of ``dp`` as the agent held it when the decision was made, in the logged format: what a live
    caller passes to :func:`live_prefix`. For chat-message adapters, the run's fields with the messages before
    the decision; for step adapters, the run with the steps before it. ``live_prefix(adapter, run_before(...),
    target) == dp.prefix`` is the train/serve parity every built-in adapter keeps (checked by
    ``tabagent.scanner.check_logs`` and the tests)."""
    raw = traj.raw
    if isinstance(adapter, OpenAIMessagesAdapter):
        messages = raw.get("messages") if isinstance(raw.get("messages"), list) else []
        base = {k: v for k, v in raw.items() if k != "messages"}

        def n_steps(m: int) -> int:
            return len(adapter._steps({**base, "messages": messages[:m]})[0])

        lo, hi = 0, len(messages)  # the most messages whose steps end before the decision step
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if n_steps(mid) <= dp.step_index:
                lo = mid
            else:
                hi = mid - 1
        return {**base, "messages": messages[:lo]}
    key = getattr(adapter, "steps_key", "steps")
    return {**raw, key: steps_of(raw, key)[: dp.step_index]}
