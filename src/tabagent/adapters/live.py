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
from tabagent.tabschema.trajectory import GenericJsonAdapter, redact

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
        cfg = {k: getattr(adapter, k) for k in _STEP_KEYS}
        out["config"] = {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}
        if cls is CugaAdapter:
            del out["config"]["steps_key"], out["config"]["step_name_key"]  # fixed by CugaAdapter
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
    if isinstance(adapter, OpenAIMessagesAdapter):
        return adapter.live_prefix(run, target if target in adapter.targets else None)
    if hasattr(adapter, "live_prefix"):
        return adapter.live_prefix(run)
    if isinstance(adapter, GenericJsonAdapter):
        a = adapter
        n = len(run.get(a.steps_key, []))
        return redact(run, a.steps_key, n, a.outcome_keys, a.outcome_steps, a.step_name_key)
    raise TypeError(
        f"{type(adapter).__name__} has no live_prefix(run): build the prefix as in training and pass it to "
        "decide(prefix, ...)"
    )
