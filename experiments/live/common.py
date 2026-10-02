"""Shared pieces of the live experiments: the task split and the scaffold's state at each decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tabagent.adapters.openai_messages import (
    TOOL_SHORTLIST,
    _calls,
    assistant_turns,
    message_role,
    unwrap_message,
)
from tabagent.tabschema.trajectory import DecisionPoint, Trajectory


def split_tasks(trajs: list[Trajectory], held_out: float = 0.3, seed: int = 0) -> dict[str, list[str]]:
    """A seeded train / held-out split of the task groups, stratified by the group's prefix (the τ²-bench
    domain in ``<domain>:<task id>``): every run of a task falls on one side."""
    by: dict[str, list[str]] = {}
    for g in sorted({str(t.group) for t in trajs}):
        by.setdefault(g.split(":")[0], []).append(g)
    rng = np.random.default_rng(seed)
    train, test = [], []
    for d in sorted(by):
        gs = by[d]
        perm = rng.permutation(len(gs))
        n = round(held_out * len(gs))
        test += [gs[i] for i in perm[:n]]
        train += [gs[i] for i in perm[n:]]
    return {"seed": seed, "held_out": held_out, "train": sorted(train), "test": sorted(test)}


def load_split(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def run_fields(adapter: Any, run: dict[str, Any]) -> dict[str, Any]:
    """The run-level fields a scaffold holds during the run: everything but the messages and the outcome."""
    outcome = set(getattr(adapter, "outcome_keys", ()))
    return {k: v for k, v in run.items() if k != "messages" and k not in outcome}


def tool_turn_starts(messages: list[dict]) -> list[int]:
    """Index of the first message of every assistant turn that calls a tool (one LLM call each)."""
    msgs = [unwrap_message(m) for m in messages]
    out = []
    for turn in assistant_turns(msgs):
        items = [msgs[i] for i in turn]
        if any(
            m.get("type") == "function_call" or (message_role(m) == "assistant" and _calls(m)) for m in items
        ):
            out.append(turn[0])
    return out


def scaffold_states(
    adapter: Any, traj: Trajectory, target: str
) -> list[tuple[DecisionPoint, dict[str, Any]]]:
    """Every decision point of ``traj`` with the run as a scaffold held it right before the decision: the
    run-level fields and the messages before the assistant turn that made it (chat-message logs).

    Raises ``AssertionError`` when the adapter's decision points and the tool-calling turns disagree in
    number, which would make the pairing (and so every parity number) meaningless.
    """
    run = traj.raw
    msgs = [m for m in run.get("messages") or [] if isinstance(m, dict)]
    starts = tool_turn_starts(msgs)
    if target == TOOL_SHORTLIST:
        starts = starts[:1]
    dps = adapter.decision_points(traj, target)
    assert len(dps) == len(starts), f"{traj.id}: {len(dps)} decision points but {len(starts)} tool turns"
    fields = run_fields(adapter, run)
    return [(dp, {**fields, "messages": msgs[:j]}) for dp, j in zip(dps, starts, strict=True)]
