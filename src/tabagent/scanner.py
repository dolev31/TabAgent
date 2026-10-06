"""Find the LLM calls in an agent's logs that TabAgent can replace.

    import tabagent
    report = tabagent.scan("./agent_logs")  # on the command line: tabagent scan ./agent_logs
    print(report.summary())                 # or report.to_markdown(), report.to_json()

``scan`` reads the logs with the adapter ``onboard`` would use and ranks every component that calls an LLM by

1. **cost**: its LLM calls per run and, where the logs record usage, its share of the agent's LLM tokens,
   latency and cost over the scanned runs ("not recorded" where they do not; ``prices=`` prices the recorded
   tokens);
2. **closed-set**: whether its outputs come from a fixed, recurring set of candidates (tools, applications,
   routes, labels), the decision type TabAgent replaces, with the evidence (chosen values, distinct values,
   catalog size, share of the choices inside the catalog). Free-text generation is not replaceable;
3. **data**: how many successful runs contain it, beside the paper's reference points (not a prediction);
4. **next step**: the ``tabagent onboard`` command for the top candidates.

Closed-set components come first, ordered by their share of the agent's LLM cost (else tokens, latency or
calls: the first measure the logs record), then components whose outputs the logs do not show, then the rest.

Adapters report LLM calls through an optional hook, ``llm_calls(trajectory) -> list[LLMCall]``, and may set
``scan_notes``. For an adapter without the hook, each decision point of each component it lists in
``components()`` counts as one LLM call, with no usage.

``check_logs`` diagnoses the logs themselves before an onboarding: which files
were read, skipped or repaired and why, the runs and their outcomes, and, for each decision, its decision
points, candidates and labels (chosen candidates that are not among the offered ones, with near-miss names),
the train/serve parity of its prefixes and a check that no prefix holds post-decision data. Each problem
says what is wrong, where (file, line, run, decision point) and how to fix it.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import textwrap
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tabagent.tabschema.trajectory import LLMCall, Trajectory

# The closed-set test. A component needs MIN_OUTPUTS calls with a readable output. With a recorded catalog, at
# least IN_CATALOG of its choices must lie inside the catalog offered to the call; without one, its choices
# must take at least two distinct values, at most MAX_DISTINCT distinct values per choice, and at least
# RECURRING of them must recur (their value occurs more than once in the logs).
MIN_OUTPUTS = 5
IN_CATALOG = 0.9
RECURRING = 0.8
MAX_DISTINCT = 0.5

PAPER = "TabAgent: Distilling LLM Agent Components into Cheap Classifiers"
PAPER_REFERENCE = (
    "TabHead reaches 90% of its full-pool P@R from 118 successful trajectories on AppWorld and from 37 to 51 "
    "on the four τ²-bench agents"
)
_ORDER = {"closed-set": 0, "unknown": 1, "open values": 2, "free text": 2}  # verdicts, best first
_MEASURES = (("tokens", "tokens"), ("latency", "latency_s"), ("cost", "cost_usd"))  # display order
_RANKING = ("cost", "tokens", "latency")  # the first one the logs record ranks the components
_RANKED_BY = {
    "cost": "share of the agent's LLM cost",
    "tokens": "share of the agent's LLM tokens",
    "latency": "share of the agent's LLM latency",
    "calls": "share of the agent's LLM calls (the logs record no tokens, latency or cost)",
}
_NOT_ONBOARDABLE = (NotImplementedError, KeyError, ValueError, TypeError)

Prices = tuple[float, float] | dict[str, tuple[float, float]]


@dataclass
class Measure:
    """One usage measure of a component over the scanned runs; None where the logs record none."""

    total: float | None = None
    per_call: float | None = None  # mean over the calls that record it
    share: float | None = None  # of the agent's total, over the calls that record it
    recorded_calls: int = 0


@dataclass
class Evidence:
    """What a component's outputs look like: the field read and how its values recur."""

    field: str  # "choices" (the decision the adapter reads) or a field of the component's output
    calls: int  # calls with this output
    values: int  # chosen values over those calls
    distinct: int
    recurring_share: float  # share of the chosen values whose value occurs more than once
    catalog_size: int | None  # distinct candidates offered; None: the logs record no catalog
    in_catalog_share: float | None  # share of the choices inside the catalog offered to that call
    top: list[tuple[str, int]]  # the most frequent values
    closed: bool


@dataclass
class ComponentScan:
    """One component that calls an LLM: its cost, the closed-set verdict, its data and the next step."""

    name: str
    verdict: str  # closed-set | unknown | open values | free text
    reason: str
    calls: int
    calls_per_run: float  # over all scanned runs
    runs: int  # runs with at least one call
    successful_runs: int
    tokens: Measure
    latency_s: Measure
    cost_usd: Measure
    call_share: float
    free_text_per_call: float  # characters of free text the calls generate, on average
    evidence: Evidence | None = None
    inferred_calls: int = 0  # calls counted from steps the logs do not mark as LLM calls
    decisions: list[str] = field(default_factory=list)  # the onboard decisions that replace its choice
    onboard_ready: bool = False  # the adapter builds candidates and labels for them
    next_steps: list[str] = field(default_factory=list)

    @property
    def replaceable(self) -> bool | None:
        """True for a closed-set component, None when the logs do not show its outputs, else False."""
        return True if self.verdict == "closed-set" else None if self.verdict == "unknown" else False

    def share(self, measure: str) -> float | None:
        """The component's share of the agent's ``cost``, ``tokens``, ``latency`` or ``calls``."""
        return self.call_share if measure == "calls" else getattr(self, dict(_MEASURES)[measure]).share


@dataclass
class ScanReport:
    """The ranked components of a :func:`scan`: ``summary()``, ``to_markdown()``, ``to_json()``."""

    logs: str
    adapter: str
    n_runs: int
    n_successful: int
    n_calls: int
    ranked_by: str  # cost | tokens | latency | calls
    cost_source: str | None  # "recorded", "your prices" or None (no cost)
    components: list[ComponentScan]
    notes: list[str] = field(default_factory=list)
    prices: Any = None
    top: int = 3

    @property
    def candidates(self) -> list[ComponentScan]:
        """The closed-set components, best first."""
        return [c for c in self.components if c.verdict == "closed-set"]

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict[str, Any]:
        """The report as a JSON-able dict."""
        comps = []
        for rank, c in enumerate(self.components, 1):
            d = asdict(c)
            d["replaceable"] = c.replaceable
            if c.evidence is not None:
                d["evidence"]["top"] = [[v, n] for v, n in c.evidence.top]
            comps.append({"rank": rank, **d})
        return {
            "logs": self.logs,
            "adapter": self.adapter,
            "runs": self.n_runs,
            "successful_runs": self.n_successful,
            "llm_calls": self.n_calls,
            "ranked_by": self.ranked_by,
            "cost_source": self.cost_source,
            "prices_usd_per_1m_tokens": self.prices,
            "components": comps,
            "paper_reference": {
                "text": PAPER_REFERENCE,
                "source": f"{PAPER}, efficiency section",
                "appworld_successful_trajectories": 118,
                "tau2_bench_agents_successful_trajectories": [37, 51],
            },
            "notes": self.notes,
        }

    def to_json(self, indent: int | None = 1) -> str:
        """The report as JSON (see :meth:`to_dict`)."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def _header(self) -> list[str]:
        return [
            f"Scanned {self.n_runs:,} runs ({self.n_successful:,} successful) with {self.adapter}: "
            f"{self.n_calls:,} LLM calls by {len(self.components)} components.",
            f"Ranked by {_RANKED_BY[self.ranked_by]}, closed-set components first.",
        ]

    def _rows(self) -> tuple[list[str], list[list[str]], list[str]]:
        """Table header, rows and the measures no component records."""
        absent = [m for m, attr in _MEASURES if all(getattr(c, attr).total is None for c in self.components)]
        cols = ["#", "component", "verdict", "calls/run"]
        cols += [f"{m}/call" for m, _ in _MEASURES if m not in absent]
        cols += [f"{self.ranked_by} share", "successful runs"]
        rows = []
        for i, c in enumerate(self.components, 1):
            row = [str(i), c.name, c.verdict, f"{c.calls_per_run:,.2f}"]
            row += [_per_call(m, getattr(c, attr)) for m, attr in _MEASURES if m not in absent]
            row += [_pct(c.share(self.ranked_by)), f"{c.successful_runs:,}"]
            rows.append(row)
        return cols, rows, absent

    def _details(self, c: ComponentScan) -> list[tuple[str, list[str]]]:
        """Labeled lines on one candidate: cost, evidence, data and next step."""
        shares, missing = [], []
        for m, attr in _MEASURES:
            meas = getattr(c, attr)
            if meas.total is None:
                missing.append(m)
            else:
                source = f" ({self.cost_source})" if m == "cost" else ""
                owner = "its" if shares else "the agent's"
                shares.append(f"{_pct(meas.share or 0.0)} of {owner} LLM {m}{source}")
        cost = [f"{c.calls_per_run:,.2f} LLM calls per run"]
        if shares:
            cost.append(_join(shares))
        if missing:
            cost.append(f"{_join(missing)} not recorded")
        evidence = [c.reason]
        if c.evidence is not None and c.evidence.top:
            top = ", ".join(f"{v} ({n:,})" for v, n in c.evidence.top)
            evidence.append(f"most frequent: {top}")
        if c.free_text_per_call:
            evidence.append(
                f"it also writes {c.free_text_per_call:,.0f} characters of free text per call, which a head "
                "does not replace"
            )
        data = [f"{c.successful_runs:,} successful runs contain it"]
        return [("cost", ["; ".join(cost)]), ("evidence", evidence), ("data", data), ("next", c.next_steps)]

    def _not_replaceable(self) -> list[str]:
        return [f"{c.name} ({c.verdict}): {c.reason}" for c in self.components if c.verdict != "closed-set"]

    def summary(self, width: int = 100) -> str:
        """The report as plain text for a terminal, prose wrapped at ``width`` characters."""

        def wrap(text: str, first: str = "", rest: str = "") -> list[str]:
            kw = {"break_long_words": False, "break_on_hyphens": False}
            return textwrap.wrap(text, width, initial_indent=first, subsequent_indent=rest, **kw) or [first]

        lines = [w for line in self._header() for w in wrap(line)]
        cols, rows, absent = self._rows()
        lines += ["", *_text_table(cols, rows)]
        if absent:
            lines += wrap(f"{_join(absent).capitalize()}: not recorded in these logs.")
        for i, c in enumerate(self.candidates[: self.top], 1):
            lines += ["", f"{i}. {c.name}: closed-set"]
            for label, texts in self._details(c):
                for j, t in enumerate(texts):
                    first = f"   {label if j == 0 else '':<9} "
                    code = label == "next" and (c.onboard_ready or j > 0)  # never wrapped, to copy and paste
                    lines += [first + t] if code else wrap(t, first, " " * 13)
        if not self.candidates:
            lines += ["", "No component's outputs come from a closed set in these logs."]
        other = self._not_replaceable()
        if other:
            lines += ["", "Not replaceable or unclear:"]
            lines += [w for o in other for w in wrap(o, "   ", "     ")]
        lines.append("")
        lines += wrap(
            f"Data reference points from the paper (not a prediction for your agent): {PAPER_REFERENCE}."
        )
        if self.notes:
            lines += ["", "Notes:"] + [w for n in self.notes for w in wrap(n, "   * ", "     ")]
        return "\n".join(lines) + "\n"

    def to_markdown(self) -> str:
        """The report as Markdown."""
        lines = ["# TabAgent scan", "", *self._header()]
        cols, rows, absent = self._rows()
        lines += ["", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        if absent:
            lines += ["", f"{_join(absent).capitalize()}: not recorded in these logs."]
        if self.candidates:
            lines += ["", "## Candidates"]
        for i, c in enumerate(self.candidates[: self.top], 1):
            lines += ["", f"### {i}. {c.name}", ""]
            for label, texts in self._details(c):
                if label == "next" and texts:
                    lang, code = ("bash", texts) if c.onboard_ready else ("python", texts[1:])
                    lines.append("* **next:**" + ("" if c.onboard_ready else f" {texts[0]}"))
                    if code:
                        lines += ["", f"  ```{lang}", *[f"  {t}" for t in code], "  ```"]
                else:
                    lines.append(f"* **{label}:** " + "; ".join(texts))
        other = self._not_replaceable()
        if other:
            lines += ["", "## Not replaceable or unclear", ""] + [f"* {o}" for o in other]
        lines += ["", "## Data reference points", ""]
        lines.append(f"From the paper (not a prediction for your agent): {PAPER_REFERENCE}.")
        if self.notes:
            lines += ["", "## Notes", ""] + [f"* {n}" for n in self.notes]
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------------- formatting
def _pct(share: float | None) -> str:
    if share is None:
        return "not recorded"
    if 0 < share < 0.01:
        return "<1%"
    return ">99%" if 0.995 <= share < 1 else f"{share:.0%}"


def _per_call(measure: str, m: Measure) -> str:
    if m.per_call is None:
        return "not recorded"
    if measure == "tokens":
        return f"{m.per_call:,.0f}"
    if measure == "latency":
        return f"{m.per_call:,.2f} s"
    return f"${m.per_call:.2g}" if m.per_call < 0.001 else f"${m.per_call:,.4f}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def _text_table(cols: list[str], rows: list[list[str]]) -> list[str]:
    left = {1, 2}  # component and verdict are text
    widths = [max(len(r[i]) for r in [cols, *rows]) for i in range(len(cols))]

    def fmt(r: list[str]) -> str:
        cells = [
            v.ljust(w) if i in left else v.rjust(w) for i, (v, w) in enumerate(zip(r, widths, strict=True))
        ]
        return "  ".join(cells).rstrip()

    return [fmt(cols), *map(fmt, rows)]


# --------------------------------------------------------------------------- the closed-set test
def _evaluate(name: str, outputs: list[tuple[tuple[str, ...], tuple[str, ...] | None]]) -> Evidence:
    """The evidence for one output field: ``outputs`` holds each call's values and offered catalog."""
    values = [v for vals, _ in outputs for v in vals]
    counts = Counter(values)
    n = len(values)
    recurring = sum(k for k in counts.values() if k > 1) / n if n else 0.0
    offered = [(vals, set(cat)) for vals, cat in outputs if cat]
    in_offered = sum(len(vals) for vals, _ in offered)
    in_catalog = sum(v in cat for vals, cat in offered for v in vals) / in_offered if in_offered else None
    catalog = set().union(*(cat for _, cat in offered)) if offered else None
    enough = len(outputs) >= MIN_OUTPUTS and n > 0
    if in_catalog is not None:
        closed = enough and in_catalog >= IN_CATALOG
    else:
        closed = enough and recurring >= RECURRING and 2 <= len(counts) <= MAX_DISTINCT * n
    size = None if catalog is None else len(catalog)
    return Evidence(
        name, len(outputs), n, len(counts), recurring, size, in_catalog, counts.most_common(5), closed
    )


def _evidence(calls: list[LLMCall]) -> Evidence | None:
    """The closed-set evidence of a component: its adapter-read choices, else its best output field."""
    chosen = [c for c in calls if c.choices is not None]
    if chosen and 2 * len(chosen) >= len(calls):
        return _evaluate("choices", [(c.choices or (), c.catalog) for c in chosen])
    present = Counter(k for c in calls for k in c.fields)
    options = [
        _evaluate(k, [(c.fields[k], None) for c in calls if k in c.fields])
        for k, n in present.items()
        if 2 * n >= len(calls)
    ]
    return max(options, key=lambda e: (e.closed, e.calls, e.recurring_share, e.distinct), default=None)


def _verdict(ev: Evidence | None, text_per_call: float) -> tuple[str, str]:
    """The verdict and its reason. A decision the adapter reads (``choices``) is judged on its choices; a
    component without one is free text when it writes text and no output field is closed-set."""
    what = "choices" if ev is not None and ev.field == "choices" else f"outputs in `{ev.field if ev else ''}`"
    if ev is not None and ev.closed:
        if ev.in_catalog_share is not None:
            return "closed-set", (
                f"{_pct(ev.in_catalog_share)} of its {ev.values:,} {what} are in the catalog offered to the "
                f"call ({ev.catalog_size:,} candidates); {ev.distinct:,} distinct values"
            )
        return "closed-set", (
            f"its {ev.values:,} {what} take {ev.distinct:,} distinct values and "
            f"{_pct(ev.recurring_share)} of them recur (no catalog recorded)"
        )
    if text_per_call > 0 and (ev is None or ev.field != "choices"):
        return "free text", (
            f"generates free text ({text_per_call:,.0f} characters per call); TabAgent replaces "
            "closed-set choices, not generation"
        )
    if ev is None or ev.values == 0:
        return "unknown", "the logs do not show what it chose"
    if ev.calls < MIN_OUTPUTS:
        return "unknown", f"too few calls with an output to tell ({ev.calls}; {MIN_OUTPUTS} needed)"
    if ev.distinct == 1:
        return "unknown", f"its {what} always take the same value in these logs: no choice to learn"
    if ev.in_catalog_share is not None:
        return "open values", f"only {_pct(ev.in_catalog_share)} of its {what} are in the offered catalog"
    return "open values", (
        f"its {ev.values:,} {what} take {ev.distinct:,} distinct values and only "
        f"{_pct(ev.recurring_share)} of them recur"
    )


# --------------------------------------------------------------------------- the scan
def _check_prices(prices: Any) -> Prices | None:
    def pair(p: Any) -> tuple[float, float]:
        if not (isinstance(p, (tuple, list)) and len(p) == 2 and all(isinstance(x, (int, float)) for x in p)):
            raise ValueError(f"a price is (USD per 1M input tokens, USD per 1M output tokens), not {p!r}")
        if min(p) < 0:
            raise ValueError("prices must not be negative")
        return float(p[0]), float(p[1])

    if prices is None:
        return None
    if isinstance(prices, dict):
        return {str(k): pair(v) for k, v in prices.items()}
    return pair(prices)


def _cost(call: LLMCall, prices: Prices | None) -> float | None:
    if prices is None:
        return call.cost_usd
    pin_pout = prices if isinstance(prices, tuple) else prices.get(call.model or "")
    if pin_pout is None or call.input_tokens is None or call.output_tokens is None:
        return None
    return (call.input_tokens * pin_pout[0] + call.output_tokens * pin_pout[1]) / 1e6


def _measure(values: list[float | None], grand: float) -> Measure:
    recorded = [v for v in values if v is not None]
    if not recorded:
        return Measure()
    total = math.fsum(recorded)  # exact, so every Python version reports the same numbers
    return Measure(total, total / len(recorded), total / grand if grand > 0 else None, len(recorded))


def _calls_of(hook: Any, traj: Trajectory, adapter: Any) -> list[LLMCall]:
    """The LLM calls of one run, naming the run when the adapter cannot read it."""
    from tabagent.adapters.logfiles import LogChangedError

    try:
        return hook(traj)
    except LogChangedError:
        raise
    except Exception as e:  # noqa: BLE001 - a run the adapter cannot read: say which one
        raise ValueError(
            f"{type(adapter).__name__} could not read the LLM calls of run {traj.id} ({traj.source}): "
            f"{type(e).__name__}: {e}"
        ) from e


def _from_decisions(adapter: Any, traj: Trajectory, names: list[str]) -> list[LLMCall]:
    """LLM calls of an adapter without the ``llm_calls`` hook: one per decision point, without usage."""
    out = []
    for name in names:
        for dp in adapter.decision_points(traj, name):
            choices = catalog = None
            try:
                choices = tuple(sorted(adapter.labels(dp)))
                catalog = tuple(c.id for c in adapter.candidates(dp))
            except _NOT_ONBOARDABLE:
                pass
            out.append(LLMCall(name, choices=choices, catalog=catalog, inferred=True))
    return out


def _onboardable(adapter: Any, trajs: list[Trajectory], decision: str) -> bool:
    """Whether ``onboard`` can build the decision from these logs: candidates and labels on one point."""
    for t in trajs[:20]:
        try:
            dps = adapter.decision_points(t, decision)
            if dps:
                adapter.candidates(dps[0])
                adapter.labels(dps[0])
                return True
        except _NOT_ONBOARDABLE:
            return False
    return False


def _next_steps(adapter: Any, logs: str, name: str, decisions: list[str], good: list[Trajectory]):
    if not good:
        return False, ["no successful run contains it; onboarding learns from successful runs"]
    ready = [d for d in decisions if _onboardable(adapter, good, d)]
    if ready:
        kinds = getattr(adapter, "decision_kinds", None) or {}
        return True, [
            f"tabagent onboard {shlex.quote(logs)} --decision {shlex.quote(d)}"
            + (f" --kind {kinds[d]}" if kinds.get(d, "ranking") != "ranking" else "")
            for d in ready
        ]
    cls = type(adapter).__name__
    return False, [
        f"{cls} builds no candidates or labels for {name}, so onboard it from Python with your own:",
        f"tabagent.onboard({logs!r}, decision={name!r}, adapter={cls}(candidates_fn=..., labels_fn=...))",
    ]


def scan(
    logs: str | Path | list[Trajectory],
    *,
    adapter: Any = "auto",
    prices: Prices | None = None,
    top: int = 3,
    lazy: bool | None = None,
    on_other_formats: str = "skip",
) -> ScanReport:
    """Rank the components of an agent's logs that call an LLM by what replacing them could save.

    Parameters
    ----------
    logs: a file or directory of agent logs, or a list of ``Trajectory`` (then pass ``adapter``).
    adapter: the log adapter, or ``"auto"`` to detect it from the files (as ``onboard`` does).
    prices: ``(USD per 1M input tokens, USD per 1M output tokens)`` for every call, or
        ``{model: (in, out)}`` by the model a call records. Calls are then priced from their recorded tokens
        instead of a recorded cost; without prices, a cost the logs record is used.
    top: how many closed-set candidates the summary details, with their next step.
    lazy: read runs again from their files when needed instead of holding them (default: from 256 MB of logs).
    on_other_formats: with ``adapter="auto"``, ``"skip"`` runs of another log format with a warning, or
        ``"error"`` (``MixedFormatsError``).

    Returns
    -------
    A :class:`ScanReport`: ``summary()``, ``to_markdown()``, ``to_dict()``, ``to_json()``.
    """
    prices = _check_prices(prices)
    if isinstance(logs, (str, Path)):
        if adapter == "auto":
            from tabagent.adapters import detect_adapter

            adapter = detect_adapter(logs, on_other_formats=on_other_formats)
        trajs = adapter.load(logs) if lazy is None else adapter.load(logs, lazy=lazy)
        where = str(logs)
    else:
        if adapter == "auto":
            raise ValueError("pass adapter= with a list of trajectories")
        trajs, where = list(logs), "LOGS"
    if not trajs:
        found = "; ".join(getattr(adapter, "warnings", [])[:3])
        raise ValueError(
            f"no agent runs found in {where} by {type(adapter).__name__}"
            + (f": {found}" if found else "")
            + " (tabagent check-logs LOGS says what was read and skipped)"
        )

    notes = list(getattr(adapter, "scan_notes", ()))
    hook = getattr(adapter, "llm_calls", None)
    if hook is not None:
        per_run = [_calls_of(hook, t, adapter) for t in trajs]
    else:
        names = list(adapter.components(trajs)) if hasattr(adapter, "components") else []
        per_run = [_from_decisions(adapter, t, names) for t in trajs]
        notes.append(
            f"{type(adapter).__name__} does not report LLM calls (no llm_calls hook): each decision point "
            "counts as one LLM call, and usage is not recorded"
        )
    notes += [f"data: {w}" for w in getattr(adapter, "warnings", [])]

    calls_of: dict[str, list[LLMCall]] = {}
    runs_of: dict[str, list[Trajectory]] = {}
    for t, calls in zip(trajs, per_run, strict=True):
        for c in calls:
            calls_of.setdefault(c.component, []).append(c)
        for name in dict.fromkeys(c.component for c in calls):
            runs_of.setdefault(name, []).append(t)
    every = [c for calls in per_run for c in calls]
    if not every:
        notes.append(f"{type(adapter).__name__} found no LLM calls in these runs")
    costs = {id(c): _cost(c, prices) for c in every}
    values = {
        "cost_usd": lambda c: costs[id(c)],
        "tokens": lambda c: c.tokens,
        "latency_s": lambda c: c.latency_s,
    }
    recorded = {attr: [v for c in every if (v := get(c)) is not None] for attr, get in values.items()}
    grand = {attr: math.fsum(v) for attr, v in recorded.items()}
    attr_of = dict(_MEASURES)
    ranked_by = next((m for m in _RANKING if grand[attr_of[m]] > 0), "calls")
    cost_source = ("your prices" if prices is not None else "recorded") if recorded["cost_usd"] else None

    comps = []
    for name, calls in calls_of.items():
        ev = _evidence(calls)
        text = sum(c.text_chars for c in calls) / len(calls)
        verdict, reason = _verdict(ev, text)
        good = [t for t in runs_of[name] if t.success]
        decisions = list(next((c.decisions for c in calls if c.decisions), (name,)))
        ready, steps = (False, [])
        if verdict == "closed-set":
            ready, steps = _next_steps(adapter, where, name, decisions, good)
        measures = {attr: _measure([get(c) for c in calls], grand[attr]) for attr, get in values.items()}
        comps.append(
            ComponentScan(
                name,
                verdict,
                reason,
                len(calls),
                len(calls) / len(trajs),
                len(runs_of[name]),
                len(good),
                measures["tokens"],
                measures["latency_s"],
                measures["cost_usd"],
                len(calls) / len(every),
                text,
                ev,
                sum(c.inferred for c in calls),
                decisions,
                ready,
                steps,
            )
        )
    comps.sort(key=lambda c: (_ORDER[c.verdict], -(c.share(ranked_by) or 0.0), -c.calls, c.name))

    if hook is not None and any(c.inferred for c in every):
        notes.append(
            "some runs do not mark which steps call an LLM (no usage, model or prompt fields): every named "
            "step of those runs counts as one call"
        )
    for m, attr in _MEASURES:
        if 0 < len(recorded[attr]) < len(every):
            k = len(recorded[attr])
            notes.append(f"{m} recorded for {k:,} of {len(every):,} LLM calls; shares use those calls")
    if prices is not None:
        notes.append(f"cost = recorded tokens x your prices {prices} (USD per 1M input, output tokens)")
    return ScanReport(
        where,
        type(adapter).__name__,
        len(trajs),
        sum(t.success for t in trajs),
        len(every),
        ranked_by,
        cost_source,
        comps,
        notes,
        prices,
        top,
    )


# --------------------------------------------------------------------------- check_logs
_LEVELS = ("error", "warning", "info")
_OTHER_SUFFIX_HINT = {
    ".gz": "decompress them first (gunzip)",
    ".zip": "unzip them first",
    ".txt": "save them as .json or .jsonl if they hold JSON",
    ".log": "save them as .jsonl if they hold one JSON object per line",
    ".csv": "TabAgent reads agent runs as JSON",
    ".yaml": "convert them to JSON",
    ".yml": "convert them to JSON",
}


@dataclass
class Finding:
    """One problem or note of a :func:`check_logs`: its level (``error`` stops onboarding), what is wrong
    and where, and how to fix it."""

    level: str  # error | warning | info
    message: str
    fix: str = ""

    def text(self) -> str:
        return f"{self.message}; {self.fix}" if self.fix and self.fix not in self.message else self.message


@dataclass
class DecisionCheck:
    """What :func:`check_logs` found for one decision."""

    name: str
    decision_points: int = 0
    runs: int = 0
    tasks: int = 0
    with_candidates: int = 0  # decision points with at least one candidate
    with_labels: int = 0  # decision points with at least one label
    labels_outside: int = 0  # decision points with a label that is not among their candidates
    none_offered: int = 0  # decision points with labels, none of them among their candidates
    outside: list[str] = field(default_factory=list)  # the first labels outside, with near misses
    parity_checked: int = 0
    parity_failed: list[str] = field(default_factory=list)  # decision points whose live prefix differs
    leak_checked: int = 0
    leak: str | None = None  # the adapter leak check's message
    leak_note: str | None = None  # why the leak check checked nothing, when it did not
    candidates_and_labels: bool = True  # whether the adapter builds them for this decision


@dataclass
class LogCheck:
    """The diagnosis of a log folder (see :func:`check_logs`): ``ok`` when nothing would stop onboarding;
    ``summary()``, ``to_dict()``, ``to_json()``."""

    logs: str
    adapter: str | None
    files: int
    runs: int
    successful: int
    components: dict[str, int]
    decisions: list[DecisionCheck]
    findings: list[Finding]

    @property
    def ok(self) -> bool:
        """Whether no finding would stop an onboarding."""
        return not any(f.level == "error" for f in self.findings)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def summary(self, width: int = 100) -> str:
        def wrap(text: str, first: str = "", rest: str = "") -> list[str]:
            kw = {"break_long_words": False, "break_on_hyphens": False}
            return textwrap.wrap(text, width, initial_indent=first, subsequent_indent=rest, **kw) or [first]

        reader = f" with {self.adapter}" if self.adapter else ""
        lines = wrap(
            f"Checked {self.logs}{reader}: {self.files:,} log files, {self.runs:,} runs "
            f"({self.successful:,} successful)."
        )
        if self.components:
            comps = ", ".join(f"{k} ({v:,})" for k, v in list(self.components.items())[:8])
            lines += wrap(f"Components in the successful runs (runs containing them): {comps}.")
        for d in self.decisions:
            lines += ["", f"{d.name}:"]
            lines += [w for t in _decision_lines(d) for w in wrap(t, "   ", "     ")]
        for level, title in (
            ("error", "Problems (onboarding would stop)"),
            ("warning", "Warnings"),
            ("info", "Notes"),
        ):
            found = [f for f in self.findings if f.level == level]
            if found:
                lines += ["", f"{title}:"] + [w for f in found for w in wrap(f.text(), "   * ", "     ")]
        lines.append("")
        if self.ok and self.decisions:
            d = self.decisions[0].name
            lines.append(  # never wrapped, to copy and paste
                f"Next: tabagent onboard {shlex.quote(self.logs)} --decision {shlex.quote(d)} --dry-run"
            )
        elif self.ok:
            lines += wrap("No problem found.")
        else:
            lines += wrap(
                f"{len(self.errors)} problem(s) would stop an onboarding; fix them and check again."
            )
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.summary()


def _pct_of(n: int, total: int) -> str:
    return _pct(n / total) if total else "0%"


def _decision_lines(d: DecisionCheck) -> list[str]:
    out = [f"{d.decision_points:,} decision points in {d.runs:,} successful runs of {d.tasks:,} tasks"]
    if not d.candidates_and_labels:
        out.append("the adapter builds no candidates or labels for it (pass candidates_fn= and labels_fn=)")
        return out
    if d.decision_points:
        out.append(
            f"candidates on {_pct_of(d.with_candidates, d.decision_points)}, labels on "
            f"{_pct_of(d.with_labels, d.decision_points)} of them; "
            + (
                "every label is among the offered candidates"
                if not d.labels_outside
                else f"{d.labels_outside:,} have a label outside the candidates"
            )
        )
        failed = len(d.parity_failed)
        out.append(
            f"live prefixes rebuilt exactly as in training on {d.parity_checked - failed:,} of "
            f"{d.parity_checked:,} checked"
        )
        if d.leak:
            out.append(d.leak)
        elif d.leak_checked:
            out.append(f"no post-decision data in prefixes ({d.leak_checked:,} checked)")
        else:
            out.append(f"the post-decision data check checked no decision point ({d.leak_note or 'not run'})")
    return out


def _near_miss(label: str, offered: set[str]) -> str | None:
    """An offered candidate the label probably means, best first: the same name in another case or spacing;
    the label with an app prefix (``id`` -> ``phone_id``) or without one; a candidate ending with the label
    (``phone_get_id``). Ties go to the first in sorted order."""

    def norm(x: str) -> str:
        return re.sub(r"[\W_]+", "", x.lower())

    def rank(c: str) -> int | None:
        if norm(c) == norm(label):
            return 0
        if c.partition("_")[2] == label or label.partition("_")[2] == c:
            return 1
        if c.endswith(f"_{label}") or label.endswith(f"_{c}"):
            return 2
        return None

    ranked = sorted((r, c) for c in offered if (r := rank(c)) is not None)
    return ranked[0][1] if ranked else None


def _other_files(root: Path) -> tuple[Counter, list[str]]:
    """Files below ``root`` that are not read as logs (other extensions), by extension, with examples."""
    counts: Counter = Counter()
    examples: list[str] = []
    if not root.is_dir():
        return counts, examples
    seen: set[str] = set()
    for dirpath, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            suffix = Path(name).suffix.lower()
            if name.startswith(".") or suffix in (".json", ".jsonl"):
                continue
            counts[suffix or "(none)"] += 1
            if len(examples) < 3:
                examples.append(Path(dirpath, name).relative_to(root).as_posix())
    return counts, examples


def _builtin_decisions(adapter: Any, comps: dict[str, int]) -> list[str]:
    targets = getattr(adapter, "targets", None)
    if targets:  # chat logs: every target is built from the runs with tool calls
        return list(targets) if comps else []
    kinds = getattr(adapter, "decision_kinds", None) or {}
    if (
        getattr(adapter, "candidates_fn", None) is not None
        and getattr(adapter, "labels_fn", None) is not None
    ):
        return list(comps)[:3]
    return [d for d in kinds if d in comps]


def _check_decision(adapter: Any, trajs: list[Trajectory], name: str, sample: int, out: list[Finding]):
    from tabagent.adapters.live import live_prefix, run_before
    from tabagent.adapters.logfiles import LogChangedError

    good = [t for t in trajs if t.success]
    d = DecisionCheck(name)
    dps_by_run = []
    for t in good:
        points = adapter.decision_points(t, name)
        if points:
            dps_by_run.append((t, points))
    dps = [(t, dp) for t, points in dps_by_run for dp in points]
    d.decision_points, d.runs = len(dps), len(dps_by_run)
    d.tasks = len({str(dp.group or dp.trajectory_id) for _, dp in dps})
    if not dps:
        out.append(
            Finding(
                "error",
                f"{name}: no decision point in the {len(good):,} successful runs",
                "check the decision name against the components listed above",
            )
        )
        return d
    if d.tasks < 2:
        out.append(
            Finding(
                "error",
                f"{name}: all {d.decision_points:,} decision points come from one task ({dps[0][1].group!r})",
                "onboarding needs successful runs of at least 2 tasks (a task id field per run)",
            )
        )
    n_check = min(sample, len(dps))  # evenly spread over the decision points, exactly ``sample`` of them
    chosen = {k * len(dps) // n_check for k in range(n_check)} if n_check else set()
    for k, (t, dp) in enumerate(dps):
        try:
            offered = {c.id for c in adapter.candidates(dp)}
            labels = {str(x) for x in adapter.labels(dp) or ()}
        except NotImplementedError:
            d.candidates_and_labels = False
            break
        d.with_candidates += bool(offered)
        d.with_labels += bool(labels)
        outside = sorted(labels - offered)
        if outside and offered:
            d.labels_outside += 1
            d.none_offered += not labels & offered
            if len(d.outside) < 5:
                hint = _near_miss(outside[0], offered)
                near = f" (offered: {hint!r})" if hint else ""
                d.outside.append(f"{dp.id} ({dp.source}): {outside[0]!r}{near}")
        if k in chosen:
            d.parity_checked += 1
            try:
                if live_prefix(adapter, run_before(adapter, t, dp), name) != dp.prefix:
                    d.parity_failed.append(dp.id)
            except LogChangedError:
                raise
            except Exception as e:  # noqa: BLE001 - reported as a failed parity
                d.parity_failed.append(f"{dp.id} ({type(e).__name__}: {e})")
    if not d.candidates_and_labels:
        out.append(
            Finding(
                "error",
                f"{name}: {type(adapter).__name__} builds no candidates or labels for it",
                "onboard it from Python with your own candidates_fn= and labels_fn=",
            )
        )
        return d
    n = d.decision_points
    if not d.with_candidates:
        out.append(
            Finding(
                "error",
                f"{name}: none of the {n:,} decision points has candidates (first: {dps[0][1].id})",
                "pass the tool catalog (--catalog FILE, tools= or catalog=), or keep it in the logs",
            )
        )
    elif d.with_candidates < n:
        out.append(
            Finding(
                "warning", f"{name}: {n - d.with_candidates:,} of {n:,} decision points have no candidates"
            )
        )
    if not d.with_labels:
        out.append(
            Finding(
                "error",
                f"{name}: none of the {n:,} decision points has a label (the chosen candidates)",
                "check that the successful runs record the choice (tool calls, API calls)",
            )
        )
    elif d.with_labels < n:
        out.append(
            Finding(
                "info",
                f"{name}: {n - d.with_labels:,} of {n:,} decision points have no label (nothing was chosen "
                "there); they count as decisions where no candidate is right",
            )
        )
    if d.labels_outside:
        stops = d.none_offered == d.with_labels  # no decision point offers any chosen candidate
        out.append(
            Finding(
                "error" if stops else "warning",
                f"{name}: on {d.labels_outside:,} of {n:,} decision points a chosen candidate is not among "
                f"the offered ones; it cannot be ranked (first: {'; '.join(d.outside[:3])})",
                "make the catalog and the logged choices use the same names (a catalog that lists every tool "
                "the agent can call)",
            )
        )
    if d.parity_failed:
        out.append(
            Finding(
                "error",
                f"{name}: the live prefix differs from the training prefix on {len(d.parity_failed):,} of "
                f"{d.parity_checked:,} checked decision points (first: {', '.join(d.parity_failed[:3])})",
                "the adapter must build live prefixes as in training; report this with the adapter's name",
            )
        )
    try:
        from tabagent.tabschema.verify import PrefixLeakError, check_adapter

        rec = check_adapter(adapter, good, [dp for _, dp in dps], name, label_of=adapter.labels)
        d.leak_checked = int(rec.get("checked", 0))
        if not d.leak_checked:
            reasons = rec.get("reasons") or {}
            d.leak_note = "; ".join(f"{r} ({n:,})" for r, n in reasons.items()) or "nothing to check"
            out.append(
                Finding(
                    "warning",
                    f"{name}: the check that no prefix holds post-decision data checked no decision point "
                    f"({d.leak_note})",
                    "onboarding runs the same check; if it cannot check this adapter either, review the "
                    "adapter's decision_points",
                )
            )
    except PrefixLeakError as e:
        d.leak = str(e)
        out.append(Finding("error", f"{name}: {e}"))
    except LogChangedError:
        raise
    except Exception as e:  # noqa: BLE001 - the leak check is a diagnosis: report, do not stop
        d.leak_note = f"it could not run: {type(e).__name__}: {e}"
        out.append(Finding("warning", f"{name}: the post-decision data check {d.leak_note}"))
    return d


def check_logs(
    logs: str | Path,
    *,
    adapter: Any = "auto",
    decision: str | None = None,
    sample: int = 200,
    lazy: bool | None = None,
    on_other_formats: str = "skip",
) -> LogCheck:
    """Diagnose agent logs before an onboarding; reads the logs only, calls no LLM.

    Reports which files were read, skipped or repaired (and why), the runs and their outcomes and, for the
    decision (default: every decision the adapter builds candidates and labels for), its decision points,
    candidates and labels (chosen candidates that are not among the offered ones, with near-miss names), the
    train/serve parity of its prefixes on ``sample`` decision points (spread evenly; all of them when there
    are fewer), and the adapter's check that no prefix holds post-decision data. ``LogCheck.ok`` is False when
    an onboarding would stop; every finding says what is wrong, where and how to fix it. ``lazy`` and
    ``on_other_formats`` are read as by ``onboard``. CLI: ``tabagent check-logs LOGS [--decision NAME]``.
    """
    from tabagent.adapters import detect_adapter
    from tabagent.adapters.logfiles import LogChangedError

    root = Path(logs)
    where = str(logs)
    findings: list[Finding] = []
    report = LogCheck(where, None, 0, 0, 0, {}, [], findings)
    if not root.exists():
        findings.append(Finding("error", f"{where}: no such file or directory", "check the path"))
        return report
    counts, examples = _other_files(root)
    if counts:
        hints = "; ".join(dict.fromkeys(_OTHER_SUFFIX_HINT[s] for s in counts if s in _OTHER_SUFFIX_HINT))
        kinds = ", ".join(f"{s} ({n:,})" for s, n in counts.most_common(5))
        findings.append(
            Finding(
                "warning" if any(s in (".gz", ".zip", ".log", ".txt") for s in counts) else "info",
                f"files that are not .json or .jsonl were not read: {kinds} (first: {', '.join(examples)})",
                hints,
            )
        )
    try:
        from tabagent.adapters.logfiles import iter_files

        report.files = len(iter_files(root))
        if adapter == "auto":
            adapter = detect_adapter(root, on_other_formats=on_other_formats)
        report.adapter = type(adapter).__name__
        trajs = adapter.load(root) if lazy is None else adapter.load(root, lazy=lazy)
    except (ValueError, OSError) as e:
        findings.append(Finding("error", str(e)))
        return report
    notes = getattr(adapter, "notes", None)
    issues = list(notes.issues.values()) if notes is not None else []
    noted = {i.message() for i in issues}
    findings += [Finding("info" if i.level == "detail" else i.level, i.message()) for i in issues]
    findings += [Finding("warning", w) for w in getattr(adapter, "warnings", []) if w not in noted]
    report.runs, report.successful = len(trajs), sum(t.success for t in trajs)
    if not trajs:
        findings.append(
            Finding(
                "error",
                f"no runs found in {where} by {report.adapter}",
                "see the warnings above for what was skipped, and the supported formats in the docs",
            )
        )
        return _sorted(report)
    good = [t for t in trajs if t.success]
    if not good:
        findings.append(
            Finding(
                "error",
                f"none of the {len(trajs):,} runs succeeded: onboarding learns from successful runs",
                "check the outcome field the adapter reads (success, reward or score; CUGA: score > 0), or "
                "pass success= to the adapter",
            )
        )
        return _sorted(report)
    report.components = comps = adapter.components(good) if hasattr(adapter, "components") else {}
    names = [decision] if decision else _builtin_decisions(adapter, comps)
    if decision and comps and decision not in comps and not getattr(adapter, "targets", None):
        findings.append(
            Finding(
                "error",
                f"no successful run has a {decision!r} step",
                f"choose one of the components: {', '.join(list(comps)[:8])}",
            )
        )
        return _sorted(report)
    if not names:
        findings.append(
            Finding(
                "info",
                f"{report.adapter} builds no candidates or labels for these components; onboard one from "
                "Python with candidates_fn= and labels_fn=",
            )
        )
    for name in names:
        try:
            report.decisions.append(_check_decision(adapter, trajs, name, sample, findings))
        except LogChangedError as e:  # the message says what changed and what to do
            findings.append(Finding("error", str(e)))
            break
        except ValueError as e:
            findings.append(Finding("error", f"{name}: {e}"))
    return _sorted(report)


def _sorted(report: LogCheck) -> LogCheck:
    report.findings.sort(key=lambda f: _LEVELS.index(f.level) if f.level in _LEVELS else len(_LEVELS))
    return report
