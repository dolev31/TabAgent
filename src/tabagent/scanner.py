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
"""

from __future__ import annotations

import json
import math
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

    Returns
    -------
    A :class:`ScanReport`: ``summary()``, ``to_markdown()``, ``to_dict()``, ``to_json()``.
    """
    prices = _check_prices(prices)
    if isinstance(logs, (str, Path)):
        if adapter == "auto":
            from tabagent.adapters import detect_adapter

            adapter = detect_adapter(logs)
        trajs = adapter.load(logs)
        where = str(logs)
    else:
        if adapter == "auto":
            raise ValueError("pass adapter= with a list of trajectories")
        trajs, where = list(logs), "LOGS"
    if not trajs:
        raise ValueError(f"no agent runs found in {where}")

    notes = list(getattr(adapter, "scan_notes", ()))
    hook = getattr(adapter, "llm_calls", None)
    if hook is not None:
        per_run = [hook(t) for t in trajs]
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
