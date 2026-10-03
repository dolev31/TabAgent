"""A small Prometheus text exporter (exposition format 0.0.4) for the HTTP service, with no dependency.

Counters, gauges and histograms with labels, thread-safe, rendered by ``Registry.render``. Gauges can also be
read from a callback at scrape time (the worker pool's state, the drift report). When ``prometheus_client`` is
installed, ``render`` appends its default registry (process CPU, memory and file descriptors, Python GC, and
any metric the embedding application registered there).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable, Sequence

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
LabelValues = tuple[str, ...]
_INF = 'le="+Inf"'


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


def _labels(names: Sequence[str], values: Sequence[str], extra: str = "") -> str:
    parts = [f'{n}="{_escape(v)}"' for n, v in zip(names, values, strict=True)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help: str, labelnames: Sequence[str] = ()):
        self.name, self.help, self.labelnames = name, help, tuple(labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str]) -> LabelValues:
        if set(labels) != set(self.labelnames):
            raise ValueError(f"{self.name} takes labels {self.labelnames}, got {sorted(labels)}")
        return tuple(str(labels[n]) for n in self.labelnames)

    def header(self) -> list[str]:
        return [f"# HELP {self.name} {_escape(self.help)}", f"# TYPE {self.name} {self.kind}"]


class Counter(_Metric):
    """A monotonically increasing value per label set."""

    kind = "counter"

    def __init__(self, name: str, help: str, labelnames: Sequence[str] = ()):
        super().__init__(name, help, labelnames)
        self._values: dict[LabelValues, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(self._key(labels), 0.0)

    def lines(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        return [f"{self.name}{_labels(self.labelnames, k)} {_number(v)}" for k, v in items]


class Gauge(_Metric):
    """A value per label set, set directly or read from ``callback`` at scrape time (a callback returns
    ``{label values: value}``, or a number for a gauge without labels)."""

    kind = "gauge"

    def __init__(
        self,
        name: str,
        help: str,
        labelnames: Sequence[str] = (),
        callback: Callable[[], dict[LabelValues, float] | float | None] | None = None,
    ):
        super().__init__(name, help, labelnames)
        self._values: dict[LabelValues, float] = {}
        self.callback = callback

    def set(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)

    def lines(self) -> list[str]:
        if self.callback is not None:
            got = self.callback()
            if got is None:
                return []
            items = sorted(got.items()) if isinstance(got, dict) else [((), got)]
        else:
            with self._lock:
                items = sorted(self._values.items())
        return [f"{self.name}{_labels(self.labelnames, k)} {_number(v)}" for k, v in items if v is not None]


class Histogram(_Metric):
    """Observations counted into cumulative buckets per label set, with their sum and count."""

    kind = "histogram"

    def __init__(self, name: str, help: str, labelnames: Sequence[str] = (), buckets: Iterable[float] = ()):
        super().__init__(name, help, labelnames)
        self.buckets = tuple(sorted(float(b) for b in buckets))
        self._data: dict[LabelValues, list] = {}  # label values -> [bucket counts..., sum, count]

    def observe(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            data = self._data.setdefault(key, [0] * len(self.buckets) + [0.0, 0])
            for i, upper in enumerate(self.buckets):
                if value <= upper:
                    data[i] += 1
            data[-2] += value
            data[-1] += 1

    def count(self, **labels: str) -> int:
        data = self._data.get(self._key(labels))
        return data[-1] if data else 0

    def lines(self) -> list[str]:
        with self._lock:
            items = sorted((k, list(v)) for k, v in self._data.items())
        out = []
        for key, data in items:
            for upper, n in zip(self.buckets, data, strict=False):
                le = f'le="{_number(upper)}"'
                out.append(f"{self.name}_bucket{_labels(self.labelnames, key, le)} {n}")
            out.append(f"{self.name}_bucket{_labels(self.labelnames, key, _INF)} {data[-1]}")
            out.append(f"{self.name}_sum{_labels(self.labelnames, key)} {_number(data[-2])}")
            out.append(f"{self.name}_count{_labels(self.labelnames, key)} {data[-1]}")
        return out


class Registry:
    """The metrics of one service, rendered in the Prometheus text format."""

    def __init__(self, include_default: bool = True):
        self._metrics: list[_Metric] = []
        self.include_default = include_default

    def _add(self, metric: _Metric) -> _Metric:
        if any(m.name == metric.name for m in self._metrics):
            raise ValueError(f"metric {metric.name} is already registered")
        self._metrics.append(metric)
        return metric

    def counter(self, name: str, help: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._add(Counter(name, help, labelnames))

    def gauge(self, name: str, help: str, labelnames: Sequence[str] = (), callback=None) -> Gauge:
        return self._add(Gauge(name, help, labelnames, callback))

    def histogram(self, name: str, help: str, labelnames: Sequence[str] = (), buckets=()) -> Histogram:
        return self._add(Histogram(name, help, labelnames, buckets))

    def render(self) -> str:
        """Every metric in the text format, then ``prometheus_client``'s default registry if installed."""
        lines: list[str] = []
        for m in self._metrics:
            lines += m.header() + m.lines()
        text = "\n".join(lines) + "\n"
        if self.include_default:
            try:
                from prometheus_client import REGISTRY, generate_latest
            except ImportError:
                return text
            text += generate_latest(REGISTRY).decode("utf-8")
        return text
