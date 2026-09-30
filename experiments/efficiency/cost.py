"""The paper's per-decision cost formulas (App. C, "Cost estimation" and "CPU cost"; App. F, Table 15).

CPU heads (TabHead runs on CPU, no GPU), priced by vCPU-hours:

    C_cpu(t) = threads * t / 3600 * p_vCPU-h

with 4 threads (CatBoost's ``thread_count``) and p_vCPU-h = $0.063, the on-demand AWS EC2 r6i price
(r6i.xlarge, 4 vCPU at about $0.252 per hour). For the paper's 2.682 ms this gives 1.9e-7 dollars per
decision, which Table 15 rounds up to 2.0e-7.

GPU-served local models (the small LLM baselines), energy plus amortized GPU time on one V100-32GB:

    C_local(t) = (P / 1000) * (t / 3600) * PUE * p_kWh + (t / 3600) * p_GPU-h

with P = 250 W, PUE = 1.4, p_kWh = $0.20 and p_GPU-h = $0.20. API-hosted models are priced at the metered
provider charge, which is not recomputed here.

``t`` is the measured wall-clock time of one decision in seconds.

    python -m experiments.efficiency.cost                      # recompute the paper's cost table
    python -m experiments.efficiency.cost --cpu-ms 2.682 --gpu-s 0.658
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

CPU = {"threads": 4, "usd_per_vcpu_hour": 0.063}
GPU = {"power_w": 250.0, "pue": 1.4, "usd_per_kwh": 0.20, "usd_per_gpu_hour": 0.20}

#: Table 15 of the paper: one AppWorld shortlisting decision (seconds, reported $, pricing).
PAPER_TABLE = {
    "GPT-5.4 shortlister (API)": (7.50, 0.052, "api"),
    "Llama-3.1-8B-Instruct": (0.658, 4.9e-5, "gpu"),
    "Llama-3.2-3B-Instruct": (0.531, 4.0e-5, "gpu"),
    "Qwen2.5-7B-Instruct": (0.474, 3.6e-5, "gpu"),
    "Llama-3.2-1B-Instruct": (0.278, 2.1e-5, "gpu"),
    "TabAgent (CatBoost)": (0.002682, 2.0e-7, "cpu"),
}


def c_cpu(
    seconds: float, threads: int = CPU["threads"], usd_per_vcpu_hour: float = CPU["usd_per_vcpu_hour"]
) -> float:
    """C_cpu = threads * t / 3600 * p_vCPU-h (dollars per decision)."""
    return threads * seconds / 3600.0 * usd_per_vcpu_hour


def c_local(
    seconds: float,
    power_w: float = GPU["power_w"],
    pue: float = GPU["pue"],
    usd_per_kwh: float = GPU["usd_per_kwh"],
    usd_per_gpu_hour: float = GPU["usd_per_gpu_hour"],
) -> float:
    """C_local = (P/1000)(t/3600) PUE p_kWh + (t/3600) p_GPU-h (dollars per decision)."""
    hours = seconds / 3600.0
    return (power_w / 1000.0) * hours * pue * usd_per_kwh + hours * usd_per_gpu_hour


def price(seconds: float, pricing: str) -> float | None:
    """Cost of one decision under ``cpu`` or ``gpu`` pricing; ``api`` (metered) returns None."""
    if pricing == "cpu":
        return c_cpu(seconds)
    if pricing == "gpu":
        return c_local(seconds)
    if pricing == "api":
        return None
    raise ValueError("pricing must be cpu, gpu or api")


def check_paper_table() -> list[dict]:
    """Recompute every locally priced row of Table 15 from its reported runtime."""
    rows = []
    for name, (seconds, reported, pricing) in PAPER_TABLE.items():
        cost = price(seconds, pricing)
        rows.append(
            {
                "method": name,
                "seconds": seconds,
                "pricing": pricing,
                "reported_usd": reported,
                "recomputed_usd": cost,
                "rel_diff": None if cost is None else (cost - reported) / reported,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-ms", type=float, nargs="*", default=[], help="CPU per-decision latencies (ms)")
    ap.add_argument("--gpu-s", type=float, nargs="*", default=[], help="GPU per-decision latencies (s)")
    ap.add_argument("--threads", type=int, default=CPU["threads"])
    a = ap.parse_args(argv)
    if not (a.cpu_ms or a.gpu_s):
        print(f"{'method':28s} {'t (s)':>9s} {'pricing':>7s} {'reported $':>11s} {'recomputed $':>13s}")
        for r in check_paper_table():
            rec = "metered" if r["recomputed_usd"] is None else f"{r['recomputed_usd']:.3g}"
            cols = f"{r['seconds']:9.6g} {r['pricing']:>7s} {r['reported_usd']:11.3g} {rec:>13s}"
            print(f"{r['method']:28s} {cols}")
    for ms in a.cpu_ms:
        print(f"CPU {ms:g} ms x {a.threads} threads -> ${c_cpu(ms / 1000, a.threads):.3g} per decision")
    for s in a.gpu_s:
        print(f"GPU {s:g} s -> ${c_local(s):.3g} per decision")


if __name__ == "__main__":
    main()
