#!/usr/bin/env python3
"""Step 2: extract raw and normalized latency/overhead/energy components."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import COLLECTION_DATASET, OUTPUT_DIR, PROJECT_ROOT, TIERS, read_jsonl, write_jsonl

DEFAULT_COLLECTION = (
    COLLECTION_DATASET
)
DEFAULT_OUT = OUTPUT_DIR / "02_cost_components.jsonl"

END_KEYS = {"m5_a": "end_m5", "pi_a": "end_pi", "jetson": "end_jetson"}
RAW_KEYS = {
    "latency": ("latency_ms", "latency_ms"),
    "overhead": ("overhead_bytes", "overhead_bytes"),
    "energy": ("energy_j", "energy_j"),
}


def percentile_minmax(values: list[float], high_pct: float) -> tuple[float, float]:
    lo = float(np.min(values))
    hi = float(np.percentile(values, high_pct))
    if hi <= lo:
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def norm_value(v: float, lo: float, hi: float) -> float:
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = read_jsonl(args.collection)
    raw_values: dict[str, list[float]] = {name: [] for name in RAW_KEYS}
    skipped = Counter()

    for rec in records:
        for tier in TIERS:
            end = rec.get(END_KEYS[tier])
            if not isinstance(end, dict):
                skipped[f"missing_{END_KEYS[tier]}"] += 1
                continue
            for axis, (raw_key, _) in RAW_KEYS.items():
                value = end.get(raw_key)
                if value is None:
                    skipped[f"missing_{tier}_{raw_key}"] += 1
                    continue
                raw_values[axis].append(float(value))

    norm_ranges = {
        axis: {"lo": lo, "hi": hi, "high_percentile": args.clip_percentile}
        for axis, vals in raw_values.items()
        for lo, hi in [percentile_minmax(vals, args.clip_percentile)]
    }

    rows: list[dict[str, Any]] = []
    for rec in sorted(records, key=lambda r: r["image_id"]):
        row: dict[str, Any] = {"type": "cost_components", "image_id": rec["image_id"]}
        ok = True
        for tier in TIERS:
            end = rec.get(END_KEYS[tier])
            if not isinstance(end, dict):
                ok = False
                continue
            tier_cost: dict[str, float] = {}
            for axis, (raw_key, out_key) in RAW_KEYS.items():
                raw = float(end[raw_key])
                rng = norm_ranges[axis]
                tier_cost[out_key] = raw
                tier_cost[f"{axis}_norm"] = norm_value(raw, rng["lo"], rng["hi"])
            row[tier] = tier_cost
        if ok:
            rows.append(row)

    summary: dict[str, Any] = {
        "type": "summary",
        "step": "02_cost_components",
        "records": len(rows),
        "input": str(args.collection),
        "output": str(args.out),
        "skipped": dict(skipped),
        "normalization": {"method": "global min to percentile-clipped max", "ranges": norm_ranges},
        "raw_stats": {
            axis: {
                "min": round(float(np.min(vals)), 6),
                "p50": round(float(np.percentile(vals, 50)), 6),
                "p95": round(float(np.percentile(vals, 95)), 6),
                "max": round(float(np.max(vals)), 6),
            }
            for axis, vals in raw_values.items()
        },
        "tier_means": {},
    }
    for tier in TIERS:
        summary["tier_means"][tier] = {}
        for axis, (raw_key, _) in RAW_KEYS.items():
            vals = [float(r[tier][raw_key]) for r in rows]
            nvals = [float(r[tier][f"{axis}_norm"]) for r in rows]
            summary["tier_means"][tier][raw_key] = round(float(np.mean(vals)), 6)
            summary["tier_means"][tier][f"{axis}_norm"] = round(float(np.mean(nvals)), 6)
    return rows + [summary]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clip-percentile", type=float, default=95.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = build(args)
    write_jsonl(args.out, rows)
    summary = rows[-1]
    print("[DONE] cost components")
    print(f"  rows    : {summary['records']}")
    print(f"  skipped : {summary['skipped']}")
    print(f"  out     : {args.out}")


if __name__ == "__main__":
    main()
