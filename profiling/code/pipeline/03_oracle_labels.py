#!/usr/bin/env python3
"""Step 3: generate oracle labels with the primary balanced-d10 policy."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import CONFIG_DIR, OUTPUT_DIR, TIERS, load_config, read_jsonl, write_jsonl

DEFAULT_QUALITY = OUTPUT_DIR / "01_quality_components.jsonl"
DEFAULT_COST = OUTPUT_DIR / "02_cost_components.jsonl"
DEFAULT_CONFIG = CONFIG_DIR / "primary_balanced_d10.json"
DEFAULT_OUT = OUTPUT_DIR / "03_oracle_labels.jsonl"


def q_value(qrow: dict[str, Any], tier: str, weights: dict[str, float]) -> float:
    t = qrow[tier]
    return (
        weights["cat"] * float(t["cat_score"])
        + weights["sev"] * float(t["sev_score"])
        + weights["text"] * float(t["text_score"])
    )


def c_value(crow: dict[str, Any], tier: str, weights: dict[str, float]) -> float:
    t = crow[tier]
    return (
        weights["latency"] * float(t["latency_norm"])
        + weights["overhead"] * float(t["overhead_norm"])
        + weights["energy"] * float(t["energy_norm"])
    )


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = load_config(args.config)
    quality_weights = cfg["quality_weights"]
    cost_weights = cfg["cost_weights"]
    delta = float(cfg["quality_regret_delta"])

    qrows = read_jsonl(args.quality, "quality_components")
    crows = {r["image_id"]: r for r in read_jsonl(args.cost, "cost_components")}
    rows: list[dict[str, Any]] = []
    skipped = Counter()

    for qrow in qrows:
        image_id = qrow["image_id"]
        crow = crows.get(image_id)
        if crow is None:
            skipped["missing_cost"] += 1
            continue
        quality = {tier: q_value(qrow, tier, quality_weights) for tier in TIERS}
        cost = {tier: c_value(crow, tier, cost_weights) for tier in TIERS}
        q_best = max(quality.values())
        best_quality_tier = max(TIERS, key=lambda tier: quality[tier])
        feasible = [tier for tier in TIERS if q_best - quality[tier] <= delta + 1e-12]
        label = min(feasible, key=lambda tier: (cost[tier], -quality[tier]))
        rows.append({
            "type": "oracle_label",
            "image_id": image_id,
            "oracle_label": label,
            "best_quality_tier": best_quality_tier,
            "feasible_tiers": feasible,
            "quality_regret": round(float(q_best - quality[label]), 8),
            "selected_quality": round(float(quality[label]), 8),
            "selected_cost": round(float(cost[label]), 8),
            "quality": {tier: round(float(quality[tier]), 8) for tier in TIERS},
            "cost": {tier: round(float(cost[tier]), 8) for tier in TIERS},
            "raw_cost": {
                tier: {
                    "latency_ms": float(crow[tier]["latency_ms"]),
                    "overhead_bytes": float(crow[tier]["overhead_bytes"]),
                    "energy_j": float(crow[tier]["energy_j"]),
                }
                for tier in TIERS
            },
            "quality_components": {
                tier: {
                    "cat_score": float(qrow[tier]["cat_score"]),
                    "sev_score": float(qrow[tier]["sev_score"]),
                    "text_score": float(qrow[tier]["text_score"]),
                    "pred_cat": qrow[tier]["pred_cat"],
                    "pred_sev": qrow[tier]["pred_sev"],
                }
                for tier in TIERS
            },
            "gt": qrow["gt"],
        })

    dist = Counter(r["oracle_label"] for r in rows)
    q_regrets = [r["quality_regret"] for r in rows]
    summary = {
        "type": "summary",
        "step": "03_oracle_labels",
        "records": len(rows),
        "inputs": {"quality": str(args.quality), "cost": str(args.cost), "config": str(args.config)},
        "output": str(args.out),
        "operating_point": {
            "name": cfg["name"],
            "quality_weights": quality_weights,
            "cost_weights": cost_weights,
            "delta": delta,
            "rule": "minimum-cost feasible tier under quality-regret constraint",
        },
        "oracle_dist": dict(dist),
        "oracle_frac": {tier: round(dist[tier] / len(rows), 6) for tier in TIERS},
        "multi_feasible_count": sum(1 for r in rows if len(r["feasible_tiers"]) > 1),
        "multi_feasible_fraction": round(sum(1 for r in rows if len(r["feasible_tiers"]) > 1) / len(rows), 6),
        "avg_quality_regret": round(float(np.mean(q_regrets)), 8),
        "p95_quality_regret": round(float(np.percentile(q_regrets, 95)), 8),
        "avg_latency_ms": round(float(np.mean([r["raw_cost"][r["oracle_label"]]["latency_ms"] for r in rows])), 3),
        "avg_overhead_bytes": round(float(np.mean([r["raw_cost"][r["oracle_label"]]["overhead_bytes"] for r in rows])), 3),
        "avg_energy_j": round(float(np.mean([r["raw_cost"][r["oracle_label"]]["energy_j"] for r in rows])), 6),
        "skipped": dict(skipped),
    }
    return rows + [summary]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    ap.add_argument("--cost", type=Path, default=DEFAULT_COST)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = build(args)
    write_jsonl(args.out, rows)
    summary = rows[-1]
    print("[DONE] oracle labels")
    print(f"  rows     : {summary['records']}")
    print(f"  dist     : {summary['oracle_dist']}")
    print(f"  q_regret : {summary['avg_quality_regret']}")
    print(f"  energy   : {summary['avg_energy_j']} J")
    print(f"  out      : {args.out}")


if __name__ == "__main__":
    main()

