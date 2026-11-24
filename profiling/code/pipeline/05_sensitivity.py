#!/usr/bin/env python3
"""Step 5: robustness check across reasonable quality-weight/delta choices."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from common import (
    COLLECTION_DATASET,
    CONFIG_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    TIERS,
    load_config,
    make_dt,
    read_jsonl,
    stage1_features,
    stage2_features,
    to_matrix,
    write_jsonl,
)

DEFAULT_COLLECTION = (
    COLLECTION_DATASET
)
DEFAULT_QUALITY = OUTPUT_DIR / "01_quality_components.jsonl"
DEFAULT_COST = OUTPUT_DIR / "02_cost_components.jsonl"
DEFAULT_OUT = OUTPUT_DIR / "05_sensitivity.jsonl"

QUALITY_PROFILES = [
    ("fdr", {"cat": 0.148, "sev": 0.250, "text": 0.602}),
    ("text_heavy", {"cat": 0.20, "sev": 0.20, "text": 0.60}),
    ("balanced", {"cat": 0.30, "sev": 0.30, "text": 0.40}),
    ("structure", {"cat": 0.40, "sev": 0.40, "text": 0.20}),
    ("structure_max", {"cat": 0.50, "sev": 0.25, "text": 0.25}),
]
DELTAS = (0.05, 0.10)


def q_value(qrow: dict[str, Any], tier: str, weights: dict[str, float]) -> float:
    t = qrow[tier]
    return weights["cat"] * float(t["cat_score"]) + weights["sev"] * float(t["sev_score"]) + weights["text"] * float(t["text_score"])


def c_value(crow: dict[str, Any], tier: str, weights: dict[str, float]) -> float:
    t = crow[tier]
    return weights["latency"] * float(t["latency_norm"]) + weights["overhead"] * float(t["overhead_norm"]) + weights["energy"] * float(t["energy_norm"])


def rederive_oracle(
    qrows: list[dict[str, Any]],
    crows: dict[str, dict[str, Any]],
    quality_weights: dict[str, float],
    cost_weights: dict[str, float],
    delta: float,
) -> list[dict[str, Any]]:
    out = []
    for qrow in qrows:
        image_id = qrow["image_id"]
        crow = crows[image_id]
        quality = {tier: q_value(qrow, tier, quality_weights) for tier in TIERS}
        cost = {tier: c_value(crow, tier, cost_weights) for tier in TIERS}
        q_best = max(quality.values())
        feasible = [tier for tier in TIERS if q_best - quality[tier] <= delta + 1e-12]
        label = min(feasible, key=lambda tier: (cost[tier], -quality[tier]))
        out.append({
            "type": "oracle_label",
            "image_id": image_id,
            "oracle_label": label,
            "feasible_tiers": feasible,
            "quality_regret": round(float(q_best - quality[label]), 8),
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
                }
                for tier in TIERS
            },
        })
    return out


def cascade_dt_cv(records: list[dict[str, Any]], oracle_rows: list[dict[str, Any]], folds: int) -> list[str]:
    y = np.array([r["oracle_label"] for r in oracle_rows], dtype=object)
    y_s1 = np.array([int(label == "m5_a") for label in y])
    y_s2 = np.array([int(label == "pi_a") for label in y])
    s1_rows = [stage1_features(r) for r in records]
    s2_rows = [stage2_features(r) for r in records]
    x1, _, c1, n1 = to_matrix(s1_rows)
    x2, _, c2, n2 = to_matrix(s2_rows)
    preds = np.empty(len(records), dtype=object)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for tr, te in skf.split(np.zeros(len(y)), y):
        clf1 = make_dt(c1, n1, depth=4)
        clf1.fit([x1[i] for i in tr], y_s1[tr])
        s1_pred = clf1.predict([x1[i] for i in te])
        tr2 = [i for i in tr if y[i] != "m5_a"]
        clf2 = make_dt(c2, n2, depth=4)
        clf2.fit([x2[i] for i in tr2], y_s2[tr2])
        s2_pred = clf2.predict([x2[i] for i in te])
        for li, gi in enumerate(te):
            if int(s1_pred[li]) == 1:
                preds[gi] = "m5_a"
            elif int(s2_pred[li]) == 1:
                preds[gi] = "pi_a"
            else:
                preds[gi] = "jetson"
    return preds.tolist()


def metrics(name: str, config_name: str, preds: list[str], oracle_rows: list[dict[str, Any]], baseline_costs: dict[str, dict[str, float]]) -> dict[str, Any]:
    y = [r["oracle_label"] for r in oracle_rows]
    qr, qsat, lat, ovh, eng = [], [], [], [], []
    for pred, row in zip(preds, oracle_rows, strict=True):
        q_best = max(row["quality"].values())
        qr.append(q_best - float(row["quality"][pred]))
        qsat.append(pred in row["feasible_tiers"])
        rc = row["raw_cost"][pred]
        lat.append(float(rc["latency_ms"]))
        ovh.append(float(rc["overhead_bytes"]))
        eng.append(float(rc["energy_j"]))
    al, ao, ae = float(np.mean(lat)), float(np.mean(ovh)), float(np.mean(eng))

    def saving(v: float, base: float) -> float:
        return round(1.0 - v / base, 6) if base > 0 else 0.0

    return {
        "type": "config_method_summary",
        "oracle_config": config_name,
        "method": name,
        "accuracy": round(float(accuracy_score(y, preds)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 6),
        "quality_satisfaction": round(float(np.mean(qsat)), 6),
        "avg_quality_regret": round(float(np.mean(qr)), 6),
        "p95_quality_regret": round(float(np.percentile(qr, 95)), 6),
        "avg_latency_ms": round(al, 3),
        "avg_overhead_bytes": round(ao, 3),
        "avg_energy_j": round(ae, 6),
        "latency_saving_vs_jetson": saving(al, baseline_costs["jetson"]["latency_ms"]),
        "overhead_saving_vs_jetson": saving(ao, baseline_costs["jetson"]["overhead_bytes"]),
        "energy_saving_vs_jetson": saving(ae, baseline_costs["jetson"]["energy_j"]),
        "pred_dist": dict(Counter(preds)),
    }


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = load_config(args.config)
    cost_weights = cfg["cost_weights"]
    qrows = read_jsonl(args.quality, "quality_components")
    crows = {r["image_id"]: r for r in read_jsonl(args.cost, "cost_components")}
    records_map = {r["image_id"]: r for r in read_jsonl(args.collection)}
    ids = sorted(set(records_map) & {r["image_id"] for r in qrows} & set(crows))
    qrows = [next(r for r in qrows if r["image_id"] == i) for i in ids]
    records = [records_map[i] for i in ids]

    # Baseline costs are independent of oracle config.
    first_rows = rederive_oracle(qrows, crows, QUALITY_PROFILES[2][1], cost_weights, 0.10)
    baseline_costs = {
        tier: {
            "latency_ms": float(np.mean([float(r["raw_cost"][tier]["latency_ms"]) for r in first_rows])),
            "overhead_bytes": float(np.mean([float(r["raw_cost"][tier]["overhead_bytes"]) for r in first_rows])),
            "energy_j": float(np.mean([float(r["raw_cost"][tier]["energy_j"]) for r in first_rows])),
        }
        for tier in TIERS
    }

    rows: list[dict[str, Any]] = [{
        "type": "config",
        "step": "05_sensitivity",
        "n": len(ids),
        "inputs": {"collection": str(args.collection), "quality": str(args.quality), "cost": str(args.cost)},
        "output": str(args.out),
        "cost_weights_fixed": cost_weights,
        "quality_profiles": [{"name": name, "weights": w} for name, w in QUALITY_PROFILES],
        "deltas": list(DELTAS),
        "main_method": "cascade_dt_dt",
    }]

    summaries = []
    for profile_name, quality_weights in QUALITY_PROFILES:
        for delta in DELTAS:
            config_name = f"{profile_name}_d{int(delta * 100):02d}"
            oracle_rows = rederive_oracle(qrows, crows, quality_weights, cost_weights, delta)
            oracle_labels = [r["oracle_label"] for r in oracle_rows]
            rows.append({
                "type": "oracle_config",
                "name": config_name,
                "quality_weights": quality_weights,
                "cost_weights": cost_weights,
                "delta": delta,
                "oracle_dist": dict(Counter(oracle_labels)),
                "multi_feasible_count": sum(1 for r in oracle_rows if len(r["feasible_tiers"]) > 1),
            })
            cascade_preds = cascade_dt_cv(records, oracle_rows, args.folds)
            for name, preds in {
                "always_m5_a": ["m5_a"] * len(ids),
                "always_pi_a": ["pi_a"] * len(ids),
                "always_jetson": ["jetson"] * len(ids),
                "cascade_dt_dt": cascade_preds,
                "oracle": oracle_labels,
            }.items():
                row = metrics(name, config_name, preds, oracle_rows, baseline_costs)
                summaries.append(row)
                rows.append(row)

    for method_name in ("cascade_dt_dt",):
        method_rows = [r for r in summaries if r["method"] == method_name]
        rows.append({
            "type": "method_robustness",
            "method": method_name,
            "n_configs": len(method_rows),
            "accuracy": {
                "min": round(min(r["accuracy"] for r in method_rows), 6),
                "mean": round(float(np.mean([r["accuracy"] for r in method_rows])), 6),
                "max": round(max(r["accuracy"] for r in method_rows), 6),
            },
            "quality_satisfaction": {
                "min": round(min(r["quality_satisfaction"] for r in method_rows), 6),
                "mean": round(float(np.mean([r["quality_satisfaction"] for r in method_rows])), 6),
                "max": round(max(r["quality_satisfaction"] for r in method_rows), 6),
            },
            "energy_saving_vs_jetson": {
                "min": round(min(r["energy_saving_vs_jetson"] for r in method_rows), 6),
                "mean": round(float(np.mean([r["energy_saving_vs_jetson"] for r in method_rows])), 6),
                "max": round(max(r["energy_saving_vs_jetson"] for r in method_rows), 6),
            },
        })
    rows.append({"type": "summary", "step": "05_sensitivity", "n_configs": len(QUALITY_PROFILES) * len(DELTAS)})
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    ap.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    ap.add_argument("--cost", type=Path, default=OUTPUT_DIR / "02_cost_components.jsonl")
    ap.add_argument("--config", type=Path, default=CONFIG_DIR / "primary_balanced_d10.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--folds", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = build(args)
    write_jsonl(args.out, rows)
    robust = next(r for r in rows if r.get("type") == "method_robustness" and r["method"] == "cascade_dt_dt")
    print("[DONE] sensitivity v9")
    print(f"  acc min/mean/max   : {robust['accuracy']['min']} / {robust['accuracy']['mean']} / {robust['accuracy']['max']}")
    print(f"  qsat min/mean/max  : {robust['quality_satisfaction']['min']} / {robust['quality_satisfaction']['mean']} / {robust['quality_satisfaction']['max']}")
    print(f"  out                : {args.out}")


if __name__ == "__main__":
    main()
