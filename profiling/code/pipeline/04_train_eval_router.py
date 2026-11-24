#!/usr/bin/env python3
"""Step 4: evaluate the main two-stage DT-DT cascade router."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from common import (
    COLLECTION_DATASET,
    OUTPUT_DIR,
    PROJECT_ROOT,
    TIERS,
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
DEFAULT_ORACLE = OUTPUT_DIR / "03_oracle_labels.jsonl"
DEFAULT_OUT = OUTPUT_DIR / "04_routing_main.jsonl"


def cascade_dt_cv(records: list[dict[str, Any]], oracle_rows: list[dict[str, Any]], folds: int) -> tuple[list[str], dict[str, Any]]:
    y = np.array([r["oracle_label"] for r in oracle_rows], dtype=object)
    y_s1 = np.array([int(label == "m5_a") for label in y])
    y_s2 = np.array([int(label == "pi_a") for label in y])
    s1_rows = [stage1_features(r) for r in records]
    s2_rows = [stage2_features(r) for r in records]
    x1, _, c1, n1 = to_matrix(s1_rows)
    x2, _, c2, n2 = to_matrix(s2_rows)
    preds = np.empty(len(records), dtype=object)
    fold_stats = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
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
        fold_stats.append({
            "fold": fold,
            "stage1_balanced_accuracy": round(float(balanced_accuracy_score(y_s1[te], s1_pred)), 6),
            "stage2_balanced_accuracy": round(float(balanced_accuracy_score(y_s2[te], s2_pred)), 6),
        })
    return preds.tolist(), {"stage1_model": "decision_tree_depth4", "stage2_model": "decision_tree_depth4", "folds": folds, "fold_stats": fold_stats}


def evaluate(name: str, preds: list[str], oracle_rows: list[dict[str, Any]], baseline_costs: dict[str, dict[str, float]]) -> dict[str, Any]:
    y = [r["oracle_label"] for r in oracle_rows]
    q_regrets, feasible_hits, latency, overhead, energy = [], [], [], [], []
    for pred, row in zip(preds, oracle_rows, strict=True):
        q_best = max(row["quality"].values())
        q_regrets.append(q_best - float(row["quality"][pred]))
        feasible_hits.append(pred in row["feasible_tiers"])
        rc = row["raw_cost"][pred]
        latency.append(float(rc["latency_ms"]))
        overhead.append(float(rc["overhead_bytes"]))
        energy.append(float(rc["energy_j"]))

    avg_latency = float(np.mean(latency))
    avg_overhead = float(np.mean(overhead))
    avg_energy = float(np.mean(energy))

    def saving(v: float, base: float) -> float:
        return round(1.0 - v / base, 6) if base > 0 else 0.0

    return {
        "type": "method_summary",
        "method": name,
        "accuracy": round(float(accuracy_score(y, preds)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 6),
        "macro_f1": round(float(f1_score(y, preds, labels=list(TIERS), average="macro", zero_division=0)), 6),
        "quality_satisfaction": round(float(np.mean(feasible_hits)), 6),
        "avg_quality_regret": round(float(np.mean(q_regrets)), 6),
        "p95_quality_regret": round(float(np.percentile(q_regrets, 95)), 6),
        "avg_latency_ms": round(avg_latency, 3),
        "avg_overhead_bytes": round(avg_overhead, 3),
        "avg_energy_j": round(avg_energy, 6),
        "latency_saving_vs_jetson": saving(avg_latency, baseline_costs["jetson"]["latency_ms"]),
        "overhead_saving_vs_jetson": saving(avg_overhead, baseline_costs["jetson"]["overhead_bytes"]),
        "energy_saving_vs_jetson": saving(avg_energy, baseline_costs["jetson"]["energy_j"]),
        "pred_dist": dict(Counter(preds)),
    }


def confusion(method: str, preds: list[str], oracle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = [r["oracle_label"] for r in oracle_rows]
    mat = confusion_matrix(y, preds, labels=list(TIERS))
    return {"type": "confusion", "method": method, "labels": list(TIERS), "matrix": mat.tolist()}


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    collection = {r["image_id"]: r for r in read_jsonl(args.collection)}
    oracle_map = {r["image_id"]: r for r in read_jsonl(args.oracle, "oracle_label")}
    ids = sorted(set(collection) & set(oracle_map))
    records = [collection[i] for i in ids]
    oracle_rows = [oracle_map[i] for i in ids]
    baseline_costs = {
        tier: {
            "latency_ms": float(np.mean([float(r["raw_cost"][tier]["latency_ms"]) for r in oracle_rows])),
            "overhead_bytes": float(np.mean([float(r["raw_cost"][tier]["overhead_bytes"]) for r in oracle_rows])),
            "energy_j": float(np.mean([float(r["raw_cost"][tier]["energy_j"]) for r in oracle_rows])),
        }
        for tier in TIERS
    }

    rows: list[dict[str, Any]] = [{
        "type": "config",
        "step": "04_train_eval_router",
        "n": len(ids),
        "inputs": {"collection": str(args.collection), "oracle": str(args.oracle)},
        "output": str(args.out),
        "router": {
            "name": "two_stage_cascade_dt_dt",
            "stage1": "M5 features -> END@M5 vs ESCALATE@Pi",
            "stage2": "M5+Pi features -> END@Pi vs ESCALATE@Jetson",
            "online_feature_groups": ["category", "severity", "confidence", "agreement"],
            "excluded": ["gt_labels", "gpt_judge_scores", "future_tier_outputs", "text_content_heuristics"],
        },
        "baseline_costs": baseline_costs,
    }]

    preds_by_method = {
        "always_m5_a": ["m5_a"] * len(ids),
        "always_pi_a": ["pi_a"] * len(ids),
        "always_jetson": ["jetson"] * len(ids),
        "oracle": [r["oracle_label"] for r in oracle_rows],
    }
    cascade_preds, meta = cascade_dt_cv(records, oracle_rows, args.folds)
    preds_by_method["cascade_dt_dt"] = cascade_preds
    rows.append({"type": "model_meta", "method": "cascade_dt_dt", **meta})

    for method, preds in preds_by_method.items():
        rows.append(evaluate(method, preds, oracle_rows, baseline_costs))
        rows.append(confusion(method, preds, oracle_rows))

    for image_id, pred in zip(ids, cascade_preds, strict=True):
        row = oracle_map[image_id]
        rows.append({
            "type": "prediction",
            "method": "cascade_dt_dt",
            "image_id": image_id,
            "oracle_label": row["oracle_label"],
            "pred_label": pred,
            "quality_satisfied": pred in row["feasible_tiers"],
            "quality_regret": round(max(row["quality"].values()) - float(row["quality"][pred]), 8),
            "selected_cost": row["raw_cost"][pred],
        })

    rows.append({
        "type": "summary",
        "step": "04_train_eval_router",
        "n": len(ids),
        "oracle_dist": dict(Counter(r["oracle_label"] for r in oracle_rows)),
        "main_method": "cascade_dt_dt",
    })
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--folds", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = build(args)
    write_jsonl(args.out, rows)
    main_row = next(r for r in rows if r.get("type") == "method_summary" and r["method"] == "cascade_dt_dt")
    print("[DONE] routing main")
    print(f"  acc     : {main_row['accuracy']}")
    print(f"  q_sat   : {main_row['quality_satisfaction']}")
    print(f"  energy  : {main_row['avg_energy_j']} J")
    print(f"  out     : {args.out}")


if __name__ == "__main__":
    main()
