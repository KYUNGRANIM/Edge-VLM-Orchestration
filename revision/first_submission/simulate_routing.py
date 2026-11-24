#!/usr/bin/env python3
"""Simulate the exported online two-stage router over a collection file."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# this script lives at revision/first_submission/ -> parents[2] is the repository root
M5VLM_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_DIR = M5VLM_ROOT / "profiling" / "code" / "pipeline"
sys.path.insert(0, str(OFFLINE_DIR))

from common import (  # noqa: E402
    OUTPUT_DIR,
    POLICY_DIR,
    PROJECT_ROOT,
    read_jsonl,
    stage1_features,
    stage2_features,
    write_jsonl,
)

DEFAULT_COLLECTION = M5VLM_ROOT / "profiling" / "results" / "collection.jsonl"
DEFAULT_ORACLE = M5VLM_ROOT / "revision/first_submission/labels_before_severity_fix/03_oracle_labels_profiling.jsonl"
DEFAULT_OUT = M5VLM_ROOT / "revision/first_submission/simulation.jsonl"


def vectorize(feature_row: dict, keys: list[str]) -> list:
    return [feature_row[k] for k in keys]


def build(args: argparse.Namespace) -> list[dict]:
    stage1 = joblib.load(args.policy_dir / "stage1_m5_router.joblib")
    stage2 = joblib.load(args.policy_dir / "stage2_pi_router.joblib")
    clf1, keys1 = stage1["model"], stage1["feature_keys"]
    clf2, keys2 = stage2["model"], stage2["feature_keys"]

    records_map = {r["image_id"]: r for r in read_jsonl(args.collection)}
    oracle_map = {r["image_id"]: r for r in read_jsonl(args.oracle, "oracle_label")} if args.oracle.exists() else {}
    ids = sorted(records_map)
    rows = [{
        "type": "config",
        "step": "simulate_online_routing",
        "collection": str(args.collection),
        "oracle": str(args.oracle) if oracle_map else None,
        "policy_dir": str(args.policy_dir),
        "evaluation_mode": "same-data implementation check if oracle is provided; use a held-out collection for final deployment evaluation",
    }]

    preds = []
    matched_oracle = []
    for image_id in ids:
        rec = records_map[image_id]
        s1_pred = int(clf1.predict([vectorize(stage1_features(rec), keys1)])[0])
        if s1_pred == 1:
            pred = "m5_a"
        else:
            s2_pred = int(clf2.predict([vectorize(stage2_features(rec), keys2)])[0])
            pred = "pi_a" if s2_pred == 1 else "jetson"
        preds.append(pred)
        out = {"type": "prediction", "image_id": image_id, "pred_label": pred}
        if image_id in oracle_map:
            oracle = oracle_map[image_id]
            matched_oracle.append(oracle)
            out.update({
                "oracle_label": oracle["oracle_label"],
                "quality_satisfied": pred in oracle["feasible_tiers"],
                "quality_regret": round(max(oracle["quality"].values()) - float(oracle["quality"][pred]), 8),
                "selected_cost": oracle["raw_cost"][pred],
            })
        rows.append(out)

    summary = {
        "type": "summary",
        "step": "simulate_online_routing",
        "records": len(ids),
        "pred_dist": dict(Counter(preds)),
    }
    if matched_oracle:
        y = [r["oracle_label"] for r in matched_oracle]
        qsat = [p in r["feasible_tiers"] for p, r in zip(preds, matched_oracle, strict=True)]
        qreg = [max(r["quality"].values()) - float(r["quality"][p]) for p, r in zip(preds, matched_oracle, strict=True)]
        energy = [float(r["raw_cost"][p]["energy_j"]) for p, r in zip(preds, matched_oracle, strict=True)]
        summary.update({
            "oracle_dist": dict(Counter(y)),
            "accuracy": round(float(accuracy_score(y, preds)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 6),
            "quality_satisfaction": round(float(np.mean(qsat)), 6),
            "avg_quality_regret": round(float(np.mean(qreg)), 6),
            "avg_energy_j": round(float(np.mean(energy)), 6),
        })
    rows.append(summary)
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    ap.add_argument("--policy-dir", type=Path, default=M5VLM_ROOT / "revision/first_submission/routers")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = build(args)
    write_jsonl(args.out, rows)
    summary = rows[-1]
    print("[DONE] online simulation")
    print(f"  pred_dist : {summary['pred_dist']}")
    if "accuracy" in summary:
        print(f"  acc       : {summary['accuracy']}")
        print(f"  q_sat     : {summary['quality_satisfaction']}")
    print(f"  out       : {args.out}")


if __name__ == "__main__":
    main()
