#!/usr/bin/env python3
"""Export the final two-stage router policy for implementation/simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import export_text

# this script lives at profiling/code/pipeline/ -> parents[3] is the repository root
M5VLM_ROOT = Path(__file__).resolve().parents[3]
OFFLINE_DIR = M5VLM_ROOT / "profiling" / "code" / "pipeline"
sys.path.insert(0, str(OFFLINE_DIR))

from common import (  # noqa: E402
    OUTPUT_DIR,
    POLICY_DIR,
    PROJECT_ROOT,
    make_dt,
    read_jsonl,
    stage1_features,
    stage2_features,
    to_matrix,
    write_jsonl,
)

DEFAULT_COLLECTION = M5VLM_ROOT / "profiling" / "results" / "collection.jsonl"
DEFAULT_ORACLE = OUTPUT_DIR / "03_oracle_labels.jsonl"
DEFAULT_OUT = POLICY_DIR / "export_summary.jsonl"


def tree_text(pipe, keys: list[str]) -> str:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    try:
        feature_names = [str(v) for v in pre.get_feature_names_out(keys)]
    except Exception:
        feature_names = [f"x{i}" for i in range(clf.n_features_in_)]
    return export_text(clf, feature_names=feature_names, decimals=4)


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def stage2_pi_b_features(rec: dict) -> dict:
    """Stage 2 features for the implementation path that runs Pi_B only."""
    out = stage1_features(rec)
    pb = rec["pi_b"]
    db = rec.get("pi_b_conf_detail") or {}
    pb_sev = _safe_int(pb.get("sev"))
    out.update({
        "pib_cat": pb.get("cat", ""),
        "pib_sev": pb_sev,
        "pib_conf": _safe_int(pb.get("conf")),
        "pib_min_logprob": _safe_float(db.get("min_logprob"), -5.0),
        "m5a_pib_cat_match": int(rec["m5_a"].get("cat") == pb.get("cat")),
        "m5a_pib_sev_diff": abs(_safe_int(rec["m5_a"].get("sev")) - pb_sev),
    })
    return out


def build(args: argparse.Namespace) -> list[dict]:
    records_map = {r["image_id"]: r for r in read_jsonl(args.collection)}
    oracle_map = {r["image_id"]: r for r in read_jsonl(args.oracle, "oracle_label")}
    ids = sorted(set(records_map) & set(oracle_map))
    records = [records_map[i] for i in ids]
    labels = np.array([oracle_map[i]["oracle_label"] for i in ids], dtype=object)

    s1_rows = [stage1_features(r) for r in records]
    s2_rows = [stage2_features(r) for r in records]
    s2b_rows = [stage2_pi_b_features(r) for r in records]
    x1, keys1, c1, n1 = to_matrix(s1_rows)
    x2, keys2, c2, n2 = to_matrix(s2_rows)
    x2b, keys2b, c2b, n2b = to_matrix(s2b_rows)

    y1 = np.array([int(label == "m5_a") for label in labels])
    y2 = np.array([int(label == "pi_a") for label in labels])
    stage2_idx = [i for i, label in enumerate(labels) if label != "m5_a"]

    clf1 = make_dt(c1, n1, depth=4)
    clf1.fit(x1, y1)
    clf2 = make_dt(c2, n2, depth=4)
    clf2.fit([x2[i] for i in stage2_idx], y2[stage2_idx])
    clf2b = make_dt(c2b, n2b, depth=4)
    clf2b.fit([x2b[i] for i in stage2_idx], y2[stage2_idx])

    args.policy_dir.mkdir(parents=True, exist_ok=True)
    stage1_joblib = args.policy_dir / "stage1_m5_router.joblib"
    stage2_joblib = args.policy_dir / "stage2_pi_router.joblib"
    stage2b_joblib = args.policy_dir / "stage2_pi_b_router.joblib"
    stage1_txt = args.policy_dir / "stage1_m5_router.txt"
    stage2_txt = args.policy_dir / "stage2_pi_router.txt"
    stage2b_txt = args.policy_dir / "stage2_pi_b_router.txt"
    joblib.dump({"model": clf1, "feature_keys": keys1}, stage1_joblib)
    joblib.dump({"model": clf2, "feature_keys": keys2}, stage2_joblib)
    joblib.dump({"model": clf2b, "feature_keys": keys2b}, stage2b_joblib)
    stage1_txt.write_text(tree_text(clf1, keys1), encoding="utf-8")
    stage2_txt.write_text(tree_text(clf2, keys2), encoding="utf-8")
    stage2b_txt.write_text(tree_text(clf2b, keys2b), encoding="utf-8")

    return [{
        "type": "summary",
        "step": "export_router_policy",
        "records": len(ids),
        "stage1": {
            "target": "END@M5 vs ESCALATE@Pi",
            "model": "decision_tree_depth4",
            "features": keys1,
            "joblib": str(stage1_joblib),
            "text_tree": str(stage1_txt),
        },
        "stage2": {
            "target": "END@Pi vs ESCALATE@Jetson",
            "model": "decision_tree_depth4",
            "features": keys2,
            "joblib": str(stage2_joblib),
            "text_tree": str(stage2_txt),
        },
        "stage2_pi_b_only": {
            "target": "END@Pi vs ESCALATE@Jetson",
            "model": "decision_tree_depth4",
            "features": keys2b,
            "joblib": str(stage2b_joblib),
            "text_tree": str(stage2b_txt),
            "note": "Use this policy when online implementation runs only Pi_B at Stage 2.",
        },
        "note": "The text_tree files expose the learned nested-if policy; the joblib files are for Python simulation.",
    }]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    ap.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = build(args)
    write_jsonl(args.out, rows)
    print("[DONE] export router policy")
    print(f"  out      : {args.out}")
    print(f"  policies : {args.policy_dir}")


if __name__ == "__main__":
    main()
