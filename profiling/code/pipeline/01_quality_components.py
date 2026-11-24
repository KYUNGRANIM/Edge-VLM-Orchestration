#!/usr/bin/env python3
"""Step 1: extract tier-wise quality components."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import COLLECTION_DATASET, MEDIC_MANIFEST, OUTPUT_DIR, TIERS, write_jsonl

DEFAULT_COLLECTION = (
    COLLECTION_DATASET
)
# The judge scores are committed alongside the other pipeline outputs, so that
# reproducing steps 01-05 does not require re-running the paid GPT-4o judge.
DEFAULT_JUDGE = OUTPUT_DIR / "00_gpt_judge_scores.jsonl"
LEGACY_JUDGES = ()
DEFAULT_MEDIC = MEDIC_MANIFEST
DEFAULT_OUT = OUTPUT_DIR / "01_quality_components.jsonl"

# MEDIC's three levels on the same 0/1/2 scale the Pi and Jetson outputs use.
SEV_MAP = {
    "little_or_no_damage": 0,
    "little_or_none": 0,
    "no_damage": 0,
    "mild": 1,
    "mild_damage": 1,
    "moderate": 1,
    "moderate_damage": 1,
    "severe": 2,
    "severe_damage": 2,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_medic_gt(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            local = row.get("local_image_path") or row.get("image_path") or ""
            basename = Path(local).name
            cat = (row.get("group") or row.get("disaster_types") or "").strip()
            sev_raw = (row.get("damage_severity") or "").strip()
            if not basename or not cat or sev_raw not in SEV_MAP:
                continue
            gt = {"cat": cat, "sev": SEV_MAP[sev_raw], "sev_raw": sev_raw}
            out[basename] = gt
            out[f"{cat}__{basename}"] = gt
    return out


def sev_soft_score(pred: Any, gt: int) -> float:
    try:
        diff = abs(int(pred) - int(gt))
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)


def component_stats(rows: list[dict[str, Any]], component: str) -> dict[str, Any]:
    by_tier = {}
    for tier in TIERS:
        vals = [float(r[tier][component]) for r in rows]
        by_tier[tier] = {
            "mean": round(float(np.mean(vals)), 6),
            "median": round(float(np.median(vals)), 6),
            "std": round(float(np.std(vals)), 6),
        }
    tier_means = [by_tier[t]["mean"] for t in TIERS]
    all_vals = [float(r[t][component]) for r in rows for t in TIERS]
    grand = float(np.mean(all_vals))
    between = float(np.mean([(m - grand) ** 2 for m in tier_means]))
    within_terms = []
    for tier in TIERS:
        vals = [float(r[tier][component]) for r in rows]
        mu = float(np.mean(vals))
        within_terms.extend([(v - mu) ** 2 for v in vals])
    within = float(np.mean(within_terms)) if within_terms else 0.0
    return {"tier": by_tier, "fdr": round(between / (within + 1e-12), 8)}


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    collection = {r["image_id"]: r for r in read_jsonl(args.collection)}
    judge = {
        r["image_id"]: r
        for r in read_jsonl(args.judge)
        if r.get("status") == "ok" and r.get("image_id")
    }
    gt_map = load_medic_gt(args.medic)
    rows: list[dict[str, Any]] = []
    skipped = Counter()

    for image_id in sorted(collection):
        rec = collection[image_id]
        gt = gt_map.get(image_id)
        j = judge.get(image_id)
        if gt is None:
            skipped["missing_gt"] += 1
            continue
        if j is None:
            skipped["missing_judge"] += 1
            continue

        row: dict[str, Any] = {"type": "quality_components", "image_id": image_id, "gt": gt}
        for tier in TIERS:
            pred = rec[tier]
            text_score = float(j["scores"][tier]["text_judge_score"])
            cat_score = 1.0 if pred.get("cat") == gt["cat"] else 0.0
            sev_score = sev_soft_score(pred.get("sev"), gt["sev"])
            row[tier] = {
                "pred_cat": pred.get("cat"),
                "pred_sev": pred.get("sev"),
                "cat_score": cat_score,
                "sev_score": sev_score,
                "sev_exact": 1.0 if sev_score == 1.0 else 0.0,
                "text_score": text_score,
                "text": pred.get("text", ""),
            }
        rows.append(row)

    summary = {
        "type": "summary",
        "step": "01_quality_components",
        "records": len(rows),
        "input": {
            "collection": str(args.collection),
            "judge": str(args.judge),
            "medic": str(args.medic),
        },
        "output": str(args.out),
        "skipped": dict(skipped),
        "gt_dist": {
            "cat": dict(Counter(r["gt"]["cat"] for r in rows)),
            "sev": dict(Counter(str(r["gt"]["sev"]) for r in rows)),
        },
        "component_stats": {
            "cat_score": component_stats(rows, "cat_score"),
            "sev_score": component_stats(rows, "sev_score"),
            "text_score": component_stats(rows, "text_score"),
        },
    }
    return rows + [summary]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--medic", type=Path, default=DEFAULT_MEDIC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.judge.is_file():
        for fallback in LEGACY_JUDGES:
            if fallback.is_file():
                print(f"[WARN] judge not found at {args.judge}; using legacy file: {fallback}")
                args.judge = fallback
                break
    rows = build(args)
    write_jsonl(args.out, rows)
    summary = rows[-1]
    print("[DONE] quality components")
    print(f"  rows    : {summary['records']}")
    print(f"  skipped : {summary['skipped']}")
    print(f"  out     : {args.out}")


if __name__ == "__main__":
    main()
