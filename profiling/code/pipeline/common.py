#!/usr/bin/env python3
"""Shared helpers for the final paper-facing pipeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# profiling/code/pipeline/common.py -> parents[3] is the m5_vlm project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PIPELINE_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "profiling" / "results" / "pipeline_outputs"
POLICY_DIR = PROJECT_ROOT / "deployment" / "routers"
COLLECTION_DATASET = PROJECT_ROOT / "profiling" / "results" / "collection.jsonl"
MEDIC_MANIFEST = PROJECT_ROOT / "profiling" / "results" / "medic_manifest.csv"

TIERS = ("m5_a", "pi_a", "jetson")


def read_jsonl(path: Path, row_type: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row_type is None or row.get("type") == row_type:
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_DIR / "primary_balanced_d10.json"
    with cfg_path.open(encoding="utf-8") as f:
        return json.load(f)


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def majority_fraction(vals: list[Any]) -> float:
    return Counter(vals).most_common(1)[0][1] / len(vals)


def stage1_features(rec: dict[str, Any]) -> dict[str, Any]:
    """M5-side online features available before escalation."""
    a, b, c = rec["m5_a"], rec["m5_b"], rec["m5_c"]
    sevs = [safe_int(a.get("sev")), safe_int(b.get("sev")), safe_int(c.get("sev"))]
    confs = [safe_int(a.get("conf")), safe_int(b.get("conf")), safe_int(c.get("conf"))]
    cats = [a.get("cat", ""), b.get("cat", ""), c.get("cat", "")]
    return {
        "m5a_cat": a.get("cat", ""),
        "m5a_sev": sevs[0],
        "m5_conf_mean": float(np.mean(confs)),
        "m5_conf_min": min(confs),
        "m5_sev_range": max(sevs) - min(sevs),
        "m5_cat_agree": majority_fraction(cats),
        "m5_sev_agree": majority_fraction(sevs),
    }


def stage2_features(rec: dict[str, Any]) -> dict[str, Any]:
    """Pi-side online features available after Stage 1 escalation."""
    out = stage1_features(rec)
    pa, pb = rec["pi_a"], rec["pi_b"]
    da = rec.get("pi_a_conf_detail") or {}
    pa_sev = safe_int(pa.get("sev"))
    pb_sev = safe_int(pb.get("sev"))
    pa_conf = safe_int(pa.get("conf"))
    pb_conf = safe_int(pb.get("conf"))
    out.update({
        "pia_cat": pa.get("cat", ""),
        "pia_sev": pa_sev,
        "pi_conf_mean": float(np.mean([pa_conf, pb_conf])),
        "pi_conf_min": min(pa_conf, pb_conf),
        "pi_cat_match": int(pa.get("cat") == pb.get("cat")),
        "pi_sev_match": int(pa_sev == pb_sev),
        "pia_min_logprob": safe_float(da.get("min_logprob"), -5.0),
        "m5a_pia_cat_match": int(rec["m5_a"].get("cat") == pa.get("cat")),
        "m5a_pia_sev_diff": abs(safe_int(rec["m5_a"].get("sev")) - pa_sev),
    })
    return out


def to_matrix(rows: list[dict[str, Any]]) -> tuple[list[list[Any]], list[str], list[int], list[int]]:
    keys = sorted(rows[0])
    x = [[row[k] for k in keys] for row in rows]
    cat_idx = [i for i, k in enumerate(keys) if isinstance(rows[0][k], str)]
    num_idx = [i for i in range(len(keys)) if i not in cat_idx]
    return x, keys, cat_idx, num_idx


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_dt(cat_idx: list[int], num_idx: list[int], depth: int = 4) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", make_ohe(), cat_idx),
        ("num", StandardScaler(), num_idx),
    ])
    clf = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])

