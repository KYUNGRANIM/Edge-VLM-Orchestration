#!/usr/bin/env python3
"""Online routing helpers for the Pi_B implementation path."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_DIR = ROOT / "routers"


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


def stage2_pi_b_features(rec: dict[str, Any], pi_b_output: dict[str, Any], pi_b_conf_detail: dict[str, Any] | None) -> dict[str, Any]:
    out = stage1_features(rec)
    pb_sev = safe_int(pi_b_output.get("sev"))
    out.update({
        "pib_cat": pi_b_output.get("cat", ""),
        "pib_sev": pb_sev,
        "pib_conf": safe_int(pi_b_output.get("conf")),
        "pib_min_logprob": safe_float((pi_b_conf_detail or {}).get("min_logprob"), -5.0),
        "m5a_pib_cat_match": int(rec["m5_a"].get("cat") == pi_b_output.get("cat")),
        "m5a_pib_sev_diff": abs(safe_int(rec["m5_a"].get("sev")) - pb_sev),
    })
    return out


def vectorize(row: dict[str, Any], keys: list[str]) -> list[Any]:
    return [row[k] for k in keys]


class OnlineRouterB:
    def __init__(self, policy_dir: Path | str = DEFAULT_POLICY_DIR):
        policy_dir = Path(policy_dir)
        s1 = joblib.load(policy_dir / "stage1_m5_router.joblib")
        s2 = joblib.load(policy_dir / "stage2_pi_b_router.joblib")
        self.stage1_model = s1["model"]
        self.stage1_keys = s1["feature_keys"]
        self.stage2_model = s2["model"]
        self.stage2_keys = s2["feature_keys"]

    def end_at_m5(self, rec: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        feat = stage1_features(rec)
        pred = int(self.stage1_model.predict([vectorize(feat, self.stage1_keys)])[0])
        return pred == 1, feat

    def end_at_pi(self, rec: dict[str, Any], pi_b_output: dict[str, Any], pi_b_conf_detail: dict[str, Any] | None) -> tuple[bool, dict[str, Any]]:
        feat = stage2_pi_b_features(rec, pi_b_output, pi_b_conf_detail)
        pred = int(self.stage2_model.predict([vectorize(feat, self.stage2_keys)])[0])
        return pred == 1, feat

