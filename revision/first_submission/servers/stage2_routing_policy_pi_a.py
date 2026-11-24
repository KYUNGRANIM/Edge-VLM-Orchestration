#!/usr/bin/env python3
"""Online routing helpers for the Pi_A + Pi_B Stage-2 path.

This module mirrors the exported offline Stage-2 policy:

  M5_A/B/C features + Pi_A/Pi_B features -> END@Pi vs ESCALATE@Jetson

It does not use GT, GPT judge scores, oracle labels, or Jetson output at
decision time. The joblib file only contains the already-trained decision tree.
"""

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
    if not vals:
        return 0.0
    return Counter(vals).most_common(1)[0][1] / len(vals)


def vectorize(row: dict[str, Any], keys: list[str]) -> list[Any]:
    return [row[k] for k in keys]


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


def stage2_features(
    rec: dict[str, Any],
    pi_a_output: dict[str, Any],
    pi_a_conf_detail: dict[str, Any] | None,
    pi_b_output: dict[str, Any],
    pi_b_conf_detail: dict[str, Any] | None,
) -> dict[str, Any]:
    out = stage1_features(rec)
    pa_sev = safe_int(pi_a_output.get("sev"))
    pb_sev = safe_int(pi_b_output.get("sev"))
    pa_conf = safe_int(pi_a_output.get("conf"))
    pb_conf = safe_int(pi_b_output.get("conf"))
    out.update({
        "pia_cat": pi_a_output.get("cat", ""),
        "pia_sev": pa_sev,
        "pi_conf_mean": float(np.mean([pa_conf, pb_conf])),
        "pi_conf_min": min(pa_conf, pb_conf),
        "pi_cat_match": int(pi_a_output.get("cat") == pi_b_output.get("cat")),
        "pi_sev_match": int(pa_sev == pb_sev),
        "pia_min_logprob": safe_float((pi_a_conf_detail or {}).get("min_logprob"), -5.0),
        "m5a_pia_cat_match": int(rec["m5_a"].get("cat") == pi_a_output.get("cat")),
        "m5a_pia_sev_diff": abs(safe_int(rec["m5_a"].get("sev")) - pa_sev),
    })
    return out


class OnlineRouterAB:
    def __init__(self, policy_dir: Path | str = DEFAULT_POLICY_DIR):
        policy_dir = Path(policy_dir)
        stage2 = joblib.load(policy_dir / "stage2_pi_router.joblib")
        self.stage2_model = stage2["model"]
        self.stage2_keys = stage2["feature_keys"]

    def end_at_pi(
        self,
        rec: dict[str, Any],
        pi_a_output: dict[str, Any],
        pi_a_conf_detail: dict[str, Any] | None,
        pi_b_output: dict[str, Any],
        pi_b_conf_detail: dict[str, Any] | None,
    ) -> tuple[bool, dict[str, Any]]:
        feat = stage2_features(rec, pi_a_output, pi_a_conf_detail, pi_b_output, pi_b_conf_detail)
        pred = int(self.stage2_model.predict([vectorize(feat, self.stage2_keys)])[0])
        return pred == 1, feat
