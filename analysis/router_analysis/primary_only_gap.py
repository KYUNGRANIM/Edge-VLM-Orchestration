#!/usr/bin/env python3
"""Why the Q-Sat gap between Consistency-Aware Routing and Primary-Only Routing went from +4.3 pp (submitted)
to +0.8 pp (MEDIC rerun): apply the changes one at a time and print the gap after each.

    python3 analysis/router_analysis/primary_only_gap.py

Primary-Only uses the submitted definition throughout (no within- or cross-tier consistency features), which
reproduces the submitted row exactly. Pre-correction labels are read from commit 948b734, the last commit before
the severity correction of 2026-10-27.
"""
import contextlib
import io
import json
import runpy
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(HERE.parent / "main_results/results_table.py"))
M5VLM = G["M5VLM"]
from common import make_dt, safe_float, safe_int, stage1_features, stage2_features, to_matrix  # noqa: E402

OLD = "948b734"
HELD = "heldout/results/labels/03_oracle_labels.jsonl"
PROF = "profiling/results/pipeline_outputs/03_oracle_labels.jsonl"


def rows(text):
    out = {}
    for line in text.splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("type") == "oracle_label":
                out[r["image_id"]] = r
    return out


BEFORE_FIX = {HELD: "03_oracle_labels_heldout.jsonl", PROF: "03_oracle_labels_profiling.jsonl"}


def git_rows(path):  # the labels as committed at OLD (before the severity fix), saved as files for the release
    return rows((M5VLM / "revision/first_submission/labels_before_severity_fix" / BEFORE_FIX[path]).read_text())


def po1(rec):
    a = rec["m5_a"]
    return {"m5a_cat": a.get("cat", ""), "m5a_sev": safe_int(a.get("sev")), "m5a_conf": safe_int(a.get("conf"))}


def po2(rec):
    out, pa = po1(rec), rec["pi_a"]
    out.update(pia_cat=pa.get("cat", ""), pia_sev=safe_int(pa.get("sev")), pia_conf=safe_int(pa.get("conf")),
               pia_min_logprob=safe_float((rec.get("pi_a_conf_detail") or {}).get("min_logprob"), -5.0))
    return out


def train_po(labels):
    prof = G["prof"]
    pids = sorted(set(prof) & set(labels))
    y = np.array([labels[i]["oracle_label"] for i in pids], dtype=object)
    x1, k1, c1, n1 = to_matrix([po1(prof[i]) for i in pids])
    tr2 = [j for j in range(len(pids)) if y[j] != "m5_a"]
    x2, k2, c2, n2 = to_matrix([po2(prof[pids[j]]) for j in tr2])
    return ((make_dt(c1, n1, depth=4).fit(x1, (y == "m5_a").astype(int)), k1),
            (make_dt(c2, n2, depth=4).fit(x2, (y[tr2] == "pi_a").astype(int)), k2))


def route_po(models, m5a, pi_rec):
    (m1, k1), (m2, k2) = models
    out = []
    for i in ids:
        if m1.predict([[po1({"m5_a": m5a[i]})[k] for k in k1]])[0]:
            out.append("m5_a")
            continue
        f = po2({"m5_a": m5a[i], **pi_rec(i)})
        out.append("pi_a" if m2.predict([[f[k] for k in k2]])[0] else "jetson")
    return out


old_run = G["jl"](M5VLM / "revision/first_submission/live_run/online_routing_final.jsonl", "online_routing_final")
run, alltier = G["run"], G["alltier"]
ids = sorted(set(G["ids"]) & set(old_run))
assert len(ids) == 600
q_old = {i: r["quality"] for i, r in git_rows(HELD).items()}
q_all, q_run = G["Q"]["all-tier"], G["Q"]["run"]
lab_old, lab_new = git_rows(PROF), rows((M5VLM / PROF).read_text())
alltier_pi = lambda i: {"pi_a": alltier[i]["pi_a"], "pi_a_conf_detail": alltier[i].get("pi_a_conf_detail")}  # noqa: E731
run_pi = lambda i: ({"pi_a": run[i]["pi"]["pi_a"], "pi_a_conf_detail": run[i].get("pi_a_conf_detail")}  # noqa: E731
                    if (run[i].get("pi") or {}).get("pi_a") else alltier_pi(i))

# the retrained (deployed in the rerun) trees replayed on the submitted run's answers
p1 = joblib.load(M5VLM / "deployment/routers/stage1_m5_router.joblib")
p2 = joblib.load(M5VLM / "deployment/routers/stage2_pi_router.joblib")
ours_new_on_old = []
for i in ids:
    m5 = {k: old_run[i]["m5"][k] for k in ("m5_a", "m5_b", "m5_c")}
    if p1["model"].predict([[stage1_features(m5)[k] for k in p1["feature_keys"]]])[0]:
        ours_new_on_old.append("m5_a")
        continue
    f = (old_run[i].get("stage2") or {}).get("features") or stage2_features(
        {**m5, "pi_a": alltier[i]["pi_a"], "pi_b": alltier[i]["pi_b"], "pi_a_conf_detail": alltier[i].get("pi_a_conf_detail") or {}})
    ours_new_on_old.append("pi_a" if p2["model"].predict([[f[k] for k in p2["feature_keys"]]])[0] else "jetson")

old_m5 = {i: old_run[i]["m5"]["m5_a"] for i in ids}
new_m5 = {i: run[i]["m5"]["m5_a"] for i in ids}
po_old = route_po(train_po(lab_old), old_m5, alltier_pi)
po_cor = route_po(train_po(lab_new), old_m5, alltier_pi)
po_rerun = route_po(train_po(lab_new), new_m5, run_pi)
ours_old = [old_run[i]["route_label"] for i in ids]
ours_rerun = [run[i]["route_label"] for i in ids]

steps = [
    ("submitted (old labels, old routers, old run, all-tier scoring)", q_old, ours_old, po_old),
    ("+ corrected severity labels (scoring, Primary-Only training)", q_all, ours_old, po_cor),
    ("+ retrained Ours routers (replayed on the old run's answers)", q_all, ours_new_on_old, po_cor),
    ("+ rerun answers as router inputs (deployed rerun routes)", q_all, ours_rerun, po_rerun),
    ("+ scored on the answers each router saw (new Table I)", q_run, ours_rerun, po_rerun),
]
prev = None
for name, Q, a, b in steps:
    sa = np.array([Q[i][t] >= 0.65 for i, t in zip(ids, a)])
    sb = np.array([Q[i][t] >= 0.65 for i, t in zip(ids, b)])
    gap = 100 * (sa.mean() - sb.mean())
    n10, n01 = int(np.sum(sa & ~sb)), int(np.sum(~sa & sb))
    step = "" if prev is None else f"  (step {gap - prev:+.1f})"
    print(f"{name:66s} Ours {100 * sa.mean():5.1f}  Primary-Only {100 * sb.mean():5.1f}  gap {gap:+.1f} pp{step:14s} "
          f"p = {binomtest(n10, n10 + n01, 0.5).pvalue:.3f} ({n10}/{n01})")
    prev = gap
