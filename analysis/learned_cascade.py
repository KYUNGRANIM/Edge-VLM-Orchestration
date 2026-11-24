#!/usr/bin/env python3
"""An established learned cascade (FrugalGPT-style) and direct execution, on the MEDIC rerun or EIDSeg.

    python3 analysis/learned_cascade.py [medic|eidseg]

Learned cascade after FrugalGPT (Chen et al., 2023): M5 -> Pi -> Jetson; at M5 and Pi a scorer predicts whether that
tier's main answer passes Q-Sat, and the cascade stops when the score clears that tier's threshold. The scorer sees
only the answer it judges (text TF-IDF + category, severity, confidence; logistic regression) in place of
FrugalGPT's DistilBERT, which the M5/Pi cannot host; a labels-only scorer (no text) shows how much the text adds.
Thresholds (0.05 grid) are chosen on out-of-fold profiling
scores (Table I's 5 folds): FrugalGPT's rule, the highest Q-Sat within Ours' out-of-fold composite cost, and as a
second point the lowest cost at Ours' out-of-fold Q-Sat. No auxiliary devices (as Confidence-Only). Scored on the
run answer set like every Table I row.

Direct execution: Direct@Pi/Jetson costs are per-tier differences of the all-tier trace (results_table.py), with the
measured camera-to-Pi upload in place of the Pi->Jetson hop for Direct@Jetson (Table I). The load test
load test measured both for 60 of the images, one request at a time, which checks that estimate.
"""
import contextlib
import io
import json
import runpy
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "medic"
sys.argv = sys.argv[:1]  # results_table.py reads an optional run directory from argv
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(HERE / "results_table.py" if DATASET == "medic" else
                           HERE.parents[0] / "revision/eidseg/runtime/pipeline/results_table.py"))
if DATASET == "medic":
    pref, pcost, Qr = G["pref"], G["pcost"], G["Q"]["run"]
    AUX = (G["AUX_M5_J"], G["AUX_PI_J"], G["AUX_PI_B"])
else:  # EIDSeg keeps profiling and held-out labels and costs in one file each
    pref, pcost, Qr = G["ref"], G["cost"], G["Q_RUN"]
    AUX = (G["AUX_M5_ENERGY"], G["AUX_PI_ENERGY"], G["AUX_PI_OVERHEAD"])
prof, pids, y, run, ids, cost, alltier, ours = (G[k] for k in ("prof", "pids", "y", "run", "ids", "cost", "alltier", "ours"))
composite, Q_TH, target, N_BOOT = G["composite"], G["Q_TH"], G["target"], G.get("N_BOOT", 2000)
TIERS = ("m5_a", "pi_a", "jetson")
PREV = {"m5_a": None, "pi_a": "m5_a", "jetson": "pi_a"}


def per_image(tiers, aux=False, direct=False):
    """latency s, comm KiB, energy J per held-out image, as results_table.py (MEDIC) accounts them."""
    out = []
    for i, t in zip(ids, tiers):
        c, p = cost[i][t], PREV[t]
        lat, ovh, eng = c["latency_ms"], c["overhead_bytes"], c["energy_j"]
        if direct and p:
            lat, ovh, eng = lat - cost[i][p]["latency_ms"], ovh - cost[i][p]["overhead_bytes"], eng - cost[i][p]["energy_j"]
            if t == "jetson":  # the camera uploads straight to the Jetson: its upload replaces the Pi->Jetson hop
                s = alltier[i]["stage_latency_ms"]
                lat += s["m5_pi_http"] - s["pi_jet_http"]
        if aux:
            eng += AUX[0] + (AUX[1] if t != "m5_a" else 0)
            ovh += AUX[2] if t != "m5_a" else 0
        out.append((lat / 1000, ovh / 1024, eng))
    return np.array(out)
GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


def label(v):
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return "?"


class Scorer:
    """P(this answer passes Q-Sat), from the answer alone; text=False keeps only category, severity, confidence."""

    def __init__(self, text=True):
        self.text = text

    def fit(self, ans, passed):
        self.tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([a.get("text") or "" for a in ans])
        self.oh = OneHotEncoder(handle_unknown="ignore").fit(self.tab(ans))
        self.lr = LogisticRegression(max_iter=3000).fit(self.x(ans), passed)
        return self

    @staticmethod
    def tab(ans):
        return [[a.get("cat") or "?", label(a.get("sev")), label(a.get("conf"))] for a in ans]

    def x(self, ans):
        tab = self.oh.transform(self.tab(ans))
        return hstack([self.tf.transform([a.get("text") or "" for a in ans]), tab]).tocsr() if self.text else tab.tocsr()

    def score(self, ans):
        return self.lr.predict_proba(self.x(ans))[:, 1]


def routes(s_m5, s_pi, a, b):
    return np.where(s_m5 >= a, "m5_a", np.where(s_pi >= b, "pi_a", "jetson"))


def mcnemar(a, b):
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0


def ci(d):
    rng = np.random.default_rng(20261104)
    bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)]
    return f"{d.mean():+.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]"


# ---------------- out-of-fold scores on profiling, threshold choice, held-out routes
pans = {t: [prof[i][t] for i in pids] for t in TIERS[:2]}
qp = {t: np.array([pref[i]["quality"][t] >= Q_TH for i in pids]) for t in TIERS}
cp = {t: np.array([composite(pcost[i], t, False) for i in pids]) for t in TIERS}
ours_oof_q = np.mean([qp[t][j] for j, t in enumerate(G["oof"])])
ours_oof_jet = np.mean(G["oof"] == "jetson")  # share sent over the backhaul to the Jetson
print(f"profiling out of fold: Ours Q-Sat {100 * ours_oof_q:.1f}%, composite {target:.4f}, to Jetson {100 * ours_oof_jet:.1f}%")
run_pi = G["run_pi"]
held = {"m5_a": [run[i]["m5"]["m5_a"] for i in ids], "pi_a": [run_pi[i] or alltier[i]["pi_a"] for i in ids]}


def cascade(text):
    oof = {t: np.zeros(len(pids)) for t in TIERS[:2]}
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y):
        for t in TIERS[:2]:
            oof[t][te] = Scorer(text).fit([pans[t][j] for j in tr], qp[t][tr].astype(int)).score([pans[t][j] for j in te])
    grid = []
    for a in GRID:
        for b in GRID:
            r = routes(oof["m5_a"], oof["pi_a"], a, b)
            grid.append((np.mean([qp[t][j] for j, t in enumerate(r)]), np.mean([cp[t][j] for j, t in enumerate(r)]), a, b,
                         np.mean(r == "jetson")))
    picks = {"budget": max((g for g in grid if g[1] <= target), key=lambda g: (g[0], -g[1])),
             # Confidence-Only's rule in results_table.py: composite cost closest to Ours' out-of-fold composite
             "closest-cost": min(grid, key=lambda g: (abs(g[1] - target), -g[0])),
             # the topology argument: the Jetson sits behind a scarce backhaul, so cap the share sent to it at Ours'
             "backhaul": max((g for g in grid if g[4] <= ours_oof_jet + 1e-12), key=lambda g: (g[0], -g[1]))}
    reach = [g for g in grid if g[0] >= ours_oof_q]
    if reach:
        picks["quality"] = min(reach, key=lambda g: (g[1], -g[0]))
    else:
        print(f"  learned cascade ({'text + labels' if text else 'labels only'}): no thresholds reach Ours' OOF Q-Sat "
              f"(best {100 * max(g[0] for g in grid):.1f}%)")
    final = {t: Scorer(text).fit(pans[t], qp[t].astype(int)) for t in TIERS[:2]}
    s_m5, s_pi = final["m5_a"].score(held["m5_a"]), final["pi_a"].score(held["pi_a"])
    SCORES[text] = (s_m5, s_pi)  # held-out scores, for operating_points.py
    out = {}
    for name, g in picks.items():
        print(f"  learned cascade ({'text + labels' if text else 'labels only'}), {name}-matched: thresholds M5 {g[2]}, Pi {g[3]}"
              f" -> OOF Q-Sat {100 * g[0]:.1f}%, composite {g[1]:.4f}, to Jetson {100 * g[4]:.1f}%")
        out[name] = (g, routes(s_m5, s_pi, g[2], g[3]))
    return out


SCORES = {}
LC = {text: cascade(text) for text in (True, False)}
rows = {}


def add(name, tiers, aux=False, direct=False, costs=None):
    sat = np.array([Qr[i][t] >= Q_TH for i, t in zip(ids, tiers)])
    c = per_image(list(tiers), aux, direct) if costs is None else costs
    rows[name] = (sat, c, tiers)


add("Fixed@Pi (runs M5 first)", ["pi_a"] * 600)
add("Fixed@Jetson (runs M5, Pi first)", ["jetson"] * 600)
add("Direct@Pi", ["pi_a"] * 600, direct=True)
add("Direct@Jetson", ["jetson"] * 600, direct=True)
add(f"Confidence-Only ({G['C1']},{G['C2']})", G["conf"])
for text, out in LC.items():
    for name, (g, r) in out.items():
        add(f"Learned cascade {'text+labels' if text else 'labels only'}, {name} ({g[2]},{g[3]})", r)
add("Ours", ours, aux=True)
# Ours' Stage 1, escalations straight to the Jetson (skips Pi_A/Pi_B). Upload and Jetson time from the Pi->Jetson hop.
skip = []
for i, t in zip(ids, ours):
    m5, pi, jt = cost[i]["m5_a"], cost[i]["pi_a"], cost[i]["jetson"]
    lat, kib, j = m5["latency_ms"] / 1000, 0.0, m5["energy_j"] + AUX[0]
    if t != "m5_a":
        lat += (jt["latency_ms"] - pi["latency_ms"]) / 1000
        kib += (jt["overhead_bytes"] - pi["overhead_bytes"]) / 1024
        j += jt["energy_j"] - pi["energy_j"]
    skip.append((lat, kib, j))
add("Ours Stage 1 -> Jetson (skip Pi; estimate)", ["m5_a" if t == "m5_a" else "jetson" for t in ours], costs=np.array(skip))

print("\nheld-out 600, run answer set: Q-Sat %, latency s, comm KiB, energy J, M5/Pi/Jetson")
so, co, _ = rows["Ours"]
for name, (sat, c, tiers) in rows.items():
    n = {t: int(np.sum(np.asarray(tiers) == t)) for t in TIERS}
    vs = "" if name == "Ours" else (f" | Ours minus this: Q-Sat {ci((so.astype(float) - sat) * 100)} pp (p {mcnemar(so, sat):.3f}), "
                                    f"lat {ci(co[:, 0] - c[:, 0])} s, energy {ci(co[:, 2] - c[:, 2])} J")
    print(f"  {name:44s} {100 * sat.mean():5.1f} {c[:, 0].mean():6.1f} {c[:, 1].mean():5.0f} {c[:, 2].mean():6.1f}  "
          f"{n['m5_a']}/{n['pi_a']}/{n['jetson']} ({100 * n['jetson'] / len(ids):.1f}% to Jetson){vs}")

if DATASET == "medic":  # the load test ran on MEDIC images only
    # ---------------- direct execution: trace estimate vs the load-test measurement (k = 1, mean of 3 repetitions)
    R = G["M5VLM"] / "revision/load_test/results"
    fin = {r["image_id"]: r for r in map(json.loads, open(R / "pi_a/online_routing_final.jsonl"))}
    t = lambda s: datetime.fromisoformat(s).timestamp()  # noqa: E731
    meas = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(R / "load_test_log.jsonl")):
        if r.get("k") != 1 or r.get("method") not in ("direct_pi", "direct_jetson") or "rid" not in r:
            continue
        if r["method"] == "direct_jetson" and r.get("http") == "200":
            meas["jetson"][r["orig"]].append(r["seconds"])
        elif r["method"] == "direct_pi" and r["rid"] in fin and fin[r["rid"]].get("status") == "ok":
            f = fin[r["rid"]]
            meas["pi_a"][r["orig"]].append(t(f["ts"]) - t(f["recv_ts"]))
    print("\ndirect execution latency, trace estimate vs load-test measurement (same images, one request at a time)")
    for tier in ("pi_a", "jetson"):
        imgs = sorted(i for i, v in meas[tier].items() if len(v) >= 3)
        sl = [alltier[i]["stage_latency_ms"] for i in imgs]
        up = np.array([s["m5_pi_http"] for s in sl]) / 1000  # camera upload; Direct@Jetson's in Table I too
        proc = np.array([s["pi_total" if tier == "pi_a" else "jetson_total"] for s in sl]) / 1000
        got = np.array([np.mean(meas[tier][i][-3:]) for i in imgs])
        # the load test timed Direct@Pi on Pi_A (arrival -> final, no upload) and Direct@Jetson at the client (with upload)
        est = proc if tier == "pi_a" else up + proc
        note = "Pi_A arrival -> final answer, upload excluded" if tier == "pi_a" else "client submit -> reply, upload included"
        print(f"  {tier:6s} {len(imgs)} images: trace estimate {est.mean():.1f} s, measured {got.mean():.1f} s "
              f"({note}; per-image correlation {np.corrcoef(est, got)[0, 1]:.2f})")
        print(f"         trace parts: camera upload {up.mean():.1f} s + processing {proc.mean():.1f} s")
