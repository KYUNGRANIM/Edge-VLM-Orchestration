#!/usr/bin/env python3
"""EIDSeg held-out results table, computed the way the first submission computed MEDIC Table I.

    python3 revision/eidseg/runtime/pipeline/results_table.py [run_dir]

Fixed-Termination and the reference use the all-tier collection (every tier ran on every image). Ours takes
its tier from the deployed run; its Q-Sat is scored on the answers the run itself returned (`01_quality.jsonl`
from build_runtime_judge_input.py), and also on the all-tier answers as MEDIC Table I did. Costs are the
all-tier main-path costs plus the auxiliary devices, whose times are measured in this run (M5_A cache log on
its SD card for M5_B/M5_C, Pi_A records for Pi_B).
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

M5VLM = Path(__file__).resolve().parents[4]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else M5VLM / "revision/eidseg/runtime/results/final_20261103"
LABELS = M5VLM / "revision/eidseg/labels"
TIERS = ("m5_a", "pi_a", "jetson")
Q_TH, W = 0.65, (0.3, 0.3, 0.4)
P_M5, P_PI = 1.2154, 7.5564  # W, measured averages (stage_power_w in the collection)


def jl(path, type_=None):
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # SD logs can hold a line cut by a power cut
            continue
        if type_ is None or r.get("type") == type_:
            out[r["image_id"]] = r
    return out


def q(row, t):
    c = row[t]
    return round(W[0] * c["cat_score"] + W[1] * c["sev_score"] + W[2] * c["text_score"], 8)


run = jl(RUN / "online_routing_final.jsonl", "online_routing_final")
q_run = jl(RUN / "01_quality.jsonl", "quality_components")
q_all = jl(LABELS / "01_quality.jsonl", "quality_components")
cost = jl(LABELS / "02_cost.jsonl", "cost_components")
ref = jl(LABELS / "03_oracle.jsonl", "oracle_label")
ids = sorted(run)
assert len(ids) == 600 and set(ids) <= set(q_run) & set(q_all) & set(cost) & set(ref)

# auxiliary execution measured in this run
aux_ms = {"b": [], "c": []}
for f in glob.glob(str(RUN / "m5a_sd/m5a_aux_cache__*.jsonl")):
    for line in open(f, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == "store" and d.get("role") in aux_ms and (d.get("m5_total_ms") or 0) > 0:
            aux_ms[d["role"]].append(d["m5_total_ms"])
end_pi = [run[i]["end_pi"]["cost"] for i in ids if run[i].get("end_pi")]
AUX_M5_ENERGY = P_M5 * sum(np.mean(v) for v in aux_ms.values()) / 1000
AUX_PI_ENERGY = P_PI * np.mean([c["latency_ms"]["pi_b_total_ms"] for c in end_pi]) / 1000
AUX_PI_OVERHEAD = np.mean([c["signal_overhead_bytes"]["m5_pi_b_tx_bytes"] + c["signal_overhead_bytes"]["m5_pi_b_rx_bytes"]
                           for c in end_pi])
print(f"aux: M5_B {np.mean(aux_ms['b'])/1000:.2f} s, M5_C {np.mean(aux_ms['c'])/1000:.2f} s -> {AUX_M5_ENERGY:.1f} J; "
      f"Pi_B {AUX_PI_ENERGY/P_PI:.2f} s -> {AUX_PI_ENERGY:.1f} J, upload {AUX_PI_OVERHEAD/1024:.1f} KB")


def row(name, tiers, quality, aux):
    lat, ovh, eng = [], [], []
    for i, t in zip(ids, tiers, strict=True):
        c = cost[i][t]
        lat.append(c["latency_ms"] / 1000)
        ovh.append((c["overhead_bytes"] + (AUX_PI_OVERHEAD if aux and t != "m5_a" else 0)) / 1024)
        eng.append(c["energy_j"] + ((AUX_M5_ENERGY + (AUX_PI_ENERGY if t != "m5_a" else 0)) if aux else 0))
    d = Counter(tiers)
    qsat = 100 * np.mean([quality[i][t] >= Q_TH for i, t in zip(ids, tiers, strict=True)])
    print(f"{name:34} {qsat:6.1f} {np.mean(lat):8.1f} {np.mean(ovh):8.1f} {np.mean(eng):8.1f}   "
          f"{d['m5_a']}/{d['pi_a']}/{d['jetson']}")


Q_ALL = {i: {t: q(q_all[i], t) for t in TIERS} for i in ids}
Q_RUN = {i: {t: q(q_run[i], t) for t in TIERS} for i in ids}
assert all(abs(Q_ALL[i][t] - ref[i]["quality"][t]) < 1e-6 for i in ids for t in TIERS)
ours = [run[i]["route_label"] for i in ids]
print(f"{'Method':34} {'Q-Sat':>6} {'Lat(s)':>8} {'Ovh(KB)':>8} {'Eng(J)':>8}   T1/T2/T3")
for name, t in (("Fixed-Termination@M5", "m5_a"), ("Fixed-Termination@Pi", "pi_a"), ("Fixed-Termination@Jetson", "jetson")):
    row(name, [t] * len(ids), Q_ALL, aux=False)
row("Ours (run's own answers)", ours, Q_RUN, aux=True)
row("Ours (all-tier answers, as MEDIC)", ours, Q_ALL, aux=True)
row("Quality-Cost Reference", [ref[i]["oracle_label"] for i in ids], Q_ALL, aux=False)

# ---- Confidence-Only, thresholds chosen on the profiling set as in the first submission's analysis: the (c1, c2) whose
# composite cost (no auxiliary devices) is closest to the out-of-fold composite cost of the proposed routers.
# Decisions and scores use the same answers: the run's own M5 answer, and its Pi answer where the run reached Pi.
sys.path.insert(0, str(M5VLM / "profiling" / "code" / "pipeline"))
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from common import make_dt, stage1_features, stage2_features, to_matrix  # noqa: E402

NORM = {"latency_ms": (4428.0, 74977.4), "overhead_bytes": (0.0, 618915.0), "energy_j": (5.4, 545.2)}  # from 02_cost


def composite(c, t, aux):
    lat, ovh, eng = c[t]["latency_ms"], c[t]["overhead_bytes"], c[t]["energy_j"]
    if aux:
        eng += AUX_M5_ENERGY + (AUX_PI_ENERGY if t != "m5_a" else 0)
        ovh += AUX_PI_OVERHEAD if t != "m5_a" else 0
    n = {k: min(max((v - NORM[k][0]) / NORM[k][1], 0.0), 1.0) for k, v in
         (("latency_ms", lat), ("overhead_bytes", ovh), ("energy_j", eng))}
    return 0.1 * n["latency_ms"] + 0.1 * n["overhead_bytes"] + 0.8 * n["energy_j"]


def conf_pick(m5c, pic, c1, c2):
    return "m5_a" if m5c >= c1 else ("pi_a" if pic >= c2 else "jetson")


prof = jl(M5VLM / "revision/eidseg/collection_eidseg_profiling.jsonl")
pids = sorted(set(prof) & set(ref) & set(cost))
y = np.array([ref[i]["oracle_label"] for i in pids], dtype=object)
x1, _, c1i, n1i = to_matrix([stage1_features(prof[i]) for i in pids])
x2, _, c2i, n2i = to_matrix([stage2_features(prof[i]) for i in pids])
oof = np.empty(len(pids), dtype=object)
for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y):
    m1 = make_dt(c1i, n1i, depth=4).fit([x1[i] for i in tr], (y[tr] == "m5_a").astype(int))
    tr2 = [i for i in tr if y[i] != "m5_a"]
    m2 = make_dt(c2i, n2i, depth=4).fit([x2[i] for i in tr2], (y[tr2] == "pi_a").astype(int))
    s1, s2 = m1.predict([x1[i] for i in te]), m2.predict([x2[i] for i in te])
    for j, g in enumerate(te):
        oof[g] = "m5_a" if s1[j] else ("pi_a" if s2[j] else "jetson")
target = np.mean([composite(cost[i], t, True) for i, t in zip(pids, oof)])
cand = sorted(((abs(np.mean([composite(cost[i], conf_pick(prof[i]["m5_a"].get("conf") or 0, prof[i]["pi_a"].get("conf") or 0, a, b), False)
                             for i in pids]) - target), a, b) for a in range(1, 6) for b in range(1, 6)))
_, C1, C2 = cand[0]
print(f"Confidence-Only thresholds on the profiling set: c1={C1}, c2={C2} (proposed OOF composite {target:.4f})")
run_pi = {i: ((run[i].get("pi") or {}).get("pi_a") or {}) for i in ids}
alltier = jl(M5VLM / "revision/eidseg/collection_eidseg_full.jsonl")
conf = [conf_pick(run[i]["m5"]["m5_a"].get("conf") or 0, (run_pi[i] or alltier[i]["pi_a"]).get("conf") or 0, C1, C2)
        for i in ids]
row(f"Confidence-Only (c1={C1}, c2={C2})", conf, Q_RUN, aux=False)
print("Fixed baselines scored on the run's answer set (run's M5 answers; Pi/Jetson where the run reached them):")
for name, t in (("  Fixed@M5", "m5_a"), ("  Fixed@Pi", "pi_a"), ("  Fixed@Jetson", "jetson")):
    row(name, [t] * len(ids), Q_RUN, aux=False)
