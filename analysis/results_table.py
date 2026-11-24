#!/usr/bin/env python3
"""MEDIC held-out Table I for the rerun with the routers retrained on the corrected severity labels.

    python3 analysis/results_table.py [run_dir]

Same accounting as revision/eidseg/runtime/pipeline/results_table.py. Costs are the all-tier main-path costs
(02_cost) plus the auxiliary devices measured in this run (M5_B/M5_C from M5_A's answer-cache logs, Pi_B from
the run's Pi records). Quality is reported on two answer sets:
  all-tier  every method scored on the all-tier collection (how the original Table I was computed);
  run       one runtime answer set for every method: M5 = the run's M5_A answer, Pi/Jetson = the run's answer
            where the run reached that tier, else the all-tier answer (01_quality.jsonl of the run's judge input).
Direct@Pi/Jetson run only that tier; their costs are per-tier differences of the all-tier trace (not measured), with the
measured camera-to-Pi upload in place of the Pi->Jetson hop for Direct@Jetson.
For 91 of the 140 Jetson-routed images the run's Jetson call timed out (Jetson VLM degraded); their Jetson
answers were re-sent after the run with the same model and settings (`jetson_answer_source` in the run file).
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

M5VLM = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else M5VLM / "deployment/live_run"
HELD = M5VLM / "heldout/results/labels"
PROF = M5VLM / "profiling/results"
TIERS = ("m5_a", "pi_a", "jetson")
PREV = {"m5_a": None, "pi_a": "m5_a", "jetson": "pi_a"}
Q_TH, TAU, W = 0.65, 0.10, (0.3, 0.3, 0.4)
P_M5, P_PI = 1.2154, 7.5564  # W, measured averages (stage_power_w in the collection)
# profiling 02_cost normalization: global min to percentile-clipped max (02_cost_components.py)
NORM = {"latency_ms": (4366.0, 76605.0), "overhead_bytes": (0.0, 634702.9), "energy_j": (5.3, 558.1)}
N_BOOT = 2000


def jl(path, type_=None):
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # SD logs can hold a line cut by a power cut
            continue
        if (type_ is None or r.get("type") == type_) and r.get("status", "ok") == "ok":
            out[r["image_id"]] = r
    return out


def q(c):
    return round(W[0] * c["cat_score"] + W[1] * c["sev_score"] + W[2] * c["text_score"], 8)


run = jl(RUN / "online_routing_final.jsonl", "online_routing_final")
q_run = jl(RUN / "01_quality.jsonl", "quality_components")
q_all = jl(HELD / "01_quality_components.jsonl", "quality_components")
cost = jl(HELD / "02_cost_components.jsonl", "cost_components")
ref = jl(HELD / "03_oracle_labels.jsonl", "oracle_label")
ids = sorted(run)
assert len(ids) == 600 and set(ids) <= set(q_run) & set(q_all) & set(cost) & set(ref)
Q = {"all-tier": {i: {t: q(q_all[i][t]) for t in TIERS} for i in ids},
     "run": {i: {t: q(q_run[i][t]) for t in TIERS} for i in ids}}
assert all(abs(Q["all-tier"][i][t] - ref[i]["quality"][t]) < 1e-6 for i in ids for t in TIERS)

# ---- auxiliary execution measured in this run
aux_ms = {"b": [], "c": []}
for f in glob.glob(str(RUN / "m5a_sd*/m5a_aux_cache__*.jsonl")) + glob.glob(str(RUN / "m5a_sd*/*.extracted.jsonl")):
    for line in open(f, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ms = d.get("m5_total_ms") or (d.get("latency") or {}).get("m5_total_ms") or 0
        if d.get("event") == "store" and d.get("role") in aux_ms and ms > 0:
            aux_ms[d["role"]].append(ms)
end_pi = [run[i]["end_pi"]["cost"] for i in ids if run[i].get("end_pi")]
AUX_M5_J = P_M5 * sum(np.mean(v) for v in aux_ms.values()) / 1000
AUX_PI_J = P_PI * np.mean([c["latency_ms"]["pi_b_total_ms"] for c in end_pi]) / 1000
AUX_PI_B = np.mean([c["signal_overhead_bytes"]["m5_pi_b_tx_bytes"] + c["signal_overhead_bytes"]["m5_pi_b_rx_bytes"]
                    for c in end_pi])
src = Counter(run[i].get("jetson_answer_source") for i in ids if run[i]["route_label"] == "jetson")
print(f"aux: M5_B {np.mean(aux_ms['b'])/1000:.2f} s (n={len(aux_ms['b'])}), M5_C {np.mean(aux_ms['c'])/1000:.2f} s "
      f"(n={len(aux_ms['c'])}) -> {AUX_M5_J:.1f} J; Pi_B {AUX_PI_J/P_PI:.1f} s -> {AUX_PI_J:.1f} J, "
      f"upload {AUX_PI_B/1024:.0f} KiB | Jetson answers: {dict(src)}")


def per_image(tiers, aux=False, direct=False):
    """latency s, overhead KiB, energy J per image."""
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
            eng += AUX_M5_J + (AUX_PI_J if t != "m5_a" else 0)
            ovh += AUX_PI_B if t != "m5_a" else 0
        out.append((lat / 1000, ovh / 1024, eng))
    return np.array(out)


ROWS = {}


def row(name, tiers, qset, aux=False, direct=False):
    sat = np.array([Q[qset][i][t] >= Q_TH for i, t in zip(ids, tiers)])
    c = per_image(tiers, aux, direct)
    d = Counter(tiers)
    ROWS[name] = (sat, c)
    print(f"  {name:42s} {100 * sat.mean():6.1f} {c[:, 0].mean():7.1f} {c[:, 1].mean():7.0f} {c[:, 2].mean():7.1f}   "
          f"{d['m5_a']}/{d['pi_a']}/{d['jetson']}")


# ---- Confidence-Only: thresholds whose profiling composite cost (no auxiliary devices) is closest to the
# out-of-fold composite cost of the proposed routers, as in the first submission's analysis.
sys.path.insert(0, str(M5VLM / "profiling" / "code" / "pipeline"))
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from common import make_dt, stage1_features, stage2_features, to_matrix  # noqa: E402


def composite(c, t, aux):
    lat, ovh, eng = c[t]["latency_ms"], c[t]["overhead_bytes"], c[t]["energy_j"]
    if aux:
        eng += AUX_M5_J + (AUX_PI_J if t != "m5_a" else 0)
        ovh += AUX_PI_B if t != "m5_a" else 0
    n = {k: min(max((v - NORM[k][0]) / NORM[k][1], 0.0), 1.0)
         for k, v in (("latency_ms", lat), ("overhead_bytes", ovh), ("energy_j", eng))}
    return 0.1 * n["latency_ms"] + 0.1 * n["overhead_bytes"] + 0.8 * n["energy_j"]


def conf_pick(m5c, pic, c1, c2):
    return "m5_a" if m5c >= c1 else ("pi_a" if pic >= c2 else "jetson")


prof = jl(PROF / "collection.jsonl")
pref = jl(PROF / "pipeline_outputs/03_oracle_labels.jsonl", "oracle_label")
pcost = jl(PROF / "pipeline_outputs/02_cost_components.jsonl", "cost_components")
pids = sorted(set(prof) & set(pref) & set(pcost))
y = np.array([pref[i]["oracle_label"] for i in pids], dtype=object)
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
target = np.mean([composite(pcost[i], t, True) for i, t in zip(pids, oof)])
cand = sorted((abs(np.mean([composite(pcost[i], conf_pick(prof[i]["m5_a"].get("conf") or 0, prof[i]["pi_a"].get("conf") or 0,
                                                        a, b), False) for i in pids]) - target), a, b)
              for a in range(1, 6) for b in range(1, 6))
_, C1, C2 = cand[0]
alltier = jl(M5VLM / "heldout/results/collection.jsonl")
run_pi = {i: ((run[i].get("pi") or {}).get("pi_a") or {}) for i in ids}
conf = [conf_pick(run[i]["m5"]["m5_a"].get("conf") or 0, (run_pi[i] or alltier[i]["pi_a"]).get("conf") or 0, C1, C2)
        for i in ids]

ours = [run[i]["route_label"] for i in ids]
print(f"Confidence-Only thresholds on the profiling set: c1={C1}, c2={C2} (proposed OOF composite {target:.4f})")
hdr = f"  {'Method':42s} {'Q-Sat':>6} {'Lat(s)':>7} {'KiB':>7} {'J':>7}   M5/Pi/Jet"
print("\n[A] run answer set (main):\n" + hdr)
for name, t in (("Fixed-Termination@M5", "m5_a"), ("Fixed-Termination@Pi", "pi_a"), ("Fixed-Termination@Jetson", "jetson")):
    row(name, [t] * len(ids), "run")
row("Direct@Pi (cost: trace difference)", ["pi_a"] * len(ids), "run", direct=True)
row("Direct@Jetson (cost: camera upload + Jetson)", ["jetson"] * len(ids), "run", direct=True)
row(f"Confidence-Only (c1={C1}, c2={C2})", conf, "run")
row("Consistency-Aware Routing (Ours)", ours, "run", aux=True)
ref_run = []
for i in ids:
    best = max(Q["run"][i].values())
    ref_run.append(min((t for t in TIERS if best - Q["run"][i][t] <= TAU + 1e-12), key=lambda t: cost[i][t]["energy_j"]))
row("Quality-Cost Reference (rebuilt on this set)", ref_run, "run")
print("\n[B] all-tier answer set (as the original Table I):\n" + hdr)
for name, t in (("Fixed-Termination@M5 ", "m5_a"), ("Fixed-Termination@Pi ", "pi_a"), ("Fixed-Termination@Jetson ", "jetson")):
    row(name, [t] * len(ids), "all-tier")
row("Ours, scored on all-tier answers", ours, "all-tier", aux=True)
row("Quality-Cost Reference ", [ref[i]["oracle_label"] for i in ids], "all-tier")

# ---- paired differences on the run answer set, 95% bootstrap CIs over images
rng = np.random.default_rng(20261104)
B = [rng.integers(len(ids), size=len(ids)) for _ in range(N_BOOT)]
print("\n[C] Ours minus baseline on the run answer set (95% paired bootstrap CI):")
so, co = ROWS["Consistency-Aware Routing (Ours)"]
for base in ("Fixed-Termination@Pi", "Fixed-Termination@Jetson", "Direct@Jetson (cost: camera upload + Jetson)",
             f"Confidence-Only (c1={C1}, c2={C2})", "Fixed-Termination@M5"):
    sb, cb = ROWS[base]
    parts = []
    for name, a, b, scale in (("Q-Sat pp", so, sb, 100), ("lat s", co[:, 0], cb[:, 0], 1), ("energy J", co[:, 2], cb[:, 2], 1)):
        d = (a.astype(float) - b.astype(float)) * scale
        bs = [d[ix].mean() for ix in B]
        parts.append(f"{name} {d.mean():+.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]")
    print(f"  vs {base:40s} " + " | ".join(parts))
