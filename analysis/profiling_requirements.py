#!/usr/bin/env python3
"""How much of the all-tier offline profiling the routers need (MEDIC rerun or EIDSeg).

    python3 analysis/profiling_requirements.py [medic|eidseg]

A. Learning curve on the run answer set (Table I accounting, auxiliary devices included): both routers trained with
   the deployed recipe on a stratified fraction of the profiling set, 10 draws per fraction, replayed on the
   held-out run features. analysis/profiling_cost.py did the same on the all-tier answers.
B. Label-preserving early stop: profile the tiers bottom-up and stop once the reference label is already fixed.
   Termination costs rise with the tier (cumulative), so the reference is the lowest feasible tier. With tiers
   left, the best quality can still reach 1.0, so an evaluated tier is surely feasible when 1.0 - q <= tau and surely
   infeasible when (best so far) - q > tau. Stop when the lowest tier not surely infeasible is surely feasible.
   Images that stop at the M5 never reach Stage 2 training, so their Pi_A, Pi_B and Jetson runs are skipped;
   images that stop at the Pi skip the Jetson. Q values are taken as given: if the judge scores fewer candidates
   at a time, its scores may move (3-4% of Q-Sat outcomes change when the companion answer changes).
"""
import contextlib
import io
import runpy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "medic"
sys.argv = [str(HERE / "learned_cascade.py"), DATASET]
with contextlib.redirect_stdout(io.StringIO()):
    L = runpy.run_path(str(HERE / "learned_cascade.py"))
G, pref, prof, pids, y, run, ids, alltier, Qr, ours, per_image = (L[k] for k in (
    "G", "pref", "prof", "pids", "y", "run", "ids", "alltier", "Qr", "ours", "per_image"))
Q_TH, TIERS = G["Q_TH"], ("m5_a", "pi_a", "jetson")
TAU = 0.10
sys.path.insert(0, str(G["M5VLM"] / "profiling/code/pipeline"))
from common import make_dt, stage1_features, stage2_features, to_matrix  # noqa: E402

# ---------------- A. learning curve on the run answer set
x1, k1, c1, n1 = to_matrix([stage1_features(prof[i]) for i in pids])
x2, k2, c2, n2 = to_matrix([stage2_features(prof[i]) for i in pids])
m5 = {i: {k: run[i]["m5"][k] for k in ("m5_a", "m5_b", "m5_c")} for i in ids}
h1 = [[stage1_features(m5[i])[k] for k in k1] for i in ids]
h2 = []
for i in ids:
    f = (run[i].get("stage2") or {}).get("features") or stage2_features(
        {**m5[i], "pi_a": alltier[i]["pi_a"], "pi_b": alltier[i]["pi_b"], "pi_a_conf_detail": alltier[i].get("pi_a_conf_detail") or {}})
    h2.append([f[k] for k in k2])


def replay(idx):
    m1 = make_dt(c1, n1, depth=4).fit([x1[j] for j in idx], (y[idx] == "m5_a").astype(int))
    s2 = [j for j in idx if y[j] != "m5_a"]
    m2 = make_dt(c2, n2, depth=4).fit([x2[j] for j in s2], (y[s2] == "pi_a").astype(int))
    tiers = ["m5_a" if a else ("pi_a" if b else "jetson") for a, b in zip(m1.predict(h1), m2.predict(h2))]
    c = per_image(tiers, aux=True).mean(axis=0)
    return 100 * np.mean([Qr[i][t] >= Q_TH for i, t in zip(ids, tiers)]), c[0], c[2], tiers


fq, fl, fe, ft = replay(np.arange(len(pids)))
pi_q = 100 * np.mean([Qr[i]["pi_a"] >= Q_TH for i in ids])
print(f"{DATASET.upper()}: A. learning curve, {len(ids)} held-out images, run answer set, 10 stratified draws per fraction")
print(f"  retrained on 100% reproduces the deployed routes on {sum(a == b for a, b in zip(ft, ours))}/{len(ids)}; "
      f"Fixed@Pi Q-Sat {pi_q:.1f}%")
print(f"  100% ({len(pids)}): Q-Sat {fq:.1f}%  latency {fl:.1f} s  energy {fe:.0f} J  "
      f"tiers {'/'.join(str(Counter(ft)[t]) for t in TIERS)}")
for frac in (0.05, 0.10, 0.25, 0.50):
    r = []
    for seed in range(10):
        idx, _ = train_test_split(np.arange(len(pids)), train_size=frac, stratify=y, random_state=seed)
        q, lat, e, t = replay(np.sort(idx))
        r.append((q, lat, e, 100 * np.mean(np.array(t) == "jetson"), 100 * np.mean([a == b for a, b in zip(t, ft)])))
    r = np.array(r)
    print(f"  {int(frac * 100):3d}% ({int(frac * len(pids))}): Q-Sat {r[:, 0].mean():.1f}% [{r[:, 0].min():.1f}-{r[:, 0].max():.1f}] "
          f"(vs 100% {r[:, 0].mean() - fq:+.1f} pp, vs Fixed@Pi {r[:, 0].mean() - pi_q:+.1f}, below Fixed@Pi in "
          f"{int(np.sum(r[:, 0] < pi_q))}/10)  latency {r[:, 1].mean():.1f} s  energy {r[:, 2].mean():.0f} J [{r[:, 2].min():.0f}-{r[:, 2].max():.0f}]"
          f"  to Jetson {r[:, 3].mean():.1f}%  same route as 100% {r[:, 4].mean():.0f}%")


# ---------------- B. label-preserving early stop on the profiling set
def early_stop(q):
    """tiers that must be run, bottom-up, before the reference label is fixed; and that label."""
    for k in range(len(TIERS)):
        seen = [q[t] for t in TIERS[:k + 1]]
        best = max(seen) if k == len(TIERS) - 1 else 1.0
        for t in TIERS[:k + 1]:
            if max(seen) - q[t] > TAU + 1e-12:
                continue                      # surely infeasible
            if best - q[t] <= TAU + 1e-12:
                return k + 1, t               # surely feasible and every cheaper tier surely infeasible
            break                             # undetermined: run the next tier
    raise AssertionError(q)


cost_up = all(pref[i]["cost"]["m5_a"] < pref[i]["cost"]["pi_a"] < pref[i]["cost"]["jetson"] for i in pids)
stops, same = Counter(), 0
dev = {k: [0.0, 0.0] for k in ("M5_A", "M5_B", "M5_C", "Pi_A", "Pi_B", "Jetson")}  # seconds: all-tier, early stop
energy, judged = [0.0, 0.0], [0, 0]
AUX_M5_S, AUX_PI_S, P_M5, P_PI = 6.08, 31.35, 1.2154, 7.5564  # per auxiliary device and image (first submission's Table I)
for i in pids:
    n, lab = early_stop(pref[i]["quality"])
    stops[n] += 1
    same += lab == pref[i]["oracle_label"]
    s = prof[i]["stage_latency_ms"]
    e = prof[i]["stage_energy_j"]
    run_s = {"M5_A": s["m5_total"] / 1000, "M5_B": AUX_M5_S, "M5_C": AUX_M5_S,
             "Pi_A": s["pi_total"] / 1000, "Pi_B": AUX_PI_S, "Jetson": s["jetson_total"] / 1000}
    need = {"M5_A": 1, "M5_B": 1, "M5_C": 1, "Pi_A": n >= 2, "Pi_B": n >= 2, "Jetson": n >= 3}
    for d in dev:
        dev[d][0] += run_s[d]
        dev[d][1] += run_s[d] * need[d]
    e_m5, e_pi, e_jet = e.get("m5", 0) + 2 * P_M5 * AUX_M5_S, e.get("pi", 0) + P_PI * AUX_PI_S, e.get("jetson", 0)
    energy[0] += e_m5 + e_pi + e_jet
    energy[1] += e_m5 + (e_pi if n >= 2 else 0) + (e_jet if n >= 3 else 0)
    judged[0] += 3
    judged[1] += n
N = len(pids)
print(f"\nB. label-preserving early stop on the {N} profiling images (tau {TAU}); termination cost rises with the tier "
      f"on every image: {cost_up}; labels identical to the stored reference: {same}/{N}")
print(f"  tiers run per image: M5 only {stops[1]} ({100 * stops[1] / N:.1f}%), M5+Pi {stops[2]} ({100 * stops[2] / N:.1f}%), "
      f"all three {stops[3]} ({100 * stops[3] / N:.1f}%)")
print("  device hours all-tier -> early stop: " + ", ".join(
    f"{d} {a / 3600:.0f}->{b / 3600:.0f}" for d, (a, b) in dev.items())
      + f" | sum {sum(a for a, _ in dev.values()) / 3600:.0f}->{sum(b for _, b in dev.values()) / 3600:.0f} h"
      f" ({100 * (1 - sum(b for _, b in dev.values()) / sum(a for a, _ in dev.values())):.1f}% less)")
print(f"  energy {energy[0] / 3.6e6:.2f}->{energy[1] / 3.6e6:.2f} kWh ({100 * (1 - energy[1] / energy[0]):.1f}% less); "
      f"answers to judge {judged[0]}->{judged[1]} ({100 * (1 - judged[1] / judged[0]):.1f}% less)")
