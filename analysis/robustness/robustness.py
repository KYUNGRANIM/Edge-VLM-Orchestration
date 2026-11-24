#!/usr/bin/env python3
"""Robustness of Table I differences beyond the per-image intervals of paired_statistics.py.

    python3 analysis/robustness/robustness.py [medic|eidseg]

A. Multiple comparisons: exact McNemar (Q-Sat) and Wilcoxon signed-rank (latency, energy) for Ours against every
   baseline, Holm-adjusted over all tests of the dataset.
B. Event-cluster bootstrap: images of one disaster event move together (EIDSeg; MEDIC's held-out events need the
   CORSAIR manifest). Unknown-event images as one cluster, and as single images.
C. Training-data resampling: both routers retrained (deployed recipe) on 50 bootstrap resamples of the profiling
   set and replayed on the held-out run answers; shows how much the result depends on the particular training set.
"""
import contextlib
import csv
import io
import runpy
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "medic"
with contextlib.redirect_stdout(io.StringIO()):
    R = runpy.run_path(str(HERE.parent / "router_analysis/profiling_requirements.py"), run_name="r3_4")  # its argv[1] is DATASET too
L, replay, ids, pids, Qr, Q_TH, per_image = R["L"], R["replay"], R["ids"], R["pids"], R["Qr"], R["Q_TH"], R["per_image"]
rows = L["rows"]
so, co, _ = rows["Ours"]
pick = {"Fixed@Pi": "Fixed@Pi (runs M5 first)", "Direct@Pi": "Direct@Pi", "Fixed@Jetson": "Fixed@Jetson (runs M5, Pi first)",
        "Direct@Jetson": "Direct@Jetson", "Confidence-Only": next(k for k in rows if k.startswith("Confidence-Only")),
        "Learned cascade (text+labels)": next(k for k in rows if k.startswith("Learned cascade text+labels, closest-cost")),
        "Learned cascade (labels only)": next(k for k in rows if k.startswith("Learned cascade labels only, closest-cost"))}
SAME_Q = {"Direct@Pi", "Direct@Jetson"}  # same answers as the Fixed rows, so the same Q-Sat test

# ---------------- A. Holm over every test
tests = []
for name, key in pick.items():
    sb, cb, _ = rows[key]
    if name not in SAME_Q:
        n10, n01 = int(np.sum(so & ~sb)), int(np.sum(~so & sb))
        tests.append((name, "Q-Sat", 100 * (so.mean() - sb.mean()), "pp", binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0))
    for col, metric, unit in ((0, "latency", "s"), (2, "energy", "J")):
        d = co[:, col] - cb[:, col]
        tests.append((name, metric, d.mean(), unit, wilcoxon(d).pvalue if np.any(d) else 1.0))
order = np.argsort([t[4] for t in tests])
holm, run_max = np.empty(len(tests)), 0.0
for rank, k in enumerate(order):
    run_max = max(run_max, min(1.0, (len(tests) - rank) * tests[k][4]))
    holm[k] = run_max
print(f"{DATASET.upper()} held-out {len(ids)}: A. Ours minus baseline, raw p and Holm-adjusted p over {len(tests)} tests")
for (name, metric, d, unit, p), ph in zip(tests, holm):
    print(f"  {name:30s} {metric:8s} {d:+8.1f} {unit:2s}  p {p:.2g}  Holm {ph:.2g}  {'significant' if ph < 0.05 else 'n.s.'}")

# ---------------- B. event-cluster bootstrap
if DATASET == "eidseg":
    man = {r["local_image_path"].split("/")[-1]: r["event_name"] for r in csv.DictReader(open(L["G"]["M5VLM"] / "revision/eidseg/dataset/eidseg_manifest.csv"))}
    ev = [man[i] for i in ids]
    rng = np.random.default_rng(20261112)
    for label, groups in (("unknown as one event", ev), ("unknown as single images", [e if e != "unknown" else f"u{n}" for n, e in enumerate(ev)])):
        names = sorted(set(groups))
        members = {g: np.array([n for n, e in enumerate(groups) if e == g]) for g in names}
        boots = [np.concatenate([members[g] for g in rng.choice(names, size=len(names))]) for _ in range(2000)]
        print(f"\nB. event-cluster bootstrap ({label}: {len(names)} clusters, largest {max(len(m) for m in members.values())} images)")
        for name in ("Fixed@Pi", "Fixed@Jetson", "Direct@Jetson", "Confidence-Only", "Learned cascade (text+labels)"):
            sb, cb, _ = rows[pick[name]]
            dq = np.array([100 * (so[ix].mean() - sb[ix].mean()) for ix in boots])
            rl = np.array([100 * (co[ix, 0].mean() / cb[ix, 0].mean() - 1) for ix in boots])
            re = np.array([100 * (co[ix, 2].mean() / cb[ix, 2].mean() - 1) for ix in boots])
            f = lambda v: f"[{np.percentile(v, 2.5):+.1f}, {np.percentile(v, 97.5):+.1f}]"
            print(f"  vs {name:30s} Q-Sat {100 * (so.mean() - sb.mean()):+5.1f} pp {f(dq)} | latency {f(rl)}% | energy {f(re)}%")
else:
    print("\nB. event-cluster bootstrap: MEDIC held-out events are in event_bootstrap.py (skipped here)")

# ---------------- C. training-data resampling
rng = np.random.default_rng(20261112)
fixed = {n: rows[pick[n]] for n in ("Fixed@Pi", "Fixed@Jetson", "Confidence-Only")}
res = []
for _ in range(50):
    q, lat, e, tiers = replay(np.sort(rng.integers(len(pids), size=len(pids))))
    res.append([q, lat, e, 100 * np.mean(np.array(tiers) == "jetson")]
               + [q - 100 * fixed[n][0].mean() for n in fixed]
               + [100 * (lat / fixed[n][1][:, 0].mean() - 1) for n in fixed] + [100 * (e / fixed[n][1][:, 2].mean() - 1) for n in fixed])
r = np.array(res)
pc = lambda v: f"{v.mean():.1f} [{np.percentile(v, 2.5):.1f}, {np.percentile(v, 97.5):.1f}]"
print(f"\nC. routers retrained on 50 bootstrap resamples of the {len(pids)} profiling images (deployed: Q-Sat {100 * so.mean():.1f}%, "
      f"{co[:, 0].mean():.1f} s, {co[:, 2].mean():.0f} J)")
print(f"  Q-Sat {pc(r[:, 0])}%  latency {pc(r[:, 1])} s  energy {pc(r[:, 2])} J  to Jetson {pc(r[:, 3])}%")
for k, n in enumerate(fixed):
    d, rl, re = r[:, 4 + k], r[:, 7 + k], r[:, 10 + k]
    print(f"  vs {n:16s} Q-Sat {pc(d)} pp (Ours higher in {int(np.sum(d > 0))}/50) | latency {pc(rl)}% | energy {pc(re)}% "
          f"(lower in {int(np.sum(re < 0))}/50)")
