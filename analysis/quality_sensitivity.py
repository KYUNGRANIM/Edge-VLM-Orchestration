#!/usr/bin/env python3
"""Sensitivity of the MEDIC rerun results to the quality weights and the reference tolerance, and whether an
answer can pass Q-Sat with a wrong category or severity.

    python3 analysis/quality_sensitivity.py

A. Pass with wrong labels: which (category, severity) outcomes can reach Q >= 0.65, and how often they do.
B. Re-scoring only: the deployed routes and baselines scored on the run answers under other weights.
C. Re-training: for each weighting and tolerance, rebuild the profiling reference labels, retrain both trees
   with the deployed recipe, replay them on the held-out run answers, and score with the same weighting.
   (Weights and tolerance only change labels; costs never decrease with tier, so a reference is the earliest
   tier within the tolerance; see cost_weights.py.)
"""
import contextlib
import io
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    P = runpy.run_path(str(HERE / "paper_numbers.py"))
G = P["G"]
M5VLM, ids, run, alltier, per_image = G["M5VLM"], G["ids"], G["run"], G["alltier"], G["per_image"]
ours, conf, po = G["ours"], G["conf"], P["tiers"]
make_dt, to_matrix, sf1, sf2 = G["make_dt"], G["to_matrix"], G["stage1_features"], G["stage2_features"]
T, TI, n = ("m5_a", "pi_a", "jetson"), {"m5_a": 0, "pi_a": 1, "jetson": 2}, len(ids)
QR = G["q_run"]  # run answer set components: cat_score, sev_score, text_score per tier
QP = {r["image_id"]: r for r in map(json.loads, open(M5VLM / "profiling/results/pipeline_outputs/01_quality_components.jsonl"))
      if r.get("type") == "quality_components"}
WEIGHTS = [(0.3, 0.3, 0.4), (0.4, 0.3, 0.3), (0.25, 0.25, 0.5), (1 / 3, 1 / 3, 1 / 3), (0.2, 0.2, 0.6), (0.4, 0.4, 0.2)]


def wname(w):
    return "1/3 each" if w[0] == 1 / 3 else "/".join(f"{x:g}" for x in w)


def qv(c, w):
    return round(w[0] * c["cat_score"] + w[1] * c["sev_score"] + w[2] * c["text_score"], 8)


def sat(tiers, w):
    return np.array([qv(QR[i][t], w) >= 0.65 for i, t in zip(ids, tiers)])


def earliest(qs, tau):
    best = max(qs.values())
    return min((t for t in T if best - qs[t] <= tau + 1e-12), key=TI.get)


def mcn(a, b):
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0


# ---------------- A. passing with a wrong category or severity
W0 = WEIGHTS[0]
print("A. Q = 0.3 cat + 0.3 sev + 0.4 text >= 0.65: category wrong needs exact severity and text >= 0.875; severity wrong (0)")
print("   needs the right category and text >= 0.875; adjacent severity (0.5) with the right category needs text >= 0.5;")
print("   category wrong with adjacent or wrong severity cannot pass.")
allans = [(QR[i][t], t) for i in ids for t in T]
for label, sel in (("all 1,800 run answers", allans), ("Ours' 600 final answers", [(QR[i][t], t) for i, t in zip(ids, ours)]),
                   ("Fixed@Pi's 600 answers", [(QR[i]["pi_a"], "pi_a") for i in ids])):
    passed = [c for c, _ in sel if qv(c, W0) >= 0.65]
    k = Counter(("cat ok" if c["cat_score"] == 1 else "cat WRONG") + ", sev " + {1.0: "exact", 0.5: "adjacent", 0.0: "WRONG"}[c["sev_score"]]
                for c in passed)
    print(f"   {label}: {len(passed)} pass; " + "; ".join(f"{a} {b}" for a, b in sorted(k.items())))
strict = {"Q-Sat": lambda c: qv(c, W0) >= 0.65, "+ category right": lambda c: qv(c, W0) >= 0.65 and c["cat_score"] == 1,
          "+ category right, severity exact": lambda c: qv(c, W0) >= 0.65 and c["cat_score"] == 1 and c["sev_score"] == 1}
methods = {"Fixed@M5": ["m5_a"] * n, "Fixed@Pi": ["pi_a"] * n, "Fixed@Jetson": ["jetson"] * n,
           f"Conf-Only({G['C1']},{G['C2']})": conf, "Ours": ours, "Primary-Only": po}
print("   pass rule".ljust(38) + "".join(f"{m:>15s}" for m in methods) + "   Ours-Pi (p)   Ours-Conf (p)")
for sn, f in strict.items():
    s = {m: np.array([f(QR[i][t]) for i, t in zip(ids, tr)]) for m, tr in methods.items()}
    cm = f"Conf-Only({G['C1']},{G['C2']})"
    print(f"   {sn:35s}" + "".join(f"{100 * v.mean():15.1f}" for v in s.values())
          + f"   {100 * (s['Ours'].mean() - s['Fixed@Pi'].mean()):+.1f} ({mcn(s['Ours'], s['Fixed@Pi']):.2f})"
          + f"   {100 * (s['Ours'].mean() - s[cm].mean()):+.1f} ({mcn(s['Ours'], s[cm]):.2f})")

# ---------------- B. re-scoring the deployed routes
print("\nB. deployed routes re-scored (run answers): Q-Sat %; reference rebuilt for tau 0.05/0.10/0.15")
print("   weights cat/sev/text".ljust(28) + "".join(f"{m:>15s}" for m in methods) + "   Ours-Pi (p)    Ours-Conf (p)   Reference")
for w in WEIGHTS:
    s = {m: sat(tr, w) for m, tr in methods.items()}
    cm = f"Conf-Only({G['C1']},{G['C2']})"
    refs = [100 * np.mean([qv(QR[i][earliest({t: qv(QR[i][t], w) for t in T}, tau)], w) >= 0.65 for i in ids]) for tau in (0.05, 0.10, 0.15)]
    print(f"   {wname(w):25s}" + "".join(f"{100 * v.mean():15.1f}" for v in s.values())
          + f"   {100 * (s['Ours'].mean() - s['Fixed@Pi'].mean()):+5.1f} ({mcn(s['Ours'], s['Fixed@Pi']):.2f})"
          + f"   {100 * (s['Ours'].mean() - s[cm].mean()):+5.1f} ({mcn(s['Ours'], s[cm]):.2f})   " + "/".join(f"{r:.1f}" for r in refs))

# ---------------- C. re-training under each weighting and tolerance
pids = sorted(set(G["prof"]) & set(QP))
prof = G["prof"]
x1, k1, c1i, n1i = to_matrix([sf1(prof[i]) for i in pids])
x2all, k2, c2i, n2i = to_matrix([sf2(prof[i]) for i in pids])
m5 = {i: {k: run[i]["m5"][k] for k in ("m5_a", "m5_b", "m5_c")} for i in ids}
h1 = [[sf1(m5[i])[k] for k in k1] for i in ids]
h2 = []
for i in ids:
    f = (run[i].get("stage2") or {}).get("features") or sf2(
        {**m5[i], "pi_a": alltier[i]["pi_a"], "pi_b": alltier[i]["pi_b"], "pi_a_conf_detail": alltier[i].get("pi_a_conf_detail") or {}})
    h2.append([f[k] for k in k2])
print("\nC. retrained per weighting/tolerance, replayed on the held-out run answers (Q-Sat %, latency s, energy J, M5/Pi/Jetson)")
print("   weights cat/sev/text   tau | Ours-retrained              | Fixed@Pi  Conf-Only | Ours-Pi (p)    Ours-Conf (p)")
for w in WEIGHTS:
    for tau in (0.05, 0.10, 0.15):
        y = np.array([earliest({t: qv(QP[i][t], w) for t in T}, tau) for i in pids], dtype=object)
        m1 = make_dt(c1i, n1i, depth=4).fit(x1, (y == "m5_a").astype(int))
        tr2 = [j for j in range(len(pids)) if y[j] != "m5_a"]
        m2 = make_dt(c2i, n2i, depth=4).fit([x2all[j] for j in tr2], (y[tr2] == "pi_a").astype(int))
        e1, e2 = m1.predict(h1), m2.predict(h2)
        tiers = ["m5_a" if a else ("pi_a" if b else "jetson") for a, b in zip(e1, e2)]
        s, sp, sc = sat(tiers, w), sat(["pi_a"] * n, w), sat(conf, w)
        c, d = per_image(tiers, True).mean(axis=0), Counter(tiers)
        same = "" if (w, tau) != (WEIGHTS[0], 0.10) else f"   [deployed routes reproduced: {sum(a == b for a, b in zip(tiers, ours))}/600]"
        print(f"   {wname(w):20s} {tau:.2f} | {100 * s.mean():5.1f} {c[0]:5.1f} {c[2]:6.1f} {d['m5_a']:3d}/{d['pi_a']:3d}/{d['jetson']:3d} "
              f"| {100 * sp.mean():6.1f} {100 * sc.mean():8.1f} | {100 * (s.mean() - sp.mean()):+5.1f} ({mcn(s, sp):.2f})"
              f"   {100 * (s.mean() - sc.mean()):+5.1f} ({mcn(s, sc):.2f}){same}")
