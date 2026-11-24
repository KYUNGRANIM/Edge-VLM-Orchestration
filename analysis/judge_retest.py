#!/usr/bin/env python3
"""Scoring variability: the final 600-image evaluation's judge inputs scored a second time with the same snapshot
(gpt-4o-2024-08-06), prompt, candidate order and images (deployment/live_run_rescored/), against the 2026-11-04 scores.

    python3 -B analysis/judge_retest.py
"""
import contextlib
import glob
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

HERE = Path(__file__).resolve().parent
M5VLM = HERE.parents[0]
R = M5VLM / "deployment"
with contextlib.redirect_stdout(io.StringIO()):
    P = runpy.run_path(str(HERE / "paper_numbers.py"))
    sys.argv = [sys.argv[0], "medic"]
    LC = runpy.run_path(str(HERE / "learned_cascade.py"))
G = P["G"]
sys.path.insert(0, str(M5VLM / "revision/human_evaluation"))
import analyze_human_eval as H  # noqa: E402

T, ids, q_run, cost = H.TIERS, G["ids"], G["q_run"], G["cost"]
first = {r["image_id"]: r for r in map(json.loads, open(R / "live_run/00_judge.jsonl"))}
second = {r["image_id"]: r for p in sorted(glob.glob(str(R / "live_run_rescored/judge_*.jsonl"))) for r in map(json.loads, open(p))}
assert len(second) == 600 and all(r["status"] == "ok" for r in second.values()) and set(second) == set(first)
assert all(second[i]["tier_outputs"][t]["text"] == first[i]["tier_outputs"][t]["text"] for i in ids for t in T)
print("second run:", sorted({(r["usage"]["resolved_model"], r["usage"]["system_fingerprint"]) for r in second.values()}))

txt = {k: {(i, t): H.text_score(src[i]["scores"][t]) for i in ids for t in T} for k, src in (("first", first), ("second", second))}
assert all(abs(txt["first"][i, t] - q_run[i][t]["text_score"]) < 1e-6 for i in ids for t in T)  # Table I used these
Q = {k: {i: {t: round(0.3 * q_run[i][t]["cat_score"] + 0.3 * q_run[i][t]["sev_score"] + 0.4 * v[i, t], 8) for t in T} for i in ids}
     for k, v in txt.items()}
pairs = [(i, t) for i in ids for t in T]
a, b = (np.array([Q[k][i][t] >= H.Q_TH for i, t in pairs]) for k in ("first", "second"))
x, y = (np.array([txt[k][p] for p in pairs]) for k in ("first", "second"))
print(f"\n=== 1,800 tier responses, first vs second scoring")
print(f"Q-Sat agreement {np.mean(a == b):.1%} ({np.sum(a != b)} changed: {np.sum(a & ~b)} pass->fail, {np.sum(~a & b)} fail->pass), "
      f"kappa {H.kappa(a.astype(int), b.astype(int), k=2):.2f}")
print(f"text score: Spearman {spearmanr(x, y)[0]:.3f}, mean |diff| {np.mean(np.abs(x - y)):.3f}, identical {np.mean(x == y):.1%}")
for t in T:
    m = np.array([p[1] == t for p in pairs])
    print(f"  {t:6s} mean text {x[m].mean():.3f} -> {y[m].mean():.3f}, Q-Sat {100 * a[m].mean():.1f}% -> {100 * b[m].mean():.1f}%, "
          f"changed {np.sum(a[m] != b[m])}")
crit = [(first[i]["scores"][t][c], second[i]["scores"][t][c]) for i, t in pairs for c in H.CRIT]
print(f"criterion scores (0-5) identical {np.mean([u == v for u, v in crit]):.1%}, within 1 point {np.mean([abs(u - v) <= 1 for u, v in crit]):.1%}")

# Table I methods (routes fixed) under both scorings; the reference is rebuilt per scoring as in judge_sensitivity.py
frugal = next(r[2] for n, r in LC["rows"].items() if n.startswith("Learned cascade text+labels, closest-cost"))  # Table I row
routes = {"Fixed@M5": ["m5_a"] * 600, "Fixed@Pi": ["pi_a"] * 600, "Fixed@Jetson": ["jetson"] * 600,
          f"Conf-Only ({G['C1']},{G['C2']})": G["conf"], "FrugalGPT cascade": list(frugal), "Primary-Only": P["tiers"],
          "M5-Pi Cascade": [t if t != "jetson" else "pi_a" for t in G["ours"]], "Ours": G["ours"]}
sat = {}
for k in Q:
    ref = [min((t for t in T if max(Q[k][i].values()) - Q[k][i][t] <= H.TAU + 1e-12), key=lambda t: cost[i][t]["energy_j"]) for i in ids]
    for name, r in {**routes, "Reference": ref}.items():
        sat[k, name] = np.array([Q[k][i][t] >= H.Q_TH for i, t in zip(ids, r)])
print("\n=== Table I Q-Sat (%), first scoring -> second scoring (routes fixed; reference rebuilt)")
for name in [*routes, "Reference"]:
    s1, s2 = sat["first", name], sat["second", name]
    print(f"  {name:22s} {100 * s1.mean():5.1f} -> {100 * s2.mean():5.1f}  ({np.sum(s1 != s2)} images changed)")
print("paired differences, second scoring (first scoring in brackets):")
for other in ("Fixed@Pi", f"Conf-Only ({G['C1']},{G['C2']})", "Fixed@Jetson", "Primary-Only", "FrugalGPT cascade"):
    out = []
    for k in ("second", "first"):
        o, s = sat[k, "Ours"], sat[k, other]
        n10, n01 = int(np.sum(o & ~s)), int(np.sum(~o & s))
        out.append(f"{100 * (o.mean() - s.mean()):+.1f} pp, McNemar p {binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0:.3f}")
    print(f"  Ours - {other:20s} {out[0]}  [{out[1]}]")
