#!/usr/bin/env python3
"""The GPT-4o text judge against the ten human raters, and what its tier bias does to the rerun's Table I.

    python3 analysis/judge_validation/judge_sensitivity.py

Human scores: revision/human_evaluation/ratings/rater*.json (10 raters, the same 60 held-out images, 3 tier answers
each, blind A/B/C), scored with the judge's own text formula (revision/human_evaluation/analyze_human_eval.py).
Those answers are the all-tier collection's; the rerun's Table I is scored on the run's answers, so the bias
measured on the 60 images is applied to the 600 as a per-tier offset (assumption: the judge's leniency is a
property of each tier's model, not of the particular answer).
"""
import contextlib
import glob
import io
import itertools
import json
import runpy
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    P = runpy.run_path(str(HERE.parent / "main_results/paper_numbers.py"))
G = P["G"]
M5VLM = G["M5VLM"]
sys.path.insert(0, str(M5VLM / "revision/human_evaluation"))
import analyze_human_eval as H  # noqa: E402

T = H.TIERS
rng = np.random.default_rng(20261109)

# ---------------- human raters vs GPT-4o on the 60 images (all-tier answers)
key = json.load(open(M5VLM / "revision/human_evaluation/rating_page/human_eval_key.json"))["key"]
judge = {r["image_id"]: r for r in map(json.loads, open(H.JUDGE)) if r.get("type") == "gpt_text_judge" and r.get("status") == "ok"}
qc = {r["image_id"]: r for r in map(json.loads, open(H.OUT / "01_quality_components.jsonl")) if r["type"] == "quality_components"}
paths = sorted(glob.glob(str(M5VLM / "revision/human_evaluation/ratings/rater*.json")))
raters = [H.load_rater(p, key, judge) for p in paths]
hid = sorted(set.intersection(*(set(r) for r in raters)))
NR = len(raters)
gpt = {(i, t): H.text_score(judge[i]["scores"][t]) for i in hid for t in T}
htxt = [{(i, t): H.text_score(r[i][t]) for i in hid for t in T} for r in raters]
med = {i: {t: {k: statistics.median_low(r[i][t][k] for r in raters) for k in H.CRIT} for t in T} for i in hid}
mtxt = {(i, t): H.text_score(med[i][t]) for i in hid for t in T}
words = {(i, t): len(judge[i]["tier_outputs"][t]["text"].split()) for i in hid for t in T}
q60 = lambda i, t, x: round(0.3 * qc[i][t]["cat_score"] + 0.3 * qc[i][t]["sev_score"] + 0.4 * x, 8)  # noqa: E731
pairs = [(i, t) for i in hid for t in T]


def sat60(txt):
    return [int(q60(i, t, txt[i, t]) >= H.Q_TH) for i, t in pairs]


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort([x[p] for p in pairs])) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ci(fn, B=2000):
    vals = []
    for _ in range(B):
        s = rng.integers(len(hid), size=len(hid))
        vals.append(fn([hid[j] for j in s]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


k_gh = [H.kappa(sat60(h), sat60(gpt), k=2) for h in htxt]
k_hh = [H.kappa(sat60(htxt[a]), sat60(htxt[b]), k=2) for a, b in itertools.combinations(range(NR), 2)]
s_gh = [spearman(h, gpt) for h in htxt]
s_hh = [spearman(htxt[a], htxt[b]) for a, b in itertools.combinations(range(NR), 2)]
print(f"=== judge validation: {NR} raters x {len(hid)} images x 3 answers = {NR * len(pairs)} human ratings")
print(f"Q-Sat kappa  GPT vs each rater: mean {np.mean(k_gh):.2f} (range {min(k_gh):.2f}-{max(k_gh):.2f}) | "
      f"rater vs rater: mean {np.mean(k_hh):.2f} (range {min(k_hh):.2f}-{max(k_hh):.2f})")
print(f"text Spearman GPT vs each rater: mean {np.mean(s_gh):.2f} (range {min(s_gh):.2f}-{max(s_gh):.2f}) | "
      f"rater vs rater: mean {np.mean(s_hh):.2f} (range {min(s_hh):.2f}-{max(s_hh):.2f})")
agree = np.mean(np.equal(sat60(mtxt), sat60(gpt)))
print(f"median of {NR} raters vs GPT: Q-Sat agreement {agree:.1%}, kappa {H.kappa(sat60(mtxt), sat60(gpt), k=2):.2f}, "
      f"text Spearman {spearman(mtxt, gpt):.2f}")
off = {}
for t in T:
    f = lambda s, t=t: float(np.mean([gpt[i, t] - mtxt[i, t] for i in s]))  # noqa: E731
    g = lambda s, t=t: float(np.mean([gpt[i, t] - np.mean([h[i, t] for h in htxt]) for i in s]))  # noqa: E731
    off[t] = f(hid)
    lo, hi = ci(f)
    lo2, hi2 = ci(g)
    print(f"  {t:6s} GPT minus human text score: vs median {off[t]:+.3f} [{lo:+.3f}, {hi:+.3f}] | vs rater mean {g(hid):+.3f} "
          f"[{lo2:+.3f}, {hi2:+.3f}] | raters with GPT higher: {sum(np.mean([gpt[i, t] - h[i, t] for i in hid]) > 0 for h in htxt)}/{NR} "
          f"| words {np.mean([words[i, t] for i in hid]):.0f}")
d = [gpt[p] - mtxt[p] for p in pairs]
w = [words[p] for p in pairs]
print(f"length: corr(words, GPT minus median) {np.corrcoef(w, d)[0, 1]:+.2f} over all answers; within tier "
      + ", ".join(f"{t} {np.corrcoef([words[i, t] for i in hid], [gpt[i, t] - mtxt[i, t] for i in hid])[0, 1]:+.2f}" for t in T))
print("per criterion, median vs GPT, quadratic-weighted kappa: "
      + ", ".join(f"{k} {H.kappa([med[i][t][k] for i, t in pairs], [judge[i]['scores'][t][k] for i, t in pairs], weighted=True, k=6):.2f}"
                  for k in H.CRIT))

# ---------------- the methods on these 60 images by scorer (rerun routes, all-tier answers)
ids, ours, conf, po = G["ids"], G["ours"], G["conf"], P["tiers"]
cascade = [t if t != "jetson" else "pi_a" for t in ours]
route = {"Fixed@M5": dict.fromkeys(ids, "m5_a"), "Fixed@Pi": dict.fromkeys(ids, "pi_a"),
         "Fixed@Jetson": dict.fromkeys(ids, "jetson"), f"Conf-Only({G['C1']},{G['C2']})": dict(zip(ids, conf)),
         "Ours": dict(zip(ids, ours)), "Primary-Only": dict(zip(ids, po))}
scorers = {"GPT-4o": gpt, f"median of {NR}": mtxt}
print(f"\n=== Q-Sat (%) on the {len(hid)} human-rated images, rerun routes, all-tier answers")
print(" " * 16 + "".join(f"{s:>14s}" for s in scorers))
qs = {}
for name, r in route.items():
    qs[name] = [100 * np.mean([q60(i, r[i], txt[i, r[i]]) >= H.Q_TH for i in hid]) for txt in scorers.values()]
    print(f"{name:16s}" + "".join(f"{x:14.1f}" for x in qs[name]))

# every subset of raters: does the comparison between methods depend on who rated?
qr = np.array([[np.mean([q60(i, r[i], h[i, r[i]]) >= H.Q_TH for i in hid]) for r in route.values()] for h in htxt])
names = list(route)
J, Pi_, M5, C, O = (names.index(n) for n in ("Fixed@Jetson", "Fixed@Pi", "Fixed@M5", f"Conf-Only({G['C1']},{G['C2']})", "Ours"))
offj = np.array([[gpt[i, "jetson"] - h[i, "jetson"] for i in hid] for h in htxt])
boots = [rng.integers(len(hid), size=len(hid)) for _ in range(1000)]
khh = dict(zip(itertools.combinations(range(NR), 2), k_hh))
print("\n=== all rater subsets: k, n | Jetson offset range, CI>0 | kappa(GPT-H) minus kappa(H-H) within 0.05 | "
      "Jetson top, Ours<Conf, Ours>M5 | Ours vs Fixed@Pi higher/tie/lower")
for k in range(1, NR + 1):
    rows = []
    for S in map(list, itertools.combinations(range(NR), k)):
        dj = offj[S].mean(axis=0)
        m = qr[S].mean(axis=0)
        dk = np.mean([k_gh[r] for r in S]) - np.mean([khh[p] for p in itertools.combinations(S, 2)]) if k > 1 else np.nan
        rows.append((dj.mean(), np.percentile([dj[b].mean() for b in boots], 2.5) > 0, abs(dk) <= .05 if k > 1 else np.nan,
                     m[J] == m.max(), m[O] < m[C], m[O] > m[M5], np.sign(round(m[O] - m[Pi_], 9))))
    a = np.array(rows, dtype=float)
    c = Counter(a[:, 6])
    print(f"  {k:2d} {len(rows):4d} | {a[:, 0].min():+.3f}..{a[:, 0].max():+.3f} {int(a[:, 1].sum()):4d} | "
          f"{'  n/a' if k == 1 else f'{int(np.nansum(a[:, 2])):5d}'} | {int(a[:, 3].sum()):4d} {int(a[:, 4].sum()):4d} {int(a[:, 5].sum()):4d} | "
          f"{int(c[1])}/{int(c[0])}/{int(c[-1])}")

# ---------------- the rerun's Table I (600 images, run answers) with the measured bias removed
q_run, cost = G["q_run"], G["cost"]
print(f"\n=== 600 images, run answer set: Q-Sat (%) with the judge's per-tier bias (vs the median rater) removed")
variants = {"GPT-4o as is": dict.fromkeys(T, 0.0), "Jetson offset removed": {"m5_a": 0.0, "pi_a": 0.0, "jetson": off["jetson"]},
            "all three offsets removed": off}
methods = {"Fixed@M5": ["m5_a"] * len(ids), "Fixed@Pi": ["pi_a"] * len(ids), "Fixed@Jetson": ["jetson"] * len(ids),
           f"Conf-Only({G['C1']},{G['C2']})": conf, "Ours": ours, "Primary-Only": po, "M5-Pi Cascade": cascade}
for vname, o in variants.items():
    Q = {i: {t: round(0.3 * q_run[i][t]["cat_score"] + 0.3 * q_run[i][t]["sev_score"]
                      + 0.4 * max(0.0, q_run[i][t]["text_score"] - o[t]), 8) for t in T} for i in ids}
    ref = []
    for i in ids:
        best = max(Q[i].values())
        ref.append(min((t for t in T if best - Q[i][t] <= H.TAU + 1e-12), key=lambda t: cost[i][t]["energy_j"]))
    sat = {n: np.array([Q[i][t] >= H.Q_TH for i, t in zip(ids, tiers)]) for n, tiers in {**methods, "Reference": ref}.items()}
    print(f"  {vname}: " + ", ".join(f"{n} {100 * s.mean():.1f}" for n, s in sat.items())
          + f" | reference tiers {'/'.join(str(Counter(ref)[t]) for t in T)}")
    for b in ("Fixed@Pi", f"Conf-Only({G['C1']},{G['C2']})", "Fixed@Jetson", "Primary-Only"):
        n10, n01 = int(np.sum(sat["Ours"] & ~sat[b])), int(np.sum(~sat["Ours"] & sat[b]))
        print(f"      Ours - {b:16s} {100 * (sat['Ours'].mean() - sat[b].mean()):+.1f} pp, McNemar p = "
              f"{binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0:.3f}")
