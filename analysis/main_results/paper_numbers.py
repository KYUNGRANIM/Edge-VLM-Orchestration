#!/usr/bin/env python3
"""The manuscript's quoted results, recomputed on the MEDIC rerun (run answer set).

    python3 analysis/main_results/paper_numbers.py

Reuses results_table.py for the accounting (costs, auxiliary devices, Confidence-Only matching, answer sets) and
repeats the first submission's analyses on the rerun:
McNemar, Q-Sat threshold sensitivity, M5-Pi Cascade, Stage-1 share of the reference gap, the Fig. 4 cutoff
sweep, matched-Q-Sat latency against all 25 Confidence-Only pairs, the Fig. 5 permutation test, the router
training statistics, and Primary-Only Routing (its original code is not in the repository; see that section).
"""
import json
import runpy
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest

G = runpy.run_path(str(Path(__file__).resolve().parent / "results_table.py"))
ids, run, cost, Qr, alltier = G["ids"], G["run"], G["cost"], G["Q"]["run"], G["alltier"]
ours, conf, ref, C1, C2 = G["ours"], G["conf"], G["ref_run"], G["C1"], G["C2"]
M5VLM, Q_TH, per_image, composite = G["M5VLM"], G["Q_TH"], G["per_image"], G["composite"]
n, TI = len(ids), {"m5_a": 0, "pi_a": 1, "jetson": 2}
POL = M5VLM / "deployment/routers"


def sat(tiers, th=Q_TH):
    return np.array([Qr[i][t] >= th for i, t in zip(ids, tiers)])


def summ(tiers, aux):
    lat, kib, j = per_image(tiers, aux).mean(axis=0)
    comp = np.mean([composite(cost[i], t, aux) for i, t in zip(ids, tiers)])
    return dict(q=100 * sat(tiers).mean(), lat=lat, kib=kib, j=j, comp=comp)


def red(base, new):  # % reduction of new relative to base
    return 100 * (base - new) / base


def mcnemar(a, b):
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return n10, n01, binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0


fixed = {t: [t] * n for t in TI}
cascade = [t if t != "jetson" else "pi_a" for t in ours]
S = {"Fixed-Termination@M5": summ(fixed["m5_a"], False), "Fixed-Termination@Pi": summ(fixed["pi_a"], False),
     "Fixed-Termination@Jetson": summ(fixed["jetson"], False), f"Confidence-Only ({C1},{C2})": summ(conf, False),
     "M5-Pi Cascade": summ(cascade, True), "Consistency-Aware (Ours)": summ(ours, True),
     "Quality-Cost Reference": summ(ref, False)}
dist = {"Ours": Counter(ours), "Reference": Counter(ref), "Confidence-Only": Counter(conf)}
print("\n================ TABLE I (run answer set; Q-Sat %, latency s, comm KiB, energy J) ================")
for k, s in S.items():
    print(f"  {k:28s} {s['q']:5.1f} {s['lat']:6.1f} {s['kib']:5.0f} {s['j']:6.1f}   (unrounded {s['q']:.3f} {s['lat']:.3f} {s['kib']:.2f} {s['j']:.3f})")
print("  tiers M5/Pi/Jetson:", {k: f"{v['m5_a']}/{v['pi_a']}/{v['jetson']}" for k, v in dist.items()})

o, p, jt, co = S["Consistency-Aware (Ours)"], S["Fixed-Termination@Pi"], S["Fixed-Termination@Jetson"], S[f"Confidence-Only ({C1},{C2})"]
so = sat(ours)
print("\n================ TEXT NUMBERS (from unrounded means) ================")
n10, n01, pv = mcnemar(so, sat(fixed["pi_a"]))
print(f"Ours vs Fixed@Pi: Q-Sat {o['q']:.1f} vs {p['q']:.1f} ({o['q'] - p['q']:+.2f} pp), exact McNemar p = {pv:.3f} "
      f"(discordant {n10}/{n01}); latency -{red(p['lat'], o['lat']):.1f}%, energy -{red(p['j'], o['j']):.1f}%, "
      f"comm +{-red(p['kib'], o['kib']):.1f}%")
d = [100 * (sat(ours, t).mean() - sat(fixed['pi_a'], t).mean()) for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)]
print(f"Q-Sat threshold 0.50..0.75, Ours - Fixed@Pi (pp): {[round(x, 1) for x in d]} -> range {min(d):+.1f} to {max(d):+.1f}")
print(f"Ours vs Fixed@Jetson: latency -{red(jt['lat'], o['lat']):.1f}%, comm -{red(jt['kib'], o['kib']):.1f}%, "
      f"energy -{red(jt['j'], o['j']):.1f}%, Q-Sat {o['q'] - jt['q']:+.1f} pp")
n10, n01, pv = mcnemar(so, sat(conf))
print(f"Ours vs Confidence-Only ({C1},{C2}): Q-Sat {o['q']:.1f} vs {co['q']:.1f}, p = {pv:.3f} (discordant {n10}/{n01}); "
      f"latency -{red(co['lat'], o['lat']):.1f}%, energy {-red(co['j'], o['j']):+.1f}%, comm {-red(co['kib'], o['kib']):+.1f}%")
c = S["M5-Pi Cascade"]
print(f"M5-Pi Cascade: Q-Sat {o['q']:.1f} -> {c['q']:.1f}, saves {o['lat'] - c['lat']:.1f} s and {o['j'] - c['j']:.1f} J")

# ---- gap to the reference, by stage
sr = sat(ref)
loss, gain = sr & ~so, ~sr & so
s1 = np.array([(r == "m5_a") != (t == "m5_a") for r, t in zip(ref, ours)])
early = np.array([TI[t] < TI[r] for r, t in zip(ref, ours)])
print(f"Reference {100 * sr.mean():.1f} vs Ours {100 * so.mean():.1f}: gap {100 * (sr.mean() - so.mean()):.1f} pp; "
      f"losses {loss.sum()} (premature {np.sum(loss & early)}, over-escalation {np.sum(loss & ~early)}), recoveries {gain.sum()}; "
      f"Stage-1 share of losses {100 * np.sum(loss & s1) / loss.sum():.1f}% ({np.sum(loss & s1)}/{loss.sum()})")

# ---- Fig. 5 permutation test (seed 42, 5000 draws, counts fixed, as in the first submission)
tidx = np.array([TI[t] for t in ours])
SAT = np.array([[Qr[i][t] >= Q_TH for t in TI] for i in ids])
E = np.array([[cost[i][t]["energy_j"] for t in TI] for i in ids])
aux_e = per_image(ours, True)[:, 2].mean() - E[np.arange(n), tidx].mean()
rng = np.random.default_rng(42)
qn, en = np.empty(5000), np.empty(5000)
for k in range(5000):
    perm = rng.permutation(tidx)
    qn[k], en[k] = 100 * SAT[np.arange(n), perm].mean(), E[np.arange(n), perm].mean() + aux_e
q0 = 100 * SAT[np.arange(n), tidx].mean()
print(f"Permutation (counts {Counter(ours)['m5_a']}/{Counter(ours)['pi_a']}/{Counter(ours)['jetson']}): deployed {q0:.2f}% vs "
      f"randomized mean {qn.mean():.2f}%, one-sided p = {(np.sum(qn >= q0) + 1) / 5001:.4f}; "
      f"energy {o['j']:.1f} vs {en.mean():.1f} J")

# ---- router training statistics (profiling, corrected labels)
meta = next(json.loads(l) for l in open(M5VLM / "profiling/results/pipeline_outputs/04_routing_main.jsonl") if '"model_meta"' in l)
b1 = [f["stage1_balanced_accuracy"] for f in meta["fold_stats"]]
b2 = [f["stage2_balanced_accuracy"] for f in meta["fold_stats"]]
y = G["y"]
print(f"Training: fold balanced accuracy Stage 1 {min(b1):.3f}-{max(b1):.3f}, Stage 2 {min(b2):.3f}-{max(b2):.3f}; "
      f"Stage 1 n={len(y)}, Stage 2 n={int(np.sum(y != 'm5_a'))} (reference {dict(Counter(y))}); "
      f"Confidence-Only thresholds c1={C1}, c2={C2}")

# ---- Fig. 4 cutoff sweep with the deployed (retrained) trees (as in the first submission)
p1, p2 = joblib.load(POL / "stage1_m5_router.joblib"), joblib.load(POL / "stage2_pi_router.joblib")
sf1, sf2 = G["stage1_features"], G["stage2_features"]
m5 = {i: {k: run[i]["m5"][k] for k in ("m5_a", "m5_b", "m5_c")} for i in ids}
pm5 = dict(zip(ids, p1["model"].predict_proba([[sf1(m5[i])[k] for k in p1["feature_keys"]] for i in ids])[:, 1]))
ppi, src = {}, {}
for i in ids:
    f = (run[i].get("stage2") or {}).get("features")
    src[i] = "run" if f else "all-tier"
    if not f:
        a = alltier[i]
        f = sf2({**m5[i], "pi_a": a["pi_a"], "pi_b": a["pi_b"], "pi_a_conf_detail": a.get("pi_a_conf_detail") or {}})
    ppi[i] = float(p2["model"].predict_proba([[f[k] for k in p2["feature_keys"]]])[0, 1])
ok1 = sum((pm5[i] >= 0.5) == (run[i]["stage1"]["decision"] == "END@M5") for i in ids)
dep2 = [i for i in ids if src[i] == "run"]
ok2 = sum((ppi[i] >= 0.5) == (run[i]["stage2"]["decision"] == "END@Pi") for i in dep2)
print(f"\nFig. 4 replay check: Stage 1 {ok1}/{n}, Stage 2 {ok2}/{len(dep2)} match the run's decisions")


def assign(t1, t2):
    return ["m5_a" if pm5[i] >= t1 else ("pi_a" if ppi[i] >= t2 else "jetson") for i in ids]


print("  arm      t1   t2   Q-Sat  composite  latency  energy   M5/Pi/Jet")
for arm, grid in (("Stage-1", [(t, 0.5) for t in (0.20, 0.35, 0.50, 0.65, 0.70, 0.80, 0.90)]),
                  ("Stage-2", [(0.5, t) for t in (0.10, 0.20, 0.35, 0.50, 0.65, 0.80)])):
    for t1, t2 in grid:
        tiers = assign(t1, t2)
        s, dd = summ(tiers, True), Counter(tiers)
        print(f"  {arm}  {t1:.2f} {t2:.2f}  {s['q']:5.1f}  {s['comp']:.4f}   {s['lat']:5.1f}  {s['j']:6.1f}   "
              f"{dd['m5_a']}/{dd['pi_a']}/{dd['jetson']}")
print(f"  fixed points: " + ", ".join(f"{k} Q {S[k]['q']:.1f} comp {S[k]['comp']:.4f}" for k in
                                       ("Fixed-Termination@M5", "Fixed-Termination@Pi", "Fixed-Termination@Jetson", "Quality-Cost Reference")))

# ---- matched Q-Sat: minimum latency over the 25 Confidence-Only pairs vs the 35-point Ours cutoff grid
run_pi = {i: ((run[i].get("pi") or {}).get("pi_a") or {}) for i in ids}
cpts = []
for a in range(1, 6):
    for b in range(1, 6):
        t = [G["conf_pick"](run[i]["m5"]["m5_a"].get("conf") or 0, (run_pi[i] or alltier[i]["pi_a"]).get("conf") or 0, a, b)
             for i in ids]
        cpts.append(summ(t, False))
opts = [summ(assign(t1, t2), True) for t1 in (0.20, 0.35, 0.50, 0.65, 0.70, 0.80, 0.90) for t2 in (0.20, 0.35, 0.50, 0.65, 0.80)]
print("\nmatched Q-Sat (min latency s / min energy J among points with Q-Sat >= floor):")
for floor in (60, 62, 64, round(o["q"], 1), 66, 68, 70):
    c_ = [x for x in cpts if x["q"] >= floor]
    o_ = [x for x in opts if x["q"] >= floor]
    f = lambda xs, k: f"{min(x[k] for x in xs):6.1f}" if xs else "   n/a"
    print(f"  >= {floor:5.1f}%: Confidence-Only lat {f(c_, 'lat')} J {f(c_, 'j')} | Ours-grid lat {f(o_, 'lat')} J {f(o_, 'j')}")

# ---- Primary-Only Routing: trees on the main agents' answers only (no M5_B, M5_C or Pi_B), trained on the corrected
# profiling labels. On the pre-correction data, dropping the cross-tier features (M5_A vs Pi_A) as well reproduces
# the submitted row exactly (55.2% / 38.9 s / 145 KB / 237.6 J, tiers 311/169/120); keeping them gives 54.8% / 38.6 s /
# 140 KB / 234.2 J. The manuscript text says only the within-tier features are removed, so both are printed.
from common import safe_float, safe_int  # noqa: E402  (results_table.py put profiling/code/pipeline on sys.path)


def po1(rec):
    a = rec["m5_a"]
    return {"m5a_cat": a.get("cat", ""), "m5a_sev": safe_int(a.get("sev")), "m5a_conf": safe_int(a.get("conf"))}


def po2(rec, cross):
    out, pa = po1(rec), rec["pi_a"]
    sev = safe_int(pa.get("sev"))
    out.update(pia_cat=pa.get("cat", ""), pia_sev=sev, pia_conf=safe_int(pa.get("conf")),
               pia_min_logprob=safe_float((rec.get("pi_a_conf_detail") or {}).get("min_logprob"), -5.0))
    if cross:
        out.update(m5a_pia_cat_match=int(rec["m5_a"].get("cat") == pa.get("cat")),
                   m5a_pia_sev_diff=abs(safe_int(rec["m5_a"].get("sev")) - sev))
    return out


prof, pids, y = G["prof"], G["pids"], G["y"]
make_dt, to_matrix = G["make_dt"], G["to_matrix"]
print("\nPrimary-Only Routing (no auxiliary devices; run answer set):")
for cross, label in ((False, "all consistency features dropped (as submitted)"), (True, "cross-tier kept (as the text says)")):
    x1, k1, c1i, n1i = to_matrix([po1(prof[i]) for i in pids])
    m1 = make_dt(c1i, n1i, depth=4).fit(x1, (y == "m5_a").astype(int))
    tr2 = [j for j in range(len(pids)) if y[j] != "m5_a"]
    x2, k2, c2i, n2i = to_matrix([po2(prof[pids[j]], cross) for j in tr2])
    m2 = make_dt(c2i, n2i, depth=4).fit(x2, (y[tr2] == "pi_a").astype(int))
    tiers = []
    for i in ids:
        m5a = run[i]["m5"]["m5_a"]
        if m1.predict([[po1({"m5_a": m5a})[k] for k in k1]])[0]:
            tiers.append("m5_a")
            continue
        pi = ({"pi_a": run_pi[i], "pi_a_conf_detail": run[i].get("pi_a_conf_detail")} if run_pi[i] else
              {"pi_a": alltier[i]["pi_a"], "pi_a_conf_detail": alltier[i].get("pi_a_conf_detail")})
        f = po2({"m5_a": m5a, **pi}, cross)
        tiers.append("pi_a" if m2.predict([[f[k] for k in k2]])[0] else "jetson")
    s, dd = summ(tiers, False), Counter(tiers)
    n10, n01, pv = mcnemar(so, sat(tiers))
    print(f"  {label:48s} {s['q']:5.1f} {s['lat']:6.1f} {s['kib']:5.0f} {s['j']:6.1f}   {dd['m5_a']}/{dd['pi_a']}/{dd['jetson']}"
          f"   (vs Ours: Q-Sat {s['q'] - o['q']:+.1f} pp, McNemar p = {pv:.2f} [{n10}/{n01}], "
          f"comm {-red(o['kib'], s['kib']):+.1f}%, energy {-red(o['j'], s['j']):+.1f}%)")
