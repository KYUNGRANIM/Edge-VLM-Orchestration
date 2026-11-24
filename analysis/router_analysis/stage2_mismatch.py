#!/usr/bin/env python3
"""Stage 2 is trained on images whose reference is Pi or Jetson, but deployed it also receives the images
Stage 1 escalates by mistake (reference M5). How large is that mismatch, and does it hurt?

    python3 analysis/router_analysis/stage2_mismatch.py

A. Mismatch: what Stage 2 is trained on vs what out-of-fold Stage 1 forwards to it (profiling, the 5 folds of
   Table I), and what the deployed Stage 1 forwarded in the MEDIC rerun (reference rebuilt on the run answers).
B. Stage 2 on the images out-of-fold Stage 1 forwards, against always-Pi, always-Jetson and a "matched" Stage 2
   trained on forwarded images (inner out-of-fold Stage 1 inside each training split). Stage-2 label: end at Pi
   when Pi is within tau of the better of Pi and Jetson. It equals the reference for Pi/Jetson-reference images
   and extends it to M5-reference ones.
C. Both Stage-2 recipes trained on all profiling images and replayed on the 600 held-out run answers.
"""
import contextlib
import io
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(HERE.parent / "main_results/results_table.py"))
prof, pref, pcost, pids, y = G["prof"], G["pref"], G["pcost"], G["pids"], G["y"]
make_dt, to_matrix, sf1, sf2 = G["make_dt"], G["to_matrix"], G["stage1_features"], G["stage2_features"]
run, ids, Qr, ref_run, ours, alltier = G["run"], G["ids"], G["Q"]["run"], G["ref_run"], G["ours"], G["alltier"]
TAU, Q_TH, N_BOOT = G["TAU"], G["Q_TH"], G["N_BOOT"]
REFS = ("m5_a", "pi_a", "jetson")

x1, k1, c1i, n1i = to_matrix([sf1(prof[i]) for i in pids])
x2, k2, c2i, n2i = to_matrix([sf2(prof[i]) for i in pids])
pq = [pref[i]["quality"] for i in pids]


def stage2_label(qs):
    """1 = end at Pi: Pi within tau of the better of Pi and Jetson."""
    return int(max(qs["pi_a"], qs["jetson"]) - qs["pi_a"] <= TAU + 1e-12)


z = np.array([stage2_label(qs) for qs in pq])
yi = (y == "pi_a").astype(int)  # the deployed Stage-2 target
assert all(z[j] == yi[j] for j in range(len(pids)) if y[j] != "m5_a")


def fit1(idx):
    return make_dt(c1i, n1i, depth=4).fit([x1[j] for j in idx], (y[idx] == "m5_a").astype(int))


def fit2(idx, labels):
    return make_dt(c2i, n2i, depth=4).fit([x2[j] for j in idx], labels[idx])


def split(idx):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(idx)), y[idx])


def mcnemar(a, b):
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0


def boot_ci(d):
    rng = np.random.default_rng(20261110)
    bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ---------------- out-of-fold Stage 1 and both Stage-2 recipes (outer folds = Table I's)
everyone = np.arange(len(pids))
s1 = np.zeros(len(pids), dtype=int)  # 1 = Stage 1 ends at M5
dep = np.zeros(len(pids), dtype=int)  # 1 = deployed-recipe Stage 2 ends at Pi
mat = np.zeros(len(pids), dtype=int)  # 1 = matched-recipe Stage 2 ends at Pi
for tr, te in split(everyone):
    s1[te] = fit1(tr).predict([x1[j] for j in te])
    dep[te] = fit2(tr[y[tr] != "m5_a"], yi).predict([x2[j] for j in te])
    inner = np.zeros(len(tr), dtype=int)
    for a, b in split(tr):
        inner[b] = fit1(tr[a]).predict([x1[j] for j in tr[b]])
    mat[te] = fit2(tr[inner == 0], z).predict([x2[j] for j in te])
route = np.array(["m5_a" if a else ("pi_a" if b else "jetson") for a, b in zip(s1, dep)], dtype=object)
assert (route == G["oof"]).all(), "out-of-fold routes differ from results_table.py"

fwd = np.where(s1 == 0)[0]
train2 = np.where(y != "m5_a")[0]
print(f"A. MISMATCH (profiling {len(pids)} images, 5-fold out-of-fold Stage 1)")
print(f"   Stage 2 is trained on {len(train2)} images (reference Pi or Jetson): {dict(Counter(y[train2]))}")
cf = Counter(y[fwd])
print(f"   out-of-fold Stage 1 forwards {len(fwd)} images: reference M5 {cf['m5_a']} ({100 * cf['m5_a'] / len(fwd):.1f}%), "
      f"Pi {cf['pi_a']} ({100 * cf['pi_a'] / len(fwd):.1f}%), Jetson {cf['jetson']} ({100 * cf['jetson'] / len(fwd):.1f}%)")
missed = np.sum((y != "m5_a") & (s1 == 1))
print(f"   of Stage 2's {len(train2)} training images, {missed} ({100 * missed / len(train2):.1f}%) never reach it "
      f"(Stage 1 ends them at M5); {np.sum((y != 'm5_a') & (s1 == 0))} do")
fh = [j for j, t in enumerate(ours) if t != "m5_a"]
ch = Counter(ref_run[j] for j in fh)
print(f"   MEDIC rerun: deployed Stage 1 forwarded {len(fh)}/600; reference (rebuilt on run answers) M5 {ch['m5_a']} "
      f"({100 * ch['m5_a'] / len(fh):.1f}%), Pi {ch['pi_a']} ({100 * ch['pi_a'] / len(fh):.1f}%), "
      f"Jetson {ch['jetson']} ({100 * ch['jetson'] / len(fh):.1f}%)")


# ---------------- B. Stage 2 on the forwarded profiling images
def summary(dec, idx, qset, costf):
    tiers = ["pi_a" if d else "jetson" for d in dec]
    sat = np.array([qset[j][t] >= Q_TH for j, t in zip(idx, tiers)])
    c = np.array([costf(j, t) for j, t in zip(idx, tiers)])
    return sat, c


def pcostf(j, t):
    c = pcost[pids[j]][t]
    return c["latency_ms"] / 1000, c["energy_j"]


print(f"\nB. STAGE 2 ON THE {len(fwd)} OUT-OF-FOLD FORWARDED PROFILING IMAGES (energy/latency: main path, cumulative)")
print(f"   {'recipe':34s} {'to Pi':>6s} {'bal.acc':>7s} | {'Q-Sat':>6s} {'lat s':>6s} {'J':>6s} | to Pi by reference M5 / Pi / Jetson")
recipes = {"deployed (trained on ref Pi/Jetson)": dep[fwd], "matched (trained on forwarded)": mat[fwd],
           "always Pi": np.ones(len(fwd), int), "always Jetson": np.zeros(len(fwd), int), "Stage-2 label (oracle)": z[fwd]}
res = {}
for name, d in recipes.items():
    sat, c = summary(d, fwd, pq, pcostf)
    res[name] = (sat, c)
    by = " / ".join(f"{100 * d[y[fwd] == r].mean():5.1f}" for r in REFS)
    print(f"   {name:34s} {100 * d.mean():6.1f} {balanced_accuracy_score(z[fwd], d):7.3f} | {100 * sat.mean():6.1f} "
          f"{c[:, 0].mean():6.1f} {c[:, 1].mean():6.1f} | {by}")
for part, mask in (("reference M5 (the mismatch)", y[fwd] == "m5_a"), ("reference Pi/Jetson", y[fwd] != "m5_a")):
    line = []
    for name in ("deployed (trained on ref Pi/Jetson)", "matched (trained on forwarded)"):
        sat, c = res[name]
        d = recipes[name][mask]
        line.append(f"{name.split()[0]} bal.acc {balanced_accuracy_score(z[fwd][mask], d):.3f}, Q-Sat {100 * sat[mask].mean():.1f}%, "
                    f"{c[mask, 1].mean():.1f} J")
    print(f"   {part} (n={mask.sum()}): " + " | ".join(line))
(sd, cd), (sm, cm) = res["deployed (trained on ref Pi/Jetson)"], res["matched (trained on forwarded)"]
lo, hi = boot_ci(cm[:, 1] - cd[:, 1])
print(f"   matched - deployed: Q-Sat {100 * (sm.mean() - sd.mean()):+.1f} pp (McNemar p {mcnemar(sm, sd):.2f}), "
      f"energy {(cm[:, 1] - cd[:, 1]).mean():+.1f} J [{lo:+.1f}, {hi:+.1f}]")
for name, d2 in (("deployed", dep), ("matched", mat)):
    tiers = ["m5_a" if a else ("pi_a" if b else "jetson") for a, b in zip(s1, d2)]
    sat = np.mean([pq[j][t] >= Q_TH for j, t in enumerate(tiers)])
    e = np.mean([pcost[pids[j]][t]["energy_j"] for j, t in enumerate(tiers)])
    print(f"   end to end, all {len(pids)} out of fold, {name:8s}: Q-Sat {100 * sat:.1f}%, energy {e:.1f} J, "
          f"M5/Pi/Jetson {'/'.join(str(Counter(tiers)[t]) for t in REFS)}")

# ---------------- C. trained on all profiling images, replayed on the held-out run answers
m1 = fit1(everyone)
m2d, m2m = fit2(train2, yi), fit2(fwd, z)
m5 = {i: {k: run[i]["m5"][k] for k in ("m5_a", "m5_b", "m5_c")} for i in ids}
h1 = [[sf1(m5[i])[k] for k in k1] for i in ids]
h2 = []
for i in ids:
    f = (run[i].get("stage2") or {}).get("features") or sf2(
        {**m5[i], "pi_a": alltier[i]["pi_a"], "pi_b": alltier[i]["pi_b"], "pi_a_conf_detail": alltier[i].get("pi_a_conf_detail") or {}})
    h2.append([f[k] for k in k2])
e1 = m1.predict(h1)
tiers = {name: ["m5_a" if a else ("pi_a" if b else "jetson") for a, b in zip(e1, m.predict(h2))]
         for name, m in (("deployed", m2d), ("matched", m2m))}
print(f"\nC. HELD-OUT 600 (run answers; costs incl. auxiliary devices, as Table I). Retrained deployed recipe "
      f"reproduces the deployed routes on {sum(a == b for a, b in zip(tiers['deployed'], ours))}/600")
out = {}
for name, tr_ in tiers.items():
    sat = np.array([Qr[i][t] >= Q_TH for i, t in zip(ids, tr_)])
    c = G["per_image"](tr_, True)
    out[name] = (sat, c)
    d = Counter(tr_)
    print(f"   {name:8s}: Q-Sat {100 * sat.mean():.1f}%, latency {c[:, 0].mean():.1f} s, energy {c[:, 2].mean():.1f} J, "
          f"M5/Pi/Jetson {d['m5_a']}/{d['pi_a']}/{d['jetson']}")
(sd, cd), (sm, cm) = out["deployed"], out["matched"]
lo, hi = boot_ci(cm[:, 2] - cd[:, 2])
lo2, hi2 = boot_ci(cm[:, 0] - cd[:, 0])
print(f"   matched - deployed: Q-Sat {100 * (sm.mean() - sd.mean()):+.1f} pp (McNemar p {mcnemar(sm, sd):.2f}), "
      f"latency {(cm[:, 0] - cd[:, 0]).mean():+.1f} s [{lo2:+.1f}, {hi2:+.1f}], energy {(cm[:, 2] - cd[:, 2]).mean():+.1f} J [{lo:+.1f}, {hi:+.1f}]; "
      f"{sum(a != b for a, b in zip(tiers['deployed'], tiers['matched']))} images change route")
for r in REFS:
    sel = [j for j in fh if ref_run[j] == r]
    to_pi = np.mean([ours[j] == "pi_a" for j in sel])
    sat = np.mean([Qr[ids[j]][ours[j]] >= Q_TH for j in sel])
    print(f"   deployed Stage 2 on the {len(sel)} forwarded held-out images with reference {r}: "
          f"{100 * to_pi:.1f}% end at Pi, Q-Sat {100 * sat:.1f}%")
