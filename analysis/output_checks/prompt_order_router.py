#!/usr/bin/env python3
"""Prompt order: do the routing inputs and the routers' decisions depend on which M5 prompt was tried first?

    /opt/anaconda3/bin/python3 analysis/output_checks/prompt_order_router.py [n_random]   # default 20 random subsets

Read-only; prints only. Data and Table I accounting come from analysis/results_table.py (run with
runpy, output suppressed); the first-prompt rule from prompt_order_provenance.profiling_first_prompt() (primary-first =
even index within the category's sorted file names, backup-first = odd; the 9 images re-collected in a later pass are
left out of both groups).
  Q3. Stage-1 features of the profiling images by first prompt, against the 600 final-run answers (runtime: primary
      first) and, as a same-image contrast, the May all-tier answers of the same 600 images. Effect sizes: standardized
      mean difference (SMD) and total variation distance (TVD); reference = 95th percentile over 200 random splits of
      the profiling pool into groups of the same two sizes.
  Q4. Both routers retrained with the deployed recipe (profiling/code/pipeline/06_export_routers.py: common.
      make_dt depth 4; Stage 1 on all training images, END@M5 = reference M5; Stage 2 on the training images whose
      reference is Pi or Jetson, END@Pi = reference Pi) on (a) primary-first, (b) backup-first, (c) n_random random
      subsets of the pool with the size of (a), and (c') their complements (size of (b)); replayed on the 600 final-run feature
      vectors exactly as stage2_mismatch.py part C. First checks that the deployed joblib files and the recipe
      retrained on all 5,526 images reproduce the run's routes.
"""
import contextlib
import io
import runpy
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
M5VLM = HERE.parents[1]
sys.path.insert(0, str(HERE))
N_RAND = int(sys.argv.pop(1)) if len(sys.argv) > 1 else 20  # popped: results_table.py reads argv[1] as its run dir
from prompt_order_provenance import may_records, profiling_first_prompt  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(M5VLM / "analysis/results_table.py"))
prof, pids, y, run, ids, Qr, ours, alltier = (G[k] for k in ("prof", "pids", "y", "run", "ids", "Q", "ours", "alltier"))
Qr = Qr["run"]
make_dt, to_matrix, sf1, sf2, per_image, Q_TH = (G[k] for k in ("make_dt", "to_matrix", "stage1_features",
                                                               "stage2_features", "per_image", "Q_TH"))
POLICY = M5VLM / "deployment/routers"
N_NULL, SEED = 200, 20261119
NUM = ("m5a_sev", "m5_conf_mean", "m5_conf_min", "m5_sev_range", "m5_cat_agree", "m5_sev_agree")
DISC = ("m5a_cat", "m5a_sev", "m5_cat_agree", "m5_sev_agree")


# ---------------- groups
fp = profiling_first_prompt()
pos = {i: j for j, i in enumerate(pids)}
A = np.array(sorted(pos[i] for i in pids if fp[i] == "primary"))
B = np.array(sorted(pos[i] for i in pids if fp[i] == "backup"))
pool = np.concatenate([A, B])
m5 = {i: {k: run[i]["m5"][k] for k in ("m5_a", "m5_b", "m5_c")} for i in ids}
rank, late, _, _, may = may_records()


def feats(recs):
    rows = [sf1(r) for r in recs]
    for r, rec in zip(rows, recs):
        r.update({f"conf_{a}": rec[f"m5_{a}"].get("conf") or 0 for a in "abc"})
    return rows


P = feats([prof[i] for i in pids])  # aligned with pids
F = feats([m5[i] for i in ids])  # aligned with ids
M = feats([may[i] for i in ids])
even = [j for j, i in enumerate(ids) if i not in late and rank[i] % 2 == 0]
odd = [j for j, i in enumerate(ids) if i not in late and rank[i] % 2 == 1]


def col(rows, k):
    return np.array([r[k] for r in rows], dtype=object if k == "m5a_cat" else float)


def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def tvd(a, b):
    ca, cb = Counter(np.round(a, 3) if a.dtype != object else a), Counter(np.round(b, 3) if b.dtype != object else b)
    return 0.5 * sum(abs(ca[v] / len(a) - cb[v] / len(b)) for v in set(ca) | set(cb))


def describe(rows):
    cat = Counter(col(rows, "m5a_cat"))
    sev = Counter(col(rows, "m5a_sev"))
    return (f"n={len(rows):4d} | sev mean {col(rows, 'm5a_sev').mean():.2f} (0/1/2 % {100 * sev[0] / len(rows):.0f}/"
            f"{100 * sev[1] / len(rows):.0f}/{100 * sev[2] / len(rows):.0f}) | conf mean {col(rows, 'm5_conf_mean').mean():.2f} "
            f"min {col(rows, 'm5_conf_min').mean():.2f} (A/B/C {'/'.join(f'{col(rows, c).mean():.2f}' for c in ('conf_a', 'conf_b', 'conf_c'))}) | "
            f"cat agree {col(rows, 'm5_cat_agree').mean():.3f} sev agree {col(rows, 'm5_sev_agree').mean():.3f} | "
            f"M5_A category % " + " ".join(f"{k[:5]} {100 * v / len(rows):.0f}" for k, v in sorted(cat.items())))


def contrast(name, r1, r2, ref=None):
    parts = [f"SMD {k} {smd(col(r1, k), col(r2, k)):+.2f}" for k in NUM]
    parts += [f"TVD {k} {tvd(col(r1, k), col(r2, k)):.3f}" for k in DISC]
    print(f"   {name}\n      " + ", ".join(parts))


print("Q3. STAGE-1 FEATURES (M5_A category/severity, conf over A/B/C, agreement = majority-vote fraction)")
PA, PB = [P[j] for j in A], [P[j] for j in B]
for name, rows in (("profiling primary-first", PA), ("profiling backup-first", PB), ("final run, all 600", F),
                   ("May all-tier, even index", [M[j] for j in even]), ("final run, same even images", [F[j] for j in even]),
                   ("May all-tier, odd index", [M[j] for j in odd]), ("final run, same odd images", [F[j] for j in odd])):
    print(f"   {name:28s} {describe(rows)}")
rng = np.random.default_rng(SEED)
arr = {k: col([P[j] for j in pool], k) for k in set(NUM) | set(DISC)}
null_smd, null_tvd = {k: [] for k in NUM}, {k: [] for k in DISC}
for _ in range(N_NULL):
    perm = rng.permutation(len(pool))
    g1, g2 = perm[:len(A)], perm[len(A):]
    for k in NUM:
        null_smd[k].append(abs(smd(arr[k][g1], arr[k][g2])))
    for k in DISC:
        null_tvd[k].append(tvd(arr[k][g1], arr[k][g2]))
print("   effect sizes (first minus second):")
contrast("profiling backup-first vs primary-first", PB, PA)
print("      random-split 95th percentile: " + ", ".join(f"|SMD| {k} {np.percentile(null_smd[k], 95):.2f}" for k in NUM) +
      ", " + ", ".join(f"TVD {k} {np.percentile(null_tvd[k], 95):.3f}" for k in DISC))
contrast("final run vs profiling primary-first", F, PA)
contrast("final run vs profiling backup-first", F, PB)
contrast("same images, prompt differs: May odd (backup first) vs final run odd (primary first)",
         [M[j] for j in odd], [F[j] for j in odd])
contrast("same images, same prompt order: May even vs final run even (run-to-run only)",
         [M[j] for j in even], [F[j] for j in even])
QA = G["Q"]["all-tier"]
print("   M5_A answer Q-Sat (Q >= 0.65), same images: " + " | ".join(
    f"{name}: May {100 * np.mean([QA[ids[j]]['m5_a'] >= Q_TH for j in js]):.1f}%, "
    f"final run {100 * np.mean([Qr[ids[j]]['m5_a'] >= Q_TH for j in js]):.1f}%" for name, js in (("even", even), ("odd", odd))))

# ---------------- Q4
print("\nQ4. ROUTERS RETRAINED ON PROMPT-ORDER SUBSETS, REPLAYED ON THE 600 FINAL-RUN FEATURE VECTORS")
x1, k1, c1i, n1i = to_matrix([sf1(prof[i]) for i in pids])
x2, k2, c2i, n2i = to_matrix([sf2(prof[i]) for i in pids])
h1 = [[sf1(m5[i])[k] for k in k1] for i in ids]
h2 = []
for i in ids:  # stage2_mismatch.py part C: the run's logged Stage-2 features, else M5 run answers + all-tier Pi answers
    f = (run[i].get("stage2") or {}).get("features") or sf2(
        {**m5[i], "pi_a": alltier[i]["pi_a"], "pi_b": alltier[i]["pi_b"], "pi_a_conf_detail": alltier[i].get("pi_a_conf_detail") or {}})
    h2.append([f[k] for k in k2])


def fit(idx):
    idx = np.sort(idx)
    m1 = make_dt(c1i, n1i, depth=4).fit([x1[j] for j in idx], (y[idx] == "m5_a").astype(int))
    i2 = idx[y[idx] != "m5_a"]
    m2 = make_dt(c2i, n2i, depth=4).fit([x2[j] for j in i2], (y[i2] == "pi_a").astype(int))
    return m1, m2


def replay(m1, m2):
    return ["m5_a" if a else ("pi_a" if b else "jetson") for a, b in zip(m1.predict(h1), m2.predict(h2))]


def metrics(tiers):
    c = per_image(tiers, True).mean(axis=0)  # costs incl. auxiliary devices, as Ours in Table I
    d = Counter(tiers)
    return {"agree": sum(t == o for t, o in zip(tiers, ours)),
            "s1agree": sum((t == "m5_a") == (o == "m5_a") for t, o in zip(tiers, ours)),
            "qsat": 100 * np.mean([Qr[i][t] >= Q_TH for i, t in zip(ids, tiers)]),
            "lat": c[0], "kib": c[1], "j": c[2], "m5": d["m5_a"], "split": f"{d['m5_a']}/{d['pi_a']}/{d['jetson']}"}


def show(name, m):
    print(f"   {name:34s} {m['agree']:4d}/600 {m['s1agree']:4d}/600  {m['qsat']:5.1f} {m['lat']:6.1f} {m['kib']:5.0f} "
          f"{m['j']:6.1f}   {m['split']}")


dep = [joblib.load(POLICY / f) for f in ("stage1_m5_router.joblib", "stage2_pi_router.joblib")]
assert dep[0]["feature_keys"] == k1 and dep[1]["feature_keys"] == k2
dep_routes = replay(dep[0]["model"], dep[1]["model"])
all_routes = replay(*fit(np.arange(len(pids))))
print(f"   deployed joblib files reproduce the run's routes on {sum(a == b for a, b in zip(dep_routes, ours))}/600; "
      f"recipe retrained on all {len(pids)} images reproduces them on {sum(a == b for a, b in zip(all_routes, ours))}/600")
assert all_routes == ours, "the retraining recipe no longer reproduces the deployed routes; Q4 below would be off"
print(f"   training groups: (a) primary-first {len(A)}, (b) backup-first {len(B)}, pool {len(pool)} "
      f"(reference labels: (a) {dict(Counter(y[A]))}, (b) {dict(Counter(y[B]))})")
print(f"   {'router (training set)':34s} {'same route':>8s} {'same S1':>8s}  {'Q-Sat':>5s} {'lat s':>6s} {'KiB':>5s} {'J':>6s}   M5/Pi/Jet")
show("deployed (run)", metrics(ours))
ra, rb = replay(*fit(A)), replay(*fit(B))
ma, mb = metrics(ra), metrics(rb)
show("(a) primary-first", ma)
show("(b) backup-first", mb)
rng = np.random.default_rng(SEED)
mc, mcc, pair = [], [], []
for _ in range(N_RAND):
    perm = rng.permutation(pool)
    rc, rcc = replay(*fit(perm[:len(A)])), replay(*fit(perm[len(A):]))
    mc.append(metrics(rc))
    mcc.append(metrics(rcc))
    pair.append(sum(s == t for s, t in zip(rc, rcc)))
for name, ms in ((f"(c) {N_RAND} random, size of (a)", mc), ("(c') complements, size of (b)", mcc)):
    print(f"   {name:34s} " + "  ".join(f"{k} {np.mean([m[k] for m in ms]):.1f} [{min(m[k] for m in ms):.1f}, "
                                          f"{max(m[k] for m in ms):.1f}]" for k in ("agree", "s1agree", "qsat", "lat", "kib", "j", "m5")))
for name, mx, ms in (("(a) among (c)", ma, mc), ("(b) among (c')", mb, mcc)):
    print(f"   {name}: number of random subsets below / equal: " + ", ".join(
        f"{k} {sum(m[k] < mx[k] for m in ms)}/{sum(m[k] == mx[k] for m in ms)}" for k in ("agree", "qsat", "lat", "kib", "j", "m5")))
print(f"   (a) vs (b): same route on {sum(s == t for s, t in zip(ra, rb))}/600 (random pairs (c) vs (c'): "
      f"{min(pair)}-{max(pair)}, median {np.median(pair):.0f})")
for k, unit in (("qsat", "pp"), ("lat", "s"), ("kib", "KiB"), ("j", "J")):
    d = [abs(m[k] - n[k]) for m, n in zip(mc, mcc)]
    print(f"      {k:4s} (a) - (b) {ma[k] - mb[k]:+6.1f} {unit}; |(c) - (c')| median {np.median(d):.1f}, max {max(d):.1f}, "
          f"larger than |(a) - (b)| in {sum(x > abs(ma[k] - mb[k]) for x in d)}/{len(d)}")
