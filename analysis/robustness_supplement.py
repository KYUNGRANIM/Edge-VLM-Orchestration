#!/usr/bin/env python3
"""Robustness supplement (MEDIC): the differences event_bootstrap.py leaves out, and how repeatable the answers are.

    /opt/anaconda3/bin/python3 analysis/robustness_supplement.py

A. Communication for the Table I baselines, and the two Table I ablations (Primary-Only, M5-Pi Cascade):
   Ours minus each, 95% bootstrap with image-level resampling, plus event-level resampling (main grouping of
   event_bootstrap.py) when the CORSAIR manifest is mounted.
B. Repeatability: the routing run re-executed every tier it reached on the same images as the earlier all-tier
   trace. Per tier: identical answer text, Q-Sat agreement and Q-Sat on each execution; then the fixed tiers on the
   two answer sets. Ours itself cannot be re-scored on the other execution: its routes were chosen from the run's
   own answers, so swapping in another execution's answers breaks that link (the 60.3% in results_table.py [B]).
"""
import contextlib
import csv
import io
import os
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "analysis"))
from event_split import event_of  # noqa: E402

MANIFEST = HERE.parents[0] / "heldout/results/medic_manifest_test.csv"  # MEDIC test-split manifest
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = [str(HERE / "learned_cascade.py"), "medic"]
    L = runpy.run_path(str(HERE / "learned_cascade.py"))
    sys.argv = [str(HERE / "paper_numbers.py")]
    P = runpy.run_path(str(HERE / "paper_numbers.py"))
rows, ids, G, per_image, ours = L["rows"], L["ids"], P["G"], L["per_image"], L["ours"]
assert list(G["ids"]) == list(ids)
Qr, Qa, Q_TH, TIERS = G["Q"]["run"], G["Q"]["all-tier"], G["Q_TH"], G["TIERS"]
so, co, _ = rows["Ours"]
pick = {"Fixed@Pi": "Fixed@Pi (runs M5 first)", "Direct@Pi": "Direct@Pi", "Fixed@Jetson": "Fixed@Jetson (runs M5, Pi first)",
        "Direct@Jetson": "Direct@Jetson", "Confidence-Only": next(k for k in rows if k.startswith("Confidence-Only")),
        "Learned cascade (text+labels)": next(k for k in rows if k.startswith("Learned cascade text+labels, closest-cost"))}
B = 2000
rng = np.random.default_rng(20261115)


def sums(groups):
    names = sorted(set(groups))
    g = np.array([names.index(x) for x in groups])
    W = np.stack([np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names)) for _ in range(B)])
    return lambda v: W @ np.bincount(g, weights=np.asarray(v, float), minlength=len(names))


LEVELS = {"image": sums([f"i{n}" for n in range(len(ids))])}
if MANIFEST.exists():  # same main grouping as event_bootstrap.py: event, else source folder
    man = {Path(r["local_image_path"]).name: r["image_path"] for r in csv.DictReader(open(MANIFEST))}
    LEVELS["event"] = sums([event_of(man[i]) or os.path.dirname(man[i]) for i in ids])
else:
    print(f"(no {MANIFEST}: image-level intervals only)")
ci = lambda v: f"[{np.percentile(v, 2.5):+6.1f}, {np.percentile(v, 97.5):+6.1f}]"


def rel(col, cb):
    point = 100 * (co[:, col].mean() / cb[:, col].mean() - 1)
    return f"{point:+6.1f}% " + " ".join(f"{lv} {ci(100 * (S(co[:, col]) / S(cb[:, col]) - 1))}" for lv, S in LEVELS.items())


print("A. Ours minus baseline, relative change of the mean, 95% bootstrap")
print("  communication (KiB per request)")
for name, key in pick.items():
    print(f"    vs {name:30s} {co[:, 1].mean():5.0f} vs {rows[key][1][:, 1].mean():5.0f}: {rel(1, rows[key][1])}")
po = list(P["tiers"])
casc = [t if t != "jetson" else "pi_a" for t in ours]
for name, tiers, aux in (("Primary-Only", po, False), ("M5-Pi Cascade", casc, True)):
    sb, cb = np.array([Qr[i][t] >= Q_TH for i, t in zip(ids, tiers)]), per_image(tiers, aux)
    dq = " ".join(f"{lv} {ci(100 * (S(so) - S(sb)) / S(np.ones(len(ids))))}" for lv, S in LEVELS.items())
    print(f"  vs {name} ({100 * sb.mean():.1f}% / {cb[:, 0].mean():.1f} s / {cb[:, 1].mean():.0f} KiB / {cb[:, 2].mean():.1f} J)")
    print(f"    Q-Sat {100 * (so.mean() - sb.mean()):+5.1f} pp {dq}")
    for col, m in ((0, "latency"), (1, "communication"), (2, "energy")):
        print(f"    {m:13s} {rel(col, cb)}")

print("\nB. the same tier executed twice on the same image (routing run vs earlier all-tier trace)")
run, alltier, run_pi = G["run"], G["alltier"], G["run_pi"]
text = {"m5_a": lambda i: run[i]["m5"]["m5_a"].get("text"), "pi_a": lambda i: (run_pi[i] or {}).get("text"),
        "jetson": lambda i: ((run[i].get("jetson") or {}).get("jetson") or {}).get("text")}
for t in TIERS:
    both = [i for i in ids if text[t](i)]
    same = np.array([text[t](i).strip() == (alltier[i][t].get("text") or "").strip() for i in both])
    a, b = np.array([Qr[i][t] >= Q_TH for i in both]), np.array([Qa[i][t] >= Q_TH for i in both])
    diff = f"{100 * np.mean(a[~same] == b[~same]):5.1f}%" if (~same).any() else "  n/a"
    print(f"  {t:6s} n={len(both):3d}: identical text {100 * same.mean():5.1f}% | Q-Sat run {100 * a.mean():5.1f}% vs earlier "
          f"{100 * b.mean():5.1f}% | same pass/fail {100 * np.mean(a == b):5.1f}% (identical text {100 * np.mean(a[same] == b[same]):5.1f}%, "
          f"different text {diff})")
print("  fixed tiers over all 600, run answer set vs earlier all-tier answers: " + ", ".join(
    f"{t} {100 * np.mean([Qr[i][t] >= Q_TH for i in ids]):.1f} vs {100 * np.mean([Qa[i][t] >= Q_TH for i in ids]):.1f}" for t in TIERS))

# ---------------- C. the deployed recipe replayed on the earlier execution
# Both routers retrained on all profiling images (as profiling_requirements.py, which reproduces the deployed routes), then
# fed the earlier execution's own answers, so routes and scores come from the same execution.
sys.path.insert(0, str(G["M5VLM"] / "profiling/code/pipeline"))
from common import make_dt, stage1_features, stage2_features, to_matrix  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

prof, pids, y = L["prof"], L["pids"], L["y"]
x1, k1, c1, n1 = to_matrix([stage1_features(prof[i]) for i in pids])
x2, k2, c2, n2 = to_matrix([stage2_features(prof[i]) for i in pids])
m1 = make_dt(c1, n1, depth=4).fit(x1, (y == "m5_a").astype(int))
s2 = [j for j in range(len(pids)) if y[j] != "m5_a"]
m2 = make_dt(c2, n2, depth=4).fit([x2[j] for j in s2], (y[s2] == "pi_a").astype(int))


def route(recs):
    a = m1.predict([[stage1_features(r)[k] for k in k1] for r in recs])
    b = m2.predict([[stage2_features(r)[k] for k in k2] for r in recs])
    return ["m5_a" if p else ("pi_a" if q else "jetson") for p, q in zip(a, b)]


run_recs = [{**run[i]["m5"], "pi_a": run_pi[i] or alltier[i]["pi_a"], "pi_b": (run[i].get("pi") or {}).get("pi_b") or alltier[i]["pi_b"],
             "pi_a_conf_detail": run[i].get("pi_a_conf_detail") or alltier[i].get("pi_a_conf_detail") or {}} for i in ids]
again = route(run_recs)
prev = route([alltier[i] for i in ids])
print(f"\nC. deployed recipe on the run's answers reproduces the deployed routes on {sum(a == b for a, b in zip(again, ours))}/600")
cnt = lambda t: "/".join(str(sum(x == k for x in t)) for k in TIERS)
cp, cfix = per_image(prev, True), per_image(["pi_a"] * len(ids))
sp, sf = np.array([Qa[i][t] >= Q_TH for i, t in zip(ids, prev)]), np.array([Qa[i]["pi_a"] >= Q_TH for i in ids])
n10, n01 = int(np.sum(sp & ~sf)), int(np.sum(~sp & sf))
S = LEVELS["image"]
print(f"  earlier execution, routed and scored on its own answers: Q-Sat {100 * sp.mean():.1f}% (tiers {cnt(prev)}; same route as "
      f"the run for {sum(a == b for a, b in zip(prev, ours))}/600) vs Fixed@Pi {100 * sf.mean():.1f}%: "
      f"{100 * (sp.mean() - sf.mean()):+.1f} pp {ci(100 * (S(sp) - S(sf)) / S(np.ones(len(ids))))}, McNemar p "
      f"{binomtest(n10, n10 + n01, 0.5).pvalue:.2f} ({n10}/{n01})")
for col, m in ((0, "latency"), (1, "communication"), (2, "energy")):
    print(f"    {m:13s} {100 * (cp[:, col].mean() / cfix[:, col].mean() - 1):+6.1f}% "
          f"{ci(100 * (S(cp[:, col]) / S(cfix[:, col]) - 1))}  (run: {100 * (co[:, col].mean() / cfix[:, col].mean() - 1):+.1f}%)")

# ---------------- D. reply-letter numbers with paired_statistics.py's own methods and bootstrap draws
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = [str(HERE / "paired_statistics.py"), "medic"]
    R = runpy.run_path(str(HERE / "paired_statistics.py"))
newcombe, wilson, boot = R["newcombe"], R["wilson"], R["boot"]
assert (R["so"] == so).all()
cb = rows[pick["Fixed@Pi"]][1]
lo, hi = boot(lambda ix: 100 * (co[ix, 1].mean() / cb[ix, 1].mean() - 1))
print(f"\nD. communication vs Fixed@Pi with r2_7's draws: {100 * (co[:, 1].mean() / cb[:, 1].mean() - 1):+.1f}% [{lo:+.1f}, {hi:+.1f}]")
sf = rows[pick["Fixed@Pi"]][0]
p1, p2, n = so.mean(), sf.mean(), len(so)
(l1, u1), (l2, u2) = wilson(int(so.sum()), n), wilson(int(sf.sum()), n)
print(f"  Q-Sat vs Fixed@Pi, paired Newcombe [{newcombe(so, sf)[0]:+.1f}, {newcombe(so, sf)[1]:+.1f}] vs the same data treated as two "
      f"independent samples [{100 * (p1 - p2 - np.hypot(p1 - l1, u2 - p2)):+.1f}, {100 * (p1 - p2 + np.hypot(u1 - p1, p2 - l2)):+.1f}]")
for name, tiers in (("Primary-Only", po), ("M5-Pi Cascade", casc)):
    sb = np.array([Qr[i][t] >= Q_TH for i, t in zip(ids, tiers)])
    lo, hi, n10, n01 = newcombe(so, sb)
    print(f"  Q-Sat vs {name:13s} {100 * (so.mean() - sb.mean()):+.1f} pp, paired Newcombe [{lo:+.1f}, {hi:+.1f}], "
          f"exact McNemar p {binomtest(n10, n10 + n01, 0.5).pvalue:.3f} ({n10}/{n01})")

# ---------------- E. multiplicity over every Ours-vs-X test this revision reports (MEDIC)
# Q-Sat: exact McNemar; latency, communication, energy: Wilcoxon signed-rank (as robustness.py). Direct@Pi and
# Direct@Jetson share the Fixed rows' answers, so their Q-Sat tests are not repeated.
from scipy.stats import wilcoxon  # noqa: E402

fam = {"Fixed@M5": (np.array([Qr[i]["m5_a"] >= Q_TH for i in ids]), per_image(["m5_a"] * len(ids))),
       **{k: (rows[v][0], rows[v][1]) for k, v in pick.items()},
       "Learned cascade (labels only)": rows[next(k for k in rows if k.startswith("Learned cascade labels only, closest-cost"))][:2],
       "Primary-Only": (np.array([Qr[i][t] >= Q_TH for i, t in zip(ids, po)]), per_image(po, False)),
       "M5-Pi Cascade": (np.array([Qr[i][t] >= Q_TH for i, t in zip(ids, casc)]), per_image(casc, True))}
tests = []
for name, (sb, cb) in fam.items():
    if name not in ("Direct@Pi", "Direct@Jetson"):
        n10, n01 = int(np.sum(so & ~sb)), int(np.sum(~so & sb))
        tests.append((name, "Q-Sat", binomtest(n10, n10 + n01, 0.5).pvalue))
    for col, m in ((0, "latency"), (1, "communication"), (2, "energy")):
        d = co[:, col] - cb[:, col]
        tests.append((name, m, wilcoxon(d).pvalue if np.any(d) else 1.0))


def holm(ps):
    out, run_max = np.empty(len(ps)), 0.0
    for r, k in enumerate(np.argsort(ps)):
        run_max = max(run_max, min(1.0, (len(ps) - r) * ps[k]))
        out[k] = run_max
    return out


ps = np.array([p for *_, p in tests])
h = holm(ps)
three = [k for k, (n, m, _) in enumerate(tests) if m == "Q-Sat" and n in ("Fixed@Pi", "Primary-Only", "M5-Pi Cascade")]
h3 = holm(ps[three])
print(f"\nE. Holm over all {len(tests)} tests; raw significant {int(np.sum(ps < 0.05))}, after Holm {int(np.sum(h < 0.05))}")
for k, (n, m, p) in enumerate(tests):
    if (p < 0.05) != (h[k] < 0.05) or k in three:
        print(f"  {n:30s} {m:13s} raw p {p:.3g} -> Holm {h[k]:.3g}" + (f" | Holm over the three Q-Sat tests {h3[three.index(k)]:.3g}" if k in three else ""))
tab = [k for k, (n, _, _) in enumerate(tests) if n != "Learned cascade (labels only)"]  # rows that appear in Table I
h_tab = holm(ps[tab])
print(f"  Table I comparisons only ({len(tab)} tests): significant raw {int(np.sum(ps[tab] < 0.05))}, after Holm "
      f"{int(np.sum(h_tab < 0.05))}; Q-Sat after Holm: " + ", ".join(
          f"{tests[k][0]} {h_tab[j]:.3g}" for j, k in enumerate(tab) if tests[k][1] == "Q-Sat"))
print("  not significant before adjustment: " + ", ".join(f"{n} {m} ({p:.2g})" for n, m, p in tests if p >= 0.05))
