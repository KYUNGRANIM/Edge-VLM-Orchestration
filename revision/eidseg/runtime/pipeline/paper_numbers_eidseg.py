#!/usr/bin/env python3
"""EIDSeg held-out results computed the way the MEDIC rerun's Table I is (analysis/
paper_numbers.py): every method scored on the run answer set (the run's M5 answers; Pi/Jetson answers where the run
reached that tier, else the all-tier collection), the reference rebuilt on that set, Primary-Only as the manuscript
text defines it (cross-tier features kept), plus the permutation test and a split by whether the image's
earthquake also appears in MEDIC.

    python3 revision/eidseg/runtime/pipeline/paper_numbers_eidseg.py
"""
import contextlib
import csv
import io
import runpy
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(HERE / "results_table.py"))
M5VLM, ids, run, cost, alltier = G["M5VLM"], G["ids"], G["run"], G["cost"], G["alltier"]
Q_RUN, Q_ALL, ours, conf, C1, C2 = G["Q_RUN"], G["Q_ALL"], G["ours"], G["conf"], G["C1"], G["C2"]
from common import make_dt, safe_float, safe_int, to_matrix  # noqa: E402

T, TI, n = ("m5_a", "pi_a", "jetson"), {"m5_a": 0, "pi_a": 1, "jetson": 2}, len(ids)
AUX = (G["AUX_M5_ENERGY"], G["AUX_PI_ENERGY"], G["AUX_PI_OVERHEAD"])


def per_image(tiers, aux):
    out = []
    for i, t in zip(ids, tiers):
        c = cost[i][t]
        ovh = c["overhead_bytes"] + (AUX[2] if aux and t != "m5_a" else 0)
        eng = c["energy_j"] + ((AUX[0] + (AUX[1] if t != "m5_a" else 0)) if aux else 0)
        out.append((c["latency_ms"] / 1000, ovh / 1024, eng))
    return np.array(out)


def sat(tiers, Q=Q_RUN):
    return np.array([Q[i][t] >= 0.65 for i, t in zip(ids, tiers)])


def mcnemar(a, b):
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0


# Primary-Only (no M5_B, M5_C, Pi_B; cross-tier M5_A-Pi_A features kept, as the text says)
def po1(rec):
    a = rec["m5_a"]
    return {"m5a_cat": a.get("cat", ""), "m5a_sev": safe_int(a.get("sev")), "m5a_conf": safe_int(a.get("conf"))}


def po2(rec):
    out, pa = po1(rec), rec["pi_a"]
    sev = safe_int(pa.get("sev"))
    out.update(pia_cat=pa.get("cat", ""), pia_sev=sev, pia_conf=safe_int(pa.get("conf")),
               pia_min_logprob=safe_float((rec.get("pi_a_conf_detail") or {}).get("min_logprob"), -5.0),
               m5a_pia_cat_match=int(rec["m5_a"].get("cat") == pa.get("cat")),
               m5a_pia_sev_diff=abs(safe_int(rec["m5_a"].get("sev")) - sev))
    return out


prof, pids, y = G["prof"], G["pids"], G["y"]
x1, k1, c1, n1 = to_matrix([po1(prof[i]) for i in pids])
m1 = make_dt(c1, n1, depth=4).fit(x1, (y == "m5_a").astype(int))
tr2 = [j for j in range(len(pids)) if y[j] != "m5_a"]
x2, k2, c2, n2 = to_matrix([po2(prof[pids[j]]) for j in tr2])
m2 = make_dt(c2, n2, depth=4).fit(x2, (y[tr2] == "pi_a").astype(int))
po = []
for i in ids:
    m5a = run[i]["m5"]["m5_a"]
    if m1.predict([[po1({"m5_a": m5a})[k] for k in k1]])[0]:
        po.append("m5_a")
        continue
    rp = (run[i].get("pi") or {}).get("pi_a")
    pi = {"pi_a": rp, "pi_a_conf_detail": run[i].get("pi_a_conf_detail")} if rp else \
         {"pi_a": alltier[i]["pi_a"], "pi_a_conf_detail": alltier[i].get("pi_a_conf_detail")}
    f = po2({"m5_a": m5a, **pi})
    po.append("pi_a" if m2.predict([[f[k] for k in k2]])[0] else "jetson")

ref = []
for i in ids:
    best = max(Q_RUN[i].values())
    ref.append(min((t for t in T if best - Q_RUN[i][t] <= 0.10 + 1e-12), key=lambda t: cost[i][t]["energy_j"]))
cascade = [t if t != "jetson" else "pi_a" for t in ours]
rows = {"Fixed@M5": (["m5_a"] * n, False), "Fixed@Pi": (["pi_a"] * n, False), "Fixed@Jetson": (["jetson"] * n, False),
        f"Confidence-Only ({C1},{C2})": (conf, False), "Primary-Only": (po, False), "M5-Pi Cascade": (cascade, True),
        "Ours": (ours, True), "Reference": (ref, False)}
S = {}
print(f"EIDSeg held-out {n} images, run answer set (Q-Sat %, latency s, comm KiB, energy J, tiers M5/Pi/Jetson)")
for name, (tiers, aux) in rows.items():
    c, s, d = per_image(tiers, aux).mean(axis=0), sat(tiers), Counter(tiers)
    S[name] = (s, c)
    print(f"  {name:22s} {100 * s.mean():5.1f} {c[0]:6.1f} {c[1]:5.0f} {c[2]:6.1f}   {d['m5_a']}/{d['pi_a']}/{d['jetson']}")
print(f"  Ours scored on the all-tier answers instead: {100 * sat(ours, Q_ALL).mean():.1f}")

so, co = S["Ours"]
red = lambda b, x: 100 * (b - x) / b  # noqa: E731
for b in ("Fixed@Pi", "Fixed@Jetson", f"Confidence-Only ({C1},{C2})", "Primary-Only", "Fixed@M5"):
    sb, cb = S[b]
    rng = np.random.default_rng(20261109)
    d = (so.astype(float) - sb) * 100
    bs = [d[rng.integers(n, size=n)].mean() for _ in range(2000)]
    print(f"Ours vs {b:22s} Q-Sat {d.mean():+.1f} pp [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}], "
          f"p = {mcnemar(so, sb):.3f} | latency {-red(cb[0], co[0]):+.1f}%, comm {-red(cb[1], co[1]):+.1f}%, energy {-red(cb[2], co[2]):+.1f}%")

# permutation test (counts fixed, seed 42, 5000 draws), as Fig. 5
tidx = np.array([TI[t] for t in ours])
SAT = np.array([[Q_RUN[i][t] >= 0.65 for t in T] for i in ids])
rng = np.random.default_rng(42)
qn = np.array([100 * SAT[np.arange(n), rng.permutation(tidx)].mean() for _ in range(5000)])
q0 = 100 * SAT[np.arange(n), tidx].mean()
print(f"permutation: deployed {q0:.2f}% vs randomized mean {qn.mean():.2f}%, one-sided p = {(np.sum(qn >= q0) + 1) / 5001:.4f}")

# by whether the image's earthquake also appears in MEDIC (images are de-duplicated against MEDIC either way)
man = {r["local_image_path"].split("/")[-1]: r for r in csv.DictReader(open(M5VLM / "revision/eidseg/dataset/eidseg_manifest.csv"))}
grp = {"event also in MEDIC": "1", "event not in MEDIC": "0", "event unknown": ""}
for gname, v in grp.items():
    m = np.array([man[i]["event_in_medic"] == v for i in ids])
    print(f"  {gname:20s} n={m.sum():3d}: " + ", ".join(f"{b} {100 * S[b][0][m].mean():.1f}" for b in ("Fixed@M5", "Fixed@Pi", "Fixed@Jetson", "Ours")))
