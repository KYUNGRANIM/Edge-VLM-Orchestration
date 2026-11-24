#!/usr/bin/env python3
"""Table I rows and Ours-vs-baseline comparisons with audited images left out, beside the full held-out set.

    python3 -B analysis/exclusion_sensitivity.py [medic|eidseg]

medic : the six held-out images with a copy in the profiling set (2 byte-identical, 4 near-duplicates confirmed by eye).
eidseg: the 50 pilot images, all inside the EIDSeg held-out 600, on which the label rules were fixed.
Same accounting and statistics as paired_statistics.py (Newcombe method 10, exact McNemar, 2,000 paired bootstrap draws).
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
M5VLM = HERE.parents[0]
DATASET = sys.argv[1] if len(sys.argv) > 1 else "medic"
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = [str(HERE / "paired_statistics.py"), DATASET]
    R = runpy.run_path(str(HERE / "paired_statistics.py"))
    sys.argv = [sys.argv[0]]
    X = runpy.run_path(str(HERE / "paper_numbers.py") if DATASET == "medic"
                       else str(M5VLM / "revision/eidseg/runtime/pipeline/paper_numbers_eidseg.py"))
rows, ids, Qr, Q_TH, newcombe, L = R["rows"], list(R["ids"]), R["Qr"], R["Q_TH"], R["newcombe"], R["L"]
n = len(ids)
sat = lambda tiers: np.array([Qr[i][t] >= Q_TH for i, t in zip(ids, tiers)])  # noqa: E731
ours = list(L["ours"])
if DATASET == "medic":
    cost_of = L["per_image"]
    extra = {"Fixed@M5": (["m5_a"] * n, False), "Primary-Only": (list(X["tiers"]), False),
             "M5-Pi Cascade": ([t if t != "jetson" else "pi_a" for t in ours], True), "Reference": (list(X["ref"]), False)}
    drop = {"908175535611707392_0__0c55de3f13.jpg", "910179107568324609_0__6ae837207f.jpg",   # byte-identical
            "930032829244805120_0__285059af01.jpg", "919639781432164352_0__040a6e7308.jpg",   # near-duplicates
            "910525745973415938_0__0b33cf24a2.jpg", "905630797926944768_0__1b0325f3ac.jpg"}
else:
    assert list(X["ids"]) == ids
    cost_of = X["per_image"]
    extra = {k: X["rows"][k] for k in ("Fixed@M5", "Primary-Only", "M5-Pi Cascade", "Reference")}
    drop = {i for i in ids if int(i.split("__")[0]) <= 50}  # pilot50 = held-out 000001-000050
    pilot = Path("/Volumes/CORSAIR/EIDSeg_eval/pilot50")
    if pilot.is_dir():
        assert drop == {p.name for p in pilot.rglob("*.jpg") if not p.name.startswith("._")}
assert len(drop) == (6 if DATASET == "medic" else 50) and drop <= set(ids)

pick = {"Fixed@M5": None, "Fixed@Pi": "Fixed@Pi (runs M5 first)", "Direct@Pi": "Direct@Pi",
        "Fixed@Jetson": "Fixed@Jetson (runs M5, Pi first)", "Direct@Jetson": "Direct@Jetson",
        "Confidence-Only": next(k for k in rows if k.startswith("Confidence-Only")),
        "FrugalGPT cascade": next(k for k in rows if k.startswith("Learned cascade text+labels, closest-cost")),
        "Primary-Only": None, "M5-Pi Cascade": None, "Ours": "Ours", "Reference": None}
M = {}
for name, key in pick.items():
    if key:
        s, c, _ = rows[key]
    else:
        tiers, aux = extra[name]
        s, c = sat(tiers), cost_of(tiers, aux)
    M[name] = (np.asarray(s, bool), np.asarray(c, float))
keep = np.array([i not in drop for i in ids])
print(f"{DATASET.upper()}: {n} held-out images, {int((~keep).sum())} left out -> {int(keep.sum())}")
print(f"{'':20s} {'all ' + str(n) + ': Q-Sat  lat   KiB    J':>36s} | {'without them: Q-Sat  lat   KiB    J':>36s}")
for name, (s, c) in M.items():
    f = lambda m: f"{100 * s[m].mean():6.1f} {c[m, 0].mean():5.1f} {c[m, 1].mean():5.0f} {c[m, 2].mean():6.1f}"  # noqa: E731
    print(f"  {name:18s} {f(np.ones(n, bool)):>36s} | {f(keep):>36s}")

rng = np.random.default_rng(20261104)
k = int(keep.sum())
IX = [rng.integers(k, size=k) for _ in range(2000)]
so, co = (x[keep] for x in M["Ours"])
print("\nOurs minus baseline without the left-out images: Q-Sat pp [Newcombe 95%], McNemar p | latency, comm, energy "
      "relative change of the mean [paired bootstrap 95%]")
for name in pick:
    if name in ("Ours", "Reference"):
        continue
    sb, cb = (x[keep] for x in M[name])
    lo, hi, n10, n01 = newcombe(so, sb)
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    rel = []
    for col in (0, 1, 2):
        if cb[:, col].mean() == 0:
            rel.append("    n/a        ")
            continue
        v = [100 * (co[ix, col].mean() / cb[ix, col].mean() - 1) for ix in IX]
        rel.append(f"{100 * (co[:, col].mean() / cb[:, col].mean() - 1):+6.1f}% [{np.percentile(v, 2.5):+6.1f}, {np.percentile(v, 97.5):+6.1f}]")
    print(f"  vs {name:18s} {100 * (so.mean() - sb.mean()):+5.1f} [{lo:+5.1f}, {hi:+5.1f}] p {p:.3f} | "
          + " | ".join(rel))
