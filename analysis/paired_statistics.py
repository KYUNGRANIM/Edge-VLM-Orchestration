#!/usr/bin/env python3
"""Paired confidence intervals, a non-inferiority margin, and uncertainty of the latency/energy savings.

    python3 analysis/paired_statistics.py [medic|eidseg]

All comparisons pair the same 600 held-out images (run answer set, Table I accounting; rows from learned_cascade.py).
Q-Sat difference (Ours minus baseline): Newcombe's hybrid score interval for paired proportions (method 10, 1998)
and a paired bootstrap over images (2,000 resamples); exact McNemar test. Non-inferiority holds at margin M when the
lower 95% bound exceeds -M (one-sided alpha 0.025); the smallest such margin is minus the lower bound. Latency and
energy: paired bootstrap intervals of the mean difference and of the relative change of the means.
"""
import contextlib
import io
import runpy
import sys
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "medic"
sys.argv = [str(HERE / "learned_cascade.py"), DATASET]
with contextlib.redirect_stdout(io.StringIO()):
    L = runpy.run_path(str(HERE / "learned_cascade.py"))
rows, ids, Qr, Q_TH = L["rows"], L["ids"], L["Qr"], L["Q_TH"]
MARGIN, Z, B = 5.0, 1.959964, 2000
m5 = np.array([Qr[i]["m5_a"] >= Q_TH for i in ids])


def wilson(k, n):
    p, d = k / n, 1 + Z * Z / n
    c, h = (p + Z * Z / (2 * n)) / d, Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(x, y):
    """95% interval for mean(x) - mean(y), x and y paired booleans (Newcombe 1998, method 10)."""
    a, b, c, d = (int(np.sum(x & y)), int(np.sum(x & ~y)), int(np.sum(~x & y)), int(np.sum(~x & ~y)))
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    (l1, u1), (l2, u2) = wilson(a + b, n), wilson(a + c, n)
    den = sqrt((a + b) * (c + d) * (a + c) * (b + d))
    num = a * d - b * c
    phi = 0.0 if den == 0 else (max(num - n / 2, 0) if num > 0 else num) / den
    lo = p1 - p2 - sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    hi = p1 - p2 + sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return 100 * lo, 100 * hi, b, c


rng = np.random.default_rng(20261104)
IX = [rng.integers(len(ids), size=len(ids)) for _ in range(B)]


def boot(f):
    v = np.array([f(ix) for ix in IX])
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


so, co, _ = rows["Ours"]
pick = {"Fixed@Pi": "Fixed@Pi (runs M5 first)", "Direct@Pi": "Direct@Pi", "Fixed@Jetson": "Fixed@Jetson (runs M5, Pi first)",
        "Direct@Jetson": "Direct@Jetson", "Confidence-Only": next(k for k in rows if k.startswith("Confidence-Only")),
        "Learned cascade (text+labels)": next(k for k in rows if k.startswith("Learned cascade text+labels, closest-cost")),
        "Learned cascade (labels only)": next(k for k in rows if k.startswith("Learned cascade labels only, closest-cost"))}
fp = rows[pick["Fixed@Pi"]][0]
print(f"{DATASET.upper()} held-out {len(ids)}: Ours Q-Sat {100 * so.mean():.1f}%, Fixed@M5 {100 * m5.mean():.1f}%, "
      f"Fixed@Pi {100 * fp.mean():.1f}% (Pi's gain over M5 {100 * (fp.mean() - m5.mean()):.1f} pp; margin {MARGIN:g} pp)")
print("\nQ-Sat, Ours minus baseline (pp): Newcombe 95% | bootstrap 95% | McNemar p (Ours-only/baseline-only wins) | "
      f"non-inferior at {MARGIN:g} pp | smallest margin for non-inferiority")
for name, key in pick.items():
    sb, cb, _ = rows[key]
    d = 100 * (so.mean() - sb.mean())
    lo, hi, n10, n01 = newcombe(so, sb)
    blo, bhi = boot(lambda ix: 100 * (so[ix].mean() - sb[ix].mean()))
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    ni = "yes" if lo > -MARGIN else "no"
    print(f"  vs {name:30s} {100 * so.mean():5.1f} vs {100 * sb.mean():5.1f}: {d:+5.1f} [{lo:+5.1f}, {hi:+5.1f}] | "
          f"[{blo:+5.1f}, {bhi:+5.1f}] | p {p:.3f} ({n10}/{n01}) | {ni:3s} | {max(0.0, -lo):.1f} pp")
print("\nlatency and energy, Ours minus baseline: mean difference [95%] and relative change of the mean [95%]")
for name, key in pick.items():
    _, cb, _ = rows[key]
    out = []
    for col, unit in ((0, "s"), (2, "J")):
        a, b_ = co[:, col], cb[:, col]
        dlo, dhi = boot(lambda ix: a[ix].mean() - b_[ix].mean())
        rlo, rhi = boot(lambda ix: 100 * (a[ix].mean() / b_[ix].mean() - 1))
        out.append(f"{a.mean() - b_.mean():+7.1f} {unit} [{dlo:+7.1f}, {dhi:+7.1f}], {100 * (a.mean() / b_.mean() - 1):+6.1f}% "
                   f"[{rlo:+6.1f}, {rhi:+6.1f}]")
    print(f"  vs {name:30s} latency {out[0]} | energy {out[1]}")
