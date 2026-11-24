#!/usr/bin/env python3
"""MEDIC event-cluster bootstrap of the Table I differences (needs the CORSAIR drive).

    /opt/anaconda3/bin/python3 analysis/event_bootstrap.py

Images of one disaster event move together when resampled, so intervals reflect between-event variation as well.
Events come from the MEDIC source paths (analysis/event_split.py's event_of); about half of the held-out images
(crawled, Google-image or Mouzannar sets) carry no event. Those are clustered three ways:
  single images (least conservative), source folder (same crawl query, collection date or category folder; main),
  and one cluster (most conservative).
Comparisons and accounting as paired_statistics.py and robustness.py (run answer set, rows from learned_cascade.py).
"""
import contextlib
import csv
import io
import os
import runpy
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "analysis"))
from event_split import event_of  # noqa: E402

MANIFEST = HERE.parents[0] / "heldout/results/medic_manifest_test.csv"  # MEDIC test-split manifest
B = 2000
sys.argv = [str(HERE / "learned_cascade.py"), "medic"]
with contextlib.redirect_stdout(io.StringIO()):
    L = runpy.run_path(str(HERE / "learned_cascade.py"))
rows, ids = L["rows"], L["ids"]
man = {Path(r["local_image_path"]).name: r["image_path"] for r in csv.DictReader(open(MANIFEST))}
paths = [man[i] for i in ids]  # KeyError here means the manifest does not match the held-out images
events = [event_of(p) for p in paths]
so, co, _ = rows["Ours"]
pick = {"Fixed@Pi": "Fixed@Pi (runs M5 first)", "Direct@Pi": "Direct@Pi", "Fixed@Jetson": "Fixed@Jetson (runs M5, Pi first)",
        "Direct@Jetson": "Direct@Jetson", "Confidence-Only": next(k for k in rows if k.startswith("Confidence-Only")),
        "Learned cascade (text+labels)": next(k for k in rows if k.startswith("Learned cascade text+labels, closest-cost")),
        "Learned cascade (labels only)": next(k for k in rows if k.startswith("Learned cascade labels only, closest-cost"))}

schemes = {
    "image level (r2_7, for comparison)": [f"i{n}" for n in range(len(ids))],
    "event; unknown as single images": [e or f"u{n}" for n, e in enumerate(events)],
    "event; unknown by source folder (main)": [e or os.path.dirname(p) for e, p in zip(events, paths)],
    "event; unknown as one cluster": [e or "unknown" for e in events],
}
known = Counter(e for e in events if e)
print(f"MEDIC held-out {len(ids)}: event known for {sum(known.values())} images in {len(known)} events "
      f"(largest {', '.join(f'{e} {n}' for e, n in known.most_common(5))}); {len(ids) - sum(known.values())} without event")
rng = np.random.default_rng(20261115)
f = lambda v: f"[{np.percentile(v, 2.5):+6.1f}, {np.percentile(v, 97.5):+6.1f}]"
for title, groups in schemes.items():
    names = sorted(set(groups))
    g = np.array([names.index(x) for x in groups])
    W = np.stack([np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names)) for _ in range(B)])
    S = lambda v: W @ np.bincount(g, weights=v.astype(float), minlength=len(names))  # resampled sums, one per draw
    n = S(np.ones(len(ids)))
    print(f"\n{title}: {len(names)} clusters, largest {np.bincount(g).max()} images; Ours minus baseline, 95% bootstrap")
    for name, key in pick.items():
        sb, cb, _ = rows[key]
        dq = 100 * (S(so) - S(sb)) / n
        rl, re = (100 * (S(co[:, c]) / S(cb[:, c]) - 1) for c in (0, 2))
        print(f"  vs {name:30s} Q-Sat {100 * (so.mean() - sb.mean()):+5.1f} pp {f(dq)} | latency "
              f"{100 * (co[:, 0].mean() / cb[:, 0].mean() - 1):+6.1f}% {f(rl)} | energy "
              f"{100 * (co[:, 2].mean() / cb[:, 2].mean() - 1):+6.1f}% {f(re)}")

# per-group consistency against the headline baseline
sb, cb, _ = rows[pick["Fixed@Pi"]]
groups = np.array(schemes["event; unknown by source folder (main)"])
print("\nOurs vs Fixed@Pi within groups of >= 25 images (source folders shortened)")
for name, k in Counter(groups).most_common():
    if k < 25:
        break
    m = groups == name
    print(f"  {name.split('/')[-2] if '/' in name else name:24s} {k:3d}: Q-Sat {100 * so[m].mean():5.1f} vs {100 * sb[m].mean():5.1f} "
          f"({100 * (so[m].mean() - sb[m].mean()):+5.1f} pp) | latency {100 * (co[m, 0].mean() / cb[m, 0].mean() - 1):+6.1f}% | "
          f"energy {100 * (co[m, 2].mean() / cb[m, 2].mean() - 1):+6.1f}%")
