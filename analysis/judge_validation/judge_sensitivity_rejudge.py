#!/usr/bin/env python3
"""The 180 human-rated answers re-scored with the final evaluation's snapshot
(gpt-4o-2024-08-06, same prompt and candidate order; revision/human_evaluation/rescored_gpt4o/), against the nine
non-author raters. judge_sensitivity.py is run unchanged with the author's file left out, once on the earlier
GPT scores and once on the re-scored ones, so both rows use the same statistics code and bootstrap draws.

    python3 analysis/judge_validation/judge_sensitivity_rejudge.py
"""
import contextlib
import glob
import io
import itertools
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
M5VLM = HERE.parents[1]
NEW = M5VLM / "revision/human_evaluation/rescored_gpt4o/judge_he60_gpt-4o-2024-08-06.jsonl"
AUTHOR = "rater00_author.json"  # rater 1: the author's own export
sys.path.insert(0, str(M5VLM / "revision/human_evaluation"))
import analyze_human_eval as H  # noqa: E402

OLD = H.JUDGE
_glob = glob.glob
glob.glob = lambda p, **k: [x for x in _glob(p, **k) if Path(x).name != AUTHOR]

new_rows = [json.loads(l) for l in open(NEW)]
assert len(new_rows) == 60 and all(r["status"] == "ok" for r in new_rows), "re-scoring incomplete"
print("snapshots served:", sorted({(r["usage"]["resolved_model"], r["usage"]["system_fingerprint"]) for r in new_rows}))

runs = {}
for label, path in (("earlier GPT scores", OLD), ("re-scored, gpt-4o-2024-08-06", NEW)):
    H.JUDGE = path
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runs[label] = g = runpy.run_path(str(HERE.parent / "judge_validation/judge_sensitivity.py"))
    H.JUDGE = OLD
    out = buf.getvalue()
    print(f"\n################ {label}\n" + out[:out.index("=== all rater subsets")])
    htxt, gpt, mtxt, pairs, sat = g["htxt"], g["gpt"], g["mtxt"], g["pairs"], g["sat60"]
    assert len(htxt) == 9
    rho = lambda a, b: spearmanr([a[p] for p in pairs], [b[p] for p in pairs])[0]  # noqa: E731
    agr = lambda a, b: float(np.mean(np.equal(sat(a), sat(b))))  # noqa: E731
    hh = list(itertools.combinations(htxt, 2))
    print(f"raw Q-Sat agreement: GPT vs each rater mean {np.mean([agr(h, gpt) for h in htxt]):.1%} | "
          f"rater vs rater {np.mean([agr(a, b) for a, b in hh]):.1%} | GPT vs median {agr(mtxt, gpt):.1%}")
    print(f"tie-aware Spearman (text): GPT vs each rater mean {np.mean([rho(h, gpt) for h in htxt]):.2f} | "
          f"rater vs rater {np.mean([rho(a, b) for a, b in hh]):.2f} | GPT vs median {rho(mtxt, gpt):.2f}")

# the snapshot effect itself: earlier vs re-scored GPT on the same 180 answers
a, b = (runs[k]["gpt"] for k in runs)
g = runs["re-scored, gpt-4o-2024-08-06"]
pairs, sat, T = g["pairs"], g["sat60"], g["T"]
print("\n=== earlier vs re-scored GPT, same 180 answers")
print(f"Q-Sat agreement {np.mean(np.equal(sat(a), sat(b))):.1%}, kappa {H.kappa(sat(a), sat(b), k=2):.2f}, "
      f"text Spearman {spearmanr([a[p] for p in pairs], [b[p] for p in pairs])[0]:.2f}, "
      f"mean |diff| {np.mean([abs(a[p] - b[p]) for p in pairs]):.3f}")
for t in T:
    print(f"  {t:6s} mean text earlier {np.mean([a[i, t] for i in g['hid']]):.3f} -> re-scored {np.mean([b[i, t] for i in g['hid']]):.3f}")

# policy-level Q-Sat on the 60 images (rerun routes, the answers the raters saw), by scorer
hid, q60, route, htxt, mtxt = g["hid"], g["q60"], g["route"], g["htxt"], g["mtxt"]
qsat = lambda txt, r: 100 * np.mean([q60(i, r[i], txt[i, r[i]]) >= H.Q_TH for i in hid])  # noqa: E731
cols = {"GPT earlier": a, "GPT re-scored": b, "median of 9": mtxt}
print("\n=== Q-Sat (%) on the 60 human-rated images by scorer (rerun routes, rated answers)")
print(" " * 16 + "".join(f"{c:>15s}" for c in [*cols, "rater mean"]))
table = {}
for name, r in route.items():
    table[name] = [qsat(txt, r) for txt in cols.values()] + [np.mean([qsat(h, r) for h in htxt])]
    print(f"{name:16s}" + "".join(f"{x:15.1f}" for x in table[name]))
for j, c in enumerate([*cols, "rater mean"]):
    order = sorted(table, key=lambda n: -table[n][j])
    print(f"order under {c:14s}: " + " > ".join(f"{n} {table[n][j]:.1f}" for n in order))
