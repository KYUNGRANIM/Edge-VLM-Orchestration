#!/usr/bin/env python3
"""Without the author's ratings: judge_sensitivity.py run unchanged on the nine non-author
raters (primary) and on all ten (sensitivity), plus the reply's numbers it does not print
(raw Q-Sat agreement, tie-aware Spearman).

    python3 analysis/judge_validation/judge_sensitivity_nonauthor.py
"""
import glob
import itertools
import runpy
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

AUTHOR = "rater00_author.json"  # rater 1: the author's own local export (2026-10-28)
HERE = Path(__file__).resolve().parent
_glob = glob.glob

for label, drop in (("9 non-author raters (primary)", {AUTHOR}), ("10 raters incl. author (sensitivity)", set())):
    glob.glob = lambda p, drop=drop, **k: [x for x in _glob(p, **k) if Path(x).name not in drop]
    print(f"\n################ {label}")
    g = runpy.run_path(str(HERE.parent / "judge_validation/judge_sensitivity.py"))
    glob.glob = _glob
    htxt, gpt, mtxt, pairs, sat = g["htxt"], g["gpt"], g["mtxt"], g["pairs"], g["sat60"]
    assert len(htxt) == 10 - len(drop) and all(AUTHOR not in p for p in g["paths"]) == bool(drop)
    rho = lambda a, b: spearmanr([a[p] for p in pairs], [b[p] for p in pairs])[0]  # noqa: E731
    agr = lambda a, b: float(np.mean(np.equal(sat(a), sat(b))))  # noqa: E731
    hh = list(itertools.combinations(htxt, 2))
    print(f"\nraw Q-Sat agreement: GPT vs each rater mean {np.mean([agr(h, gpt) for h in htxt]):.1%} | "
          f"rater vs rater mean {np.mean([agr(a, b) for a, b in hh]):.1%} ({len(hh)} pairs)")
    print(f"tie-aware Spearman (text): GPT vs each rater mean {np.mean([rho(h, gpt) for h in htxt]):.2f} | "
          f"rater vs rater mean {np.mean([rho(a, b) for a, b in hh]):.2f} | GPT vs median {rho(mtxt, gpt):.2f}")
