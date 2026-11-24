#!/usr/bin/env python3
"""Does the cost-matching target depend on auxiliary costs taken from the final (held-out) run?

    /opt/anaconda3/bin/python3 -B analysis/output_checks/cost_matching_sources.py

results_table.py builds the cost-matching target (Ours' out-of-fold composite cost on the profiling set) with
auxiliary-device costs averaged over the final November run: M5_B/C energy, Pi_B energy, Pi_B upload bytes. That
target selects Confidence-Only's thresholds and learned_cascade.py's cost-matched FrugalGPT-style thresholds. This
script recomputes only those selections with auxiliary costs from the profiling records where they exist (M5_B/C
time in profiling/results/raw_per_category/m5_*.jsonl; Pi_B upload = the same image Pi_A receives, from the profiling
main-path bytes) and sweeps each auxiliary component from 0x to 2x of the run value. The quality-constrained
FrugalGPT point does not use the target. Read-only; prints only.
"""
import contextlib
import glob
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "analysis"
sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    L = runpy.run_path(str(PIPE / "learned_cascade.py"))  # also runs results_table.py
G = runpy.run_path  # noqa: F841 (placeholder to keep the namespace explicit)
NORM, P_M5 = L["G"]["NORM"], 1.2154
pcost, pids, oof, prof = L["pcost"], L["pids"], L["G"]["oof"], L["prof"]
RUN_AUX = (L["AUX"][0], L["AUX"][1], L["AUX"][2])  # M5_B/C J, Pi_B J, Pi_B bytes (final run)


def composite(c, t, aux):
    lat, ovh, eng = c[t]["latency_ms"], c[t]["overhead_bytes"], c[t]["energy_j"]
    if aux is not None:
        eng += aux[0] + (aux[1] if t != "m5_a" else 0)
        ovh += aux[2] if t != "m5_a" else 0
    n = {k: min(max((v - NORM[k][0]) / NORM[k][1], 0.0), 1.0)
         for k, v in (("latency_ms", lat), ("overhead_bytes", ovh), ("energy_j", eng))}
    return 0.1 * n["latency_ms"] + 0.1 * n["overhead_bytes"] + 0.8 * n["energy_j"]


def target(aux):
    return float(np.mean([composite(pcost[i], t, aux) for i, t in zip(pids, oof)]))


# ---- Confidence-Only: thresholds whose no-auxiliary composite is closest to the target (results_table.py rule)
conf_cost = {(a, b): float(np.mean([composite(pcost[i], L["G"]["conf_pick"](prof[i]["m5_a"].get("conf") or 0,
                                                                          prof[i]["pi_a"].get("conf") or 0, a, b), None)
                                    for i in pids])) for a in range(1, 6) for b in range(1, 6)}


def conf_choice(tg):
    return min(conf_cost, key=lambda k: (abs(conf_cost[k] - tg), k[0], k[1]))


# ---- FrugalGPT-style cascade (text + labels): out-of-fold grid as learned_cascade.py, target-free
Scorer, pans, qp, cp, y, routes, GRID = (L[k] for k in ("Scorer", "pans", "qp", "cp", "y", "routes", "GRID"))
oofs = {t: np.zeros(len(pids)) for t in ("m5_a", "pi_a")}
for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y):
    for t in oofs:
        oofs[t][te] = Scorer(True).fit([pans[t][j] for j in tr], qp[t][tr].astype(int)).score([pans[t][j] for j in te])
grid = []
for a in GRID:
    for b in GRID:
        r = routes(oofs["m5_a"], oofs["pi_a"], a, b)
        grid.append((np.mean([qp[t][j] for j, t in enumerate(r)]), np.mean([cp[t][j] for j, t in enumerate(r)]), a, b))


def frugal(tg):
    budget = max((g for g in grid if g[1] <= tg), key=lambda g: (g[0], -g[1]))
    closest = min(grid, key=lambda g: (abs(g[1] - tg), -g[0]))
    return (budget[2], budget[3]), (closest[2], closest[3])


# ---- auxiliary costs from the profiling records
m5_aux_ms = []
for f in glob.glob(str(ROOT / "profiling/results/raw_per_category/m5_*.jsonl")):
    for line in open(f, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        a = r.get("aux_latency_ms") or {}
        if a.get("m5_b_total_ms") and a.get("m5_c_total_ms"):
            m5_aux_ms.append(a["m5_b_total_ms"] + a["m5_c_total_ms"])
prof_m5_j = P_M5 * float(np.mean(m5_aux_ms)) / 1000
prof_pib_bytes = float(np.mean([pcost[i]["pi_a"]["overhead_bytes"] - pcost[i]["m5_a"]["overhead_bytes"] for i in pids]))

print(f"final-run auxiliary costs: M5_B/C {RUN_AUX[0]:.1f} J, Pi_B {RUN_AUX[1]:.1f} J, Pi_B upload {RUN_AUX[2] / 1024:.0f} KiB")
print(f"profiling auxiliary costs: M5_B/C {prof_m5_j:.1f} J (n={len(m5_aux_ms)}), Pi_B energy not recorded, "
      f"Pi_B upload proxy {prof_pib_bytes / 1024:.0f} KiB (main-path upload of the same image)")
print(f"no auxiliary costs at all: target {target(None):.4f}")
rows = [("final run (as reported)", RUN_AUX), ("profiling M5_B/C + profiling upload, run Pi_B energy",
                                             (prof_m5_j, RUN_AUX[1], prof_pib_bytes))]
for s in (0.0, 0.5, 1.5, 2.0):
    rows.append((f"all three x{s}", tuple(v * s for v in RUN_AUX)))
for k, name in enumerate(("M5_B/C energy", "Pi_B energy", "Pi_B upload")):
    for s in (0.0, 2.0):
        aux = list(RUN_AUX)
        aux[k] *= s
        rows.append((f"{name} x{s}", tuple(aux)))
print(f"\n{'auxiliary costs':52s} {'target':>7s}  Conf-Only  FrugalGPT budget / closest-cost")
for name, aux in rows:
    tg = target(aux)
    fb, fc = frugal(tg)
    print(f"{name:52s} {tg:7.4f}  {conf_choice(tg)}     {fb} / {fc}")
assert conf_choice(target(RUN_AUX)) == (L["G"]["C1"], L["G"]["C2"]), "does not reproduce results_table.py"

# ---- held-out results of the FrugalGPT-style thresholds that appear above (run answer set, Table I accounting)
s_m5, s_pi = L["SCORES"][True]
ids, Qr, Q_TH = L["ids"], L["Qr"], L["Q_TH"]
ours = np.array(L["ours"])
so = np.array([Qr[i][t] >= Q_TH for i, t in zip(ids, ours)])
print("\nheld-out 600: thresholds -> Q-Sat %, latency s, KiB, J, M5/Pi/Jetson, Q-Sat vs Ours (McNemar p)")
for a, b in sorted({(0.75, 0.6), (0.8, 0.3), (0.7, 0.8), (0.75, 0.65), (0.75, 0.55)}):
    r = L["routes"](s_m5, s_pi, a, b)
    sat = np.array([Qr[i][t] >= Q_TH for i, t in zip(ids, r)])
    c = L["per_image"](list(r))
    n = [int(np.sum(r == t)) for t in ("m5_a", "pi_a", "jetson")]
    print(f"  ({a}, {b}): {100 * sat.mean():.1f}  {c[:, 0].mean():.1f}  {c[:, 1].mean():.0f}  {c[:, 2].mean():.1f}  "
          f"{n[0]}/{n[1]}/{n[2]}  p={L['mcnemar'](so, sat):.3f}")
