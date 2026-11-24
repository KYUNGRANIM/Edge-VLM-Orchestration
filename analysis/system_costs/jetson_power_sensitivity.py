#!/usr/bin/env python3
"""How the energy comparisons move if the Jetson's power is underestimated (MEDIC rerun or EIDSeg).

    python3 analysis/system_costs/jetson_power_sensitivity.py [medic|eidseg]

The M5 and the Pi were metered at their power input (AVHzY CT-3); the Jetson figure is its on-module VDD_IN rail,
which leaves out the carrier board. Every Jetson stage's energy is scaled to a higher Jetson power (+2 W, +4 W,
x1.5, x2) and the Table I energy of each method (run answer set, rows of learned_cascade.py) is recomputed.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "medic"
sys.argv = [str(HERE.parent / "main_results/learned_cascade.py"), DATASET]
with contextlib.redirect_stdout(io.StringIO()):
    L = runpy.run_path(str(HERE.parent / "main_results/learned_cascade.py"))
rows, ids, cost, alltier = L["rows"], L["ids"], L["cost"], L["alltier"]
P_JET = np.mean([alltier[i]["stage_power_w"]["jetson"] for i in ids])
jet_e = np.array([cost[i]["jetson"]["energy_j"] - cost[i]["pi_a"]["energy_j"] for i in ids])  # Jetson stage energy
pick = {"Fixed@Pi": "Fixed@Pi (runs M5 first)", "Direct@Pi": "Direct@Pi", "Fixed@Jetson": "Fixed@Jetson (runs M5, Pi first)",
        "Direct@Jetson": "Direct@Jetson", "Confidence-Only": next(k for k in rows if k.startswith("Confidence-Only")),
        "Learned cascade": next(k for k in rows if k.startswith("Learned cascade text+labels, closest-cost"))}


def split(key):
    """mean energy without the Jetson stage, mean Jetson-stage energy."""
    tiers, e = np.asarray(rows[key][2]), rows[key][1][:, 2]
    j = jet_e if key == "Direct@Jetson" else np.where(tiers == "jetson", jet_e, 0.0)
    return (e - j).mean(), j.mean()


SCEN = {"as measured": 1.0, "+2 W": (P_JET + 2) / P_JET, "+4 W": (P_JET + 4) / P_JET, "x1.5": 1.5, "x2": 2.0}
oa, oj = split("Ours")
print(f"{DATASET.upper()} held-out {len(ids)}: Jetson power {P_JET:.2f} W; Ours energy {oa + oj:.0f} J "
      f"(Jetson stage {oj:.0f} J, {100 * oj / (oa + oj):.0f}%)")
print("Ours minus baseline, relative change of mean energy (negative = Ours uses less)")
print("  " + " " * 18 + "".join(f"{s:>14s}" for s in SCEN))
for name, key in pick.items():
    ba, bj = split(key)
    cells = "".join(f"{100 * ((oa + k * oj) / (ba + k * bj) - 1):+13.1f}%" for k in SCEN.values())
    even = (oa - ba) / (bj - oj) if bj != oj else np.inf  # Jetson factor at which the two energies are equal
    note = f"  equal at Jetson x{even:.2f} ({even * P_JET:.1f} W)" if 0 < even < 10 else ""
    print(f"  vs {name:16s}{cells}{note}   [{ba + bj:.0f} J, Jetson stage {100 * bj / (ba + bj):.0f}%]")
