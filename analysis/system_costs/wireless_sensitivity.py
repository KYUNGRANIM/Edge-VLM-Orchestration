#!/usr/bin/env python3
"""Wireless conditions, estimated (not measured): mean latency when a link's bandwidth drops.

    python3 analysis/system_costs/wireless_sensitivity.py [medic|eidseg]

Per held-out image the all-tier trace gives each device's processing time and each hop's bytes and transfer time.
Processing stays as measured; a degraded hop takes bytes x 8 / bandwidth instead of its measured time.
  local link  camera (M5) -> Pi; Ours also sends the image to Pi_B, counted on the same link when it is degraded
  backhaul    Pi -> Jetson, and the camera -> Jetson upload of Direct@Jetson
Direct@Jetson's upload runs at the slower of the camera's measured rate and the backhaul (the trace only has the fast
Pi -> Jetson hop, so its 11.7 s leaves out the camera's own upload). Energy is not modelled: the cost model charges
processing time only, so radio time on a slow link is missing for every method.
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
rows, ids, alltier, AUX = L["rows"], L["ids"], L["alltier"], L["AUX"]
sl = [alltier[i]["stage_latency_ms"] for i in ids]
sb = [alltier[i]["stage_overhead_bytes"] for i in ids]
local_b = np.array([b["m5_pi_tx"] + b["m5_pi_rx"] for b in sb], float)
back_b = np.array([b["pi_jet_tx"] + b["pi_jet_rx"] for b in sb], float)
local_ms = np.array([s["m5_pi_http"] for s in sl], float)
back_ms = np.array([s["pi_jet_http"] for s in sl], float)
m5, pi, jet = (np.array([s[k] for s in sl], float) for k in ("m5_total", "pi_total", "jetson_total"))
cam_rate = local_b * 8 / np.maximum(local_ms, 1) * 1000  # bit/s the camera reached on its own Wi-Fi
print(f"{DATASET.upper()} held-out {len(ids)}: measured local link {np.median(cam_rate) / 1e6:.2f} Mbit/s (median), "
      f"backhaul {np.median(back_b * 8 / np.maximum(back_ms, 1) * 1000) / 1e6:.1f} Mbit/s; image upload "
      f"{local_b.mean() / 1024:.0f} KiB, Pi_B copy {AUX[2] / 1024:.0f} KiB")


def latency(name, tiers, b_local=None, b_back=None):
    """mean latency s of a method; None keeps a link as measured."""
    t = np.asarray(tiers)
    extra = AUX[2] if name == "Ours" else 0.0
    up = local_ms if b_local is None else (local_b + np.where(t != "m5_a", extra, 0)) * 8 / b_local * 1000
    hop = back_ms if b_back is None else back_b * 8 / b_back * 1000
    if name == "Direct@Pi":
        return np.mean(up + pi) / 1000
    if name == "Direct@Jetson":
        rate = np.minimum(cam_rate if b_local is None else b_local, np.inf if b_back is None else b_back)
        return np.mean(local_b * 8 / rate * 1000 + jet) / 1000
    lat = m5 + np.where(t != "m5_a", up + pi, 0) + np.where(t == "jetson", hop + jet, 0)
    return lat.mean() / 1000


methods = {"Ours": rows["Ours"][2], "Fixed@Pi": rows["Fixed@Pi (runs M5 first)"][2], "Direct@Pi": rows["Direct@Pi"][2],
           "Direct@Jetson": rows["Direct@Jetson"][2], "Fixed@Jetson": rows["Fixed@Jetson (runs M5, Pi first)"][2],
           "Confidence-Only": rows[next(k for k in rows if k.startswith("Confidence-Only"))][2],
           "Learned cascade": rows[next(k for k in rows if k.startswith("Learned cascade text+labels, closest-cost"))][2]}
share = {m: np.mean(np.asarray(t) == "jetson") if m != "Direct@Jetson" else 1.0 for m, t in methods.items()}
print("backhaul traffic per image (KiB): " + ", ".join(
    f"{m} {np.mean(np.where(np.asarray(t) == 'jetson', back_b, 0)) / 1024 if m != 'Direct@Jetson' else local_b.mean() / 1024:.0f}"
    f" ({100 * share[m]:.0f}% of images)" for m, t in methods.items() if m not in ("Fixed@Pi", "Direct@Pi")))
for title, grid, kw in (("backhaul degraded (local link as measured)", (None, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05), "b_back"),
                        ("local link degraded (backhaul as measured)", (None, 1.0, 0.5, 0.2, 0.1), "b_local"),
                        ("both links at the same bandwidth", (None, 1.0, 0.5, 0.2, 0.1), "both")):
    print(f"\nmean latency s, {title}")
    print("  Mbit/s    " + "".join(f"{m:>16s}" for m in methods))
    for bw in grid:
        args = {} if bw is None else ({"b_local": bw * 1e6, "b_back": bw * 1e6} if kw == "both" else {kw: bw * 1e6})
        cells = "".join(f"{latency(m, t, **args):16.1f}" for m, t in methods.items())
        print(f"  {'measured' if bw is None else f'{bw:g}':9s}" + cells)
# where Ours' mean latency meets Direct@Jetson's when only the backhaul degrades
grid = np.geomspace(20e6, 5e3, 400)
gap = [latency("Ours", methods["Ours"], b_back=b) - latency("Direct@Jetson", methods["Direct@Jetson"], b_back=b) for b in grid]
cross = next((b for b, g in zip(grid, gap) if g < 0), None)
print(f"\nOurs' mean latency drops below Direct@Jetson's once the backhaul is below "
      f"{'never (down to 5 kbit/s)' if cross is None else f'{cross / 1e3:.0f} kbit/s'}")
