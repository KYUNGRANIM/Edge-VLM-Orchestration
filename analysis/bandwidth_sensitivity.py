#!/usr/bin/env python3
"""Does the extra traffic cost Ours its latency advantage when bandwidth is scarce? (estimate, not measured)

    python3 analysis/bandwidth_sensitivity.py [medic|eidseg]

1. Mean and p95 latency per held-out image when a link is slowed, with the per-image model of
   wireless_sensitivity.py (processing as measured; a slowed hop takes bytes x 8 / bandwidth; Ours' Pi_B copy shares
   the camera link), and the bandwidth below which Ours loses to Fixed@Pi / Direct@Pi.
2. Link load at server saturation: the load test throughput (images/h, one request stream per server) times
   the bytes each method puts on a link; below that bandwidth the link, not the servers, limits throughput.
Timeouts and retries are not modelled, so completion rate under a slow link is not estimated.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "medic"
sys.argv = [str(HERE / "wireless_sensitivity.py"), DATASET]
with contextlib.redirect_stdout(io.StringIO()):
    W = runpy.run_path(str(HERE / "wireless_sensitivity.py"))
m5, pi, jet, local_b, back_b, local_ms, back_ms, cam_rate, AUX = (W[k] for k in (
    "m5", "pi", "jet", "local_b", "back_b", "local_ms", "back_ms", "cam_rate", "AUX"))
tiers = {k: np.asarray(W["methods"][k]) for k in ("Ours", "Fixed@Pi", "Direct@Pi", "Direct@Jetson")}


def per_image(name, b_local=None, b_back=None):
    """latency s per image, same model as wireless_sensitivity.latency() but before the mean."""
    t = tiers[name]
    extra = AUX[2] if name == "Ours" else 0.0
    up = local_ms if b_local is None else (local_b + np.where(t != "m5_a", extra, 0)) * 8 / b_local * 1000
    hop = back_ms if b_back is None else back_b * 8 / b_back * 1000
    if name == "Direct@Pi":
        return (up + pi) / 1000
    if name == "Direct@Jetson":
        rate = np.minimum(cam_rate if b_local is None else b_local, np.inf if b_back is None else b_back)
        return (local_b * 8 / rate * 1000 + jet) / 1000
    return (m5 + np.where(t != "m5_a", up + pi, 0) + np.where(t == "jetson", hop + jet, 0)) / 1000


print(f"{DATASET.upper()} held-out {len(m5)} (estimate): mean / p95 latency s")
for title, kw in (("camera link slowed", "b_local"), ("Pi-Jetson link slowed", "b_back"), ("both links slowed", "both")):
    print(f"\n  {title}")
    print("  Mbit/s   " + "".join(f"{m:>18s}" for m in tiers))
    for bw in (None, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
        args = {} if bw is None else ({"b_local": bw * 1e6, "b_back": bw * 1e6} if kw == "both" else {kw: bw * 1e6})
        cells = "".join(f"{np.mean(v):9.1f} /{np.percentile(v, 95):6.1f}" for v in (per_image(m, **args) for m in tiers))
        print(f"  {'measured' if bw is None else f'{bw:g}':8s}" + cells)
grid = np.geomspace(5e6, 2e3, 600)
for base in ("Fixed@Pi", "Direct@Pi"):
    for stat, f in (("mean", np.mean), ("p95", lambda v: np.percentile(v, 95))):
        lose = next((b for b in grid if f(per_image("Ours", b_local=b, b_back=b)) > f(per_image(base, b_local=b, b_back=b))), None)
        print(f"  Ours' {stat:4s} latency exceeds {base}'s when both links fall below "
              f"{'never (down to 2 kbit/s)' if lose is None else f'{lose / 1e3:.0f} kbit/s'}")

# ---------------- 2. link load at server saturation (load test, k = 4)
IMG_H = {"Ours": 137, "Fixed@Pi": 73, "Direct@Pi": 73, "Direct@Jetson": 350}
print("\nlink load when the servers run flat out (load-test throughput x bytes per image; kbit/s)")
for name, t in tiers.items():
    cam = np.mean(local_b + (np.where(t != "m5_a", AUX[2], 0) if name == "Ours" else 0)) if name != "Ours" else \
        np.mean(np.where(t != "m5_a", local_b + AUX[2], 0))
    if name in ("Fixed@Pi", "Direct@Pi", "Direct@Jetson"):
        cam = np.mean(local_b)
    back = np.mean(np.where(t == "jetson", back_b, 0)) if name == "Ours" else (np.mean(local_b) if name == "Direct@Jetson" else 0.0)
    rate = IMG_H[name] / 3600
    print(f"  {name:14s} {IMG_H[name]:4d} img/h: camera link {cam / 1024:4.0f} KiB/img -> {cam * 8 * rate / 1e3:6.1f} kbit/s"
          + (f" | Pi-Jetson {back / 1024:4.0f} KiB/img -> {back * 8 * rate / 1e3:6.1f} kbit/s" if name != "Direct@Jetson" else
             " | (Direct@Jetson's upload is the same bytes, over whichever link reaches the Jetson)"))
print(f"  measured camera link {np.median(cam_rate) / 1e6:.2f} Mbit/s (median)")
