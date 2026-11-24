#!/usr/bin/env python3
"""When does selective escalation beat sending every image to a shared Jetson? A queueing projection.

    python3 -B analysis/system_costs/shared_jetson.py

Setting: many camera clusters, each with its own M5 trio and Pi pair, share one Jetson. Images arrive as a Poisson
stream at aggregate rate lam; the Jetson serves one request at a time (FIFO), as the load test found (service
flat in the concurrency level). Per-image costs come from the measured traces of the 600 held-out images:
Direct@Jetson = camera upload + Jetson service (Table I's 13.4 s); Ours = its trace path (39.5 s) plus, for the 23.3%
it sends to the Jetson, the Jetson queueing delay. Mean wait from Pollaczek-Khinchine (M/G/1).
Assumes clusters are added so that no M5 or Pi queues (one cluster handles about 137 images/h in the load test) and
that Wi-Fi is not saturated; it is a projection from measured service times, not a measurement.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = [str(HERE.parent / "system_costs/wireless_sensitivity.py"), "medic"]
    W = runpy.run_path(str(HERE.parent / "system_costs/wireless_sensitivity.py"))
ours = np.asarray(W["rows"]["Ours"][2])
ours_lat = W["rows"]["Ours"][1][:, 0]
U, S = W["local_ms"] / 1000, W["jet"] / 1000          # camera upload and Jetson service per image, s
SJ = S[ours == "jetson"]                                 # Jetson service of the images Ours escalates
p = np.mean(ours == "jetson")
CLUSTER_IMG_H = 137                                      # load test: Ours, images/h per cluster at saturation


def wait(lam_s, s):  # M/G/1 mean queueing delay, lam_s in requests per second
    rho = lam_s * s.mean()
    return np.inf if rho >= 1 else lam_s * np.mean(s ** 2) / (2 * (1 - rho))


direct = lambda lam: (U + S).mean() + wait(lam / 3600, S)            # noqa: E731
sel = lambda lam: ours_lat.mean() + p * wait(p * lam / 3600, SJ)     # noqa: E731
cap_d, cap_o = 3600 / S.mean(), 3600 / (p * SJ.mean())
print(f"trace: Direct@Jetson {direct(1e-9):.1f} s, Ours {sel(1e-9):.1f} s at low load; Jetson service {S.mean():.2f} s "
      f"(SCV {S.var() / S.mean() ** 2:.2f}); Ours sends {100 * p:.1f}% to the Jetson")
print(f"Jetson saturates at {cap_d:.0f} images/h with Direct@Jetson and at {cap_o:.0f} images/h with Ours "
      f"({cap_o / cap_d:.1f}x; needs about {cap_o / CLUSTER_IMG_H:.0f} clusters)")
lam_x = brentq(lambda lam: direct(lam) - sel(lam), 1, cap_d - 1e-6)
print(f"mean latencies cross at {lam_x:.0f} images/h (Jetson utilisation {lam_x / cap_d:.0%} under Direct@Jetson), "
      f"{direct(lam_x):.1f} s each; about {np.ceil(lam_x / CLUSTER_IMG_H):.0f} clusters")
print("aggregate images/h:  Direct@Jetson mean s | Ours mean s")
for lam in (50, 100, 150, 200, 250, lam_x, 280, 290, 300, 400, 800, 1200):
    d, o = direct(lam), sel(lam)
    print(f"  {lam:7.0f}  {'saturated' if np.isinf(d) else f'{d:6.1f}':>10s} | {'saturated' if np.isinf(o) else f'{o:6.1f}':>9s}")
