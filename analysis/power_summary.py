#!/usr/bin/env python3
"""Mean active power of each device and its run-to-run repeatability, from the raw meter logs.

    python3 analysis/power_summary.py

Logs (profiling/results/power/, 1 sample per second):
  M5Stack CoreS3 + Module LLM  m5.csv                        AVHzY CT-3 USB meter at the power input
  Raspberry Pi 5               pi_run{1,2}.csv               same meter
  Jetson Orin Nano (15 W mode) jetson_run{1,2}.log           tegrastats VDD_IN (module rail only)
Mean power is the mean of all samples. The USB meter also accumulates energy from its own start (t = 0), so its final
Energy(Wh) is compared with mean power x the last timestamp. The resulting means are the per-device powers used in
the cost model (results_table.py P_M5 / P_PI; the Jetson value is stage_power_w in the collection records).
"""
import csv
import re
from pathlib import Path
from statistics import mean

POWER = Path(__file__).resolve().parents[1] / "profiling/results/power"


def usb(name):
    rows = list(csv.DictReader(open(POWER / name)))
    t, p = [float(r["Time(s)"]) for r in rows], [float(r["PBUS(W)"]) for r in rows]
    return {"name": name, "p": p, "minutes": (t[-1] - t[0]) / 60, "meter_wh": float(rows[-1]["Energy(Wh)"]),
            "recon_wh": mean(p) * t[-1] / 3600}


def tegrastats(name):
    p = [int(m.group(1)) / 1000 for m in map(re.compile(r"VDD_IN (\d+)mW/").search, open(POWER / name)) if m]
    return {"name": name, "p": p, "minutes": len(p) / 60}  # one line per second


devices = {"M5Stack": [usb("m5.csv")],
           "Pi": [usb("pi_run1.csv"), usb("pi_run2.csv")],
           "Jetson": [tegrastats("jetson_run1.log"), tegrastats("jetson_run2.log")]}
pooled = {}
for dev, runs in devices.items():
    for r in runs:
        extra = (f" | meter {r['meter_wh']:.5f} Wh vs mean x time {r['recon_wh']:.5f} Wh "
                 f"({100 * (r['recon_wh'] / r['meter_wh'] - 1):+.3f}%)" if "meter_wh" in r else "")
        print(f"{dev:8s} {r['name']:28s} {len(r['p']):5d} samples {r['minutes']:6.2f} min  mean {mean(r['p']):.4f} W{extra}")
    pooled[dev] = mean([x for r in runs for x in r["p"]])
    line = f"{dev:8s} pooled {pooled[dev]:.4f} W over {sum(r['minutes'] for r in runs):.2f} min"
    if len(runs) == 2:
        a, b = (mean(r["p"]) for r in runs)
        line += f"; the two recordings differ by {100 * abs(a - b) / a:.2f}%"
    print(line + "\n")

# the cost model's constants (rounded) come from these logs
assert round(pooled["M5Stack"], 4) == 1.2154 and round(pooled["Pi"], 4) == 7.5564 and round(pooled["Jetson"], 3) == 11.199
print("matches the cost-model powers: M5 1.2154 W, Pi 7.5564 W, Jetson 11.199 W")
