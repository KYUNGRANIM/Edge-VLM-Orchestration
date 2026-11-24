#!/usr/bin/env python3
"""Does Table I's latency for policies with auxiliary agents include waiting for those agents?

    python3 -B analysis/system_costs/critical_path.py

Table I (results_table.py) takes latency from the all-tier trace of the main agents only (M5_A; upload + Pi_A;
hop + Jetson) and adds auxiliary energy and traffic, not time. At runtime, Stage 1 waits for M5_B and M5_C and Stage 2
waits for Pi_B. Here the same all-tier trace supplies the auxiliary times, and the critical path is rebuilt as
  Stage 1 ready = max(M5_A, M5_B, M5_C);  Stage 2 ready = Stage 1 + max(upload_A + Pi_A, upload_B + Pi_B),
assuming the agents of a tier start together and the two uploads run concurrently (B/C result messages ignored).
The final run's own stage2_wall_ms (Pi_A start until Pi_B's answer and the Stage 2 decision) is shown as a check.
"""
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M5VLM = HERE.parents[1]
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = [str(HERE.parent / "main_results/paired_statistics.py"), "medic"]
    R = runpy.run_path(str(HERE.parent / "main_results/paired_statistics.py"))
    sys.argv = [sys.argv[0]]
    X = runpy.run_path(str(HERE.parent / "main_results/paper_numbers.py"))
L, rows, ids, IX = R["L"], R["rows"], list(R["ids"]), R["IX"]
cost = L["cost"]
A = M5VLM / "heldout/results/all_tier_reference/online_result_m5.jsonl"
PA = M5VLM / "revision/first_submission/run_traces/online_pi.jsonl"
PB = M5VLM / "revision/first_submission/run_traces/pi_b_online_routing_v2.jsonl"
m5 = {r["image_id"]: r for r in map(json.loads, open(A))}
pa = {r["image_id"]: r for r in map(json.loads, open(PA))}
pb = {r["image_id"]: r for r in map(json.loads, open(PB))}
a = np.array([m5[i]["end_m5"]["cost"]["latency_ms"]["m5_total_ms"] for i in ids], float)
b = np.array([m5[i]["aux_latency_ms"]["m5_b_total_ms"] for i in ids], float)
c = np.array([m5[i]["aux_latency_ms"]["m5_c_total_ms"] for i in ids], float)
upa = np.array([pa[i]["pi_a"]["costs"]["m5_pi_http_e2e_ms"] for i in ids], float)
pia = np.array([pa[i]["pi_a"]["costs"]["pi_total_ms"] for i in ids], float)
upb = np.array([pb[i]["end_pi"]["cost"]["latency_ms"]["m5_pi_b_http_e2e_ms"] for i in ids], float)
pib = np.array([pb[i]["end_pi"]["cost"]["latency_ms"]["pi_b_total_ms"] for i in ids], float)
# the trace Table I uses must be these same main-agent numbers
assert np.allclose(a, [cost[i]["m5_a"]["latency_ms"] for i in ids])
assert np.allclose(a + upa + pia, [cost[i]["pi_a"]["latency_ms"] for i in ids])
d1 = np.maximum.reduce([a, b, c]) - a                              # Stage 1 wait for M5_B / M5_C, ms
d2 = np.maximum(upa + pia, upb + pib) - (upa + pia)                # Stage 2 wait for Pi_B, ms
print(f"Stage 1: M5_B or M5_C finishes after M5_A on {np.mean(d1 > 0):.1%} of 600 images; extra wait mean {d1.mean() / 1000:.2f} s "
      f"(median when > 0 {np.median(d1[d1 > 0]) / 1000:.2f} s, max {d1.max() / 1000:.1f} s)")
print(f"Stage 2: Pi_B ready after Pi_A on {np.mean(d2 > 0):.1%}; extra wait mean {d2.mean() / 1000:.2f} s "
      f"(median when > 0 {np.median(d2[d2 > 0]) / 1000:.2f} s, max {d2.max() / 1000:.1f} s); "
      f"Pi_A {pia.mean() / 1000:.1f} s vs Pi_B {pib.mean() / 1000:.1f} s")


def crit(tiers):
    t = np.asarray(tiers)
    return np.array([cost[i][x]["latency_ms"] for i, x in zip(ids, t)], float) + d1 + np.where(t != "m5_a", d2, 0)


so, co, ours = rows["Ours"][0], rows["Ours"][1], list(rows["Ours"][2])
casc = [x if x != "jetson" else "pi_a" for x in ours]
new = {"Ours": crit(ours) / 1000, "M5-Pi Cascade": crit(casc) / 1000}
old = {"Ours": co[:, 0], "M5-Pi Cascade": L["per_image"](casc, True)[:, 0]}
for k in new:
    print(f"{k}: Table I {old[k].mean():.1f} s -> critical path {new[k].mean():.1f} s (+{new[k].mean() - old[k].mean():.1f} s)")
x = new["Ours"]
print(f"Ours latency <= 15 s {np.mean(x <= 15):.1%} (Table I {np.mean(co[:, 0] <= 15):.1%}), <= 90 s {np.mean(x <= 90):.1%} "
      f"(Table I {np.mean(co[:, 0] <= 90):.1%}); M5-terminated mean {x[np.asarray(ours) == 'm5_a'].mean():.1f} s")
base = {"Fixed@Pi": rows["Fixed@Pi (runs M5 first)"][1][:, 0], "Direct@Pi": rows["Direct@Pi"][1][:, 0],
        "Fixed@Jetson": rows["Fixed@Jetson (runs M5, Pi first)"][1][:, 0],
        "Confidence-Only": rows[next(k for k in rows if k.startswith("Confidence-Only"))][1][:, 0],
        "Primary-Only": L["per_image"](list(X["tiers"]), False)[:, 0]}
print("Ours latency relative to baseline, Table I -> critical path [paired bootstrap 95%, r2_7 draws]")
for k, y in base.items():
    out = []
    for o in (co[:, 0], x):
        v = [100 * (o[ix].mean() / y[ix].mean() - 1) for ix in IX]
        out.append(f"{100 * (o.mean() / y.mean() - 1):+6.1f}% [{np.percentile(v, 2.5):+6.1f}, {np.percentile(v, 97.5):+6.1f}]")
    print(f"  vs {k:16s} {out[0]}  ->  {out[1]}")
# the final run's measured Stage 2 wait (Pi_A start until Pi_B's answer and the decision)
run = {r["image_id"]: r for r in map(json.loads, open(M5VLM / "deployment/live_run/online_routing_final.jsonl"))}
w = np.array([[run[i]["end_pi"]["cost"]["latency_ms"][k] for k in ("stage2_wall_ms", "pi_a_total_ms", "pi_b_total_ms")]
              for i in ids if run[i].get("end_pi")], float)
print(f"final run, {len(w)} escalated images: stage2_wall - Pi_A > 0.5 s on {np.mean(w[:, 0] - w[:, 1] > 500):.1%}, "
      f"mean {np.mean(w[:, 0] - w[:, 1]) / 1000:.2f} s; Pi_B > Pi_A on {np.mean(w[:, 2] > w[:, 1]):.1%}")
