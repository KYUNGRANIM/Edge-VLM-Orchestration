#!/usr/bin/env python3
"""Summarize the load test (3 repetitions x Ours / Direct@Pi / Direct@Jetson x k = 1, 2, 4).

    python3 revision/load_test/code/analyze_load_test.py

Response time is the extra time after the M5 response is ready:
  Pi path (Ours, Direct@Pi): Pi_A final ts - Pi_A recv_ts; wait = work_start_ts - recv_ts (both on Pi_A's clock).
  Direct@Jetson: the client's submit -> reply (includes the image upload); Jetson service = ts - start_ts.
END@M5 images in Ours send nothing and count as 0 s in the "all 60 images" columns.
Each (rep, method, k) uses the run id whose 60 rows all returned HTTP 200; rep 1 was finished in pieces, and
the two broken tries (one lost Jetson reply, the stall when the Mac slept) are left out by that rule.
"""
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
RUN_FINAL = Path(__file__).resolve().parents[3] / "deployment/live_run/online_routing_final.jsonl"
METHODS, KS = ("ours", "direct_pi", "direct_jetson"), (1, 2, 4)
NAME = {"ours": "Ours", "direct_pi": "Direct@Pi", "direct_jetson": "Direct@Jetson"}


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def t(s):
    return datetime.fromisoformat(s).timestamp()


def ok(row):
    h = row.get("http")
    return h is None or set([h] if isinstance(h, str) else h) == {"200"}


def p95(x):
    return float(np.percentile(x, 95))


log = [r for r in jl(R / "load_test_log.jsonl") if "rid" in r]
fin = {r["image_id"]: r for r in jl(R / "pi_a/online_routing_final.jsonl")}
jet = {r["image_id"]: r for r in jl(R / "jetson/online_jetson_results.jsonl")}
rerun_route = {r["image_id"]: r["route_label"] for r in jl(RUN_FINAL)}

# ---- pick one complete run id per condition
groups = defaultdict(list)
for r in log:
    groups[(r["rep"], r["method"], r["k"], r["rid"].split("_rep")[0])].append(r)
cond = {}
for (rep, m, k, rid), rows in groups.items():
    if len(rows) == 60 and all(ok(r) for r in rows):
        assert (rep, m, k) not in cond, f"two complete tries for {(rep, m, k)}"
        cond[(rep, m, k)] = (rid, rows)
assert len(cond) == 27, sorted(cond)
print("run ids used:", dict(Counter(f"rep{rep} {rid}" for (rep, _, _), (rid, _) in sorted(cond.items()))))

# ---- per-request measurements
per = defaultdict(list)  # (method, k) -> request dicts
span = defaultdict(list)  # (method, k) -> per-rep wall time of the shared-path work, s
fails = Counter()
routes = defaultdict(set)  # original image -> Ours final tier in each condition
for (rep, m, k), (rid, rows) in sorted(cond.items()):
    times = []
    for r in rows:
        if r["kind"] == "m5_end":
            per[(m, k)].append({"rep": rep, "resp": 0.0, "m5_end": True})
            routes[r["orig"]].add("m5_a")
            continue
        if r["kind"] == "jetson":
            j = jet[r["rid"]]
            fails["jetson not ok"] += j["status"] != "ok"
            svc = t(j["ts"]) - t(j["start_ts"])
            per[(m, k)].append({"rep": rep, "resp": r["seconds"], "svc": svc, "wait": r["seconds"] - svc})
            times += [t(r["submit"]), t(r["done"])]
            continue
        f = fin[r["rid"]]
        fails["Pi_A final not ok"] += f["status"] != "ok"
        q = {"rep": rep, "resp": t(f["ts"]) - t(f["recv_ts"]), "wait": t(f["work_start_ts"]) - t(f["recv_ts"]),
             "svc": t(f["ts"]) - t(f["work_start_ts"]), "tier": f["final_tier"]}
        if m == "ours":
            lat = f["end_pi"]["cost"]["latency_ms"]
            q["pi_a_vlm"] = lat["pi_a_vlm_ms"] / 1000
            q["jet_call"] = (f.get("go_jetson") or {}).get("pi_jet_http_e2e_ms", 0) / 1000
            routes[r["orig"]].add(f["final_tier"])
            routes[r["orig"]].add("rerun:" + rerun_route[r["orig"]])
        else:
            q["pi_a_vlm"] = f["pi_a_vlm_ms"] / 1000
        per[(m, k)].append(q)
        times += [t(f["recv_ts"]), t(f["ts"])]
    span[(m, k)].append(max(times) - min(times))

retry = {p.name: len(re.findall(r"RETRY", p.read_text())) for p in sorted((R / "jetson").glob("router_r28*.out"))}
print("failures:", dict(fails), "| vLLM timeouts (RETRY lines) per Jetson router log:", retry)

# ---- table
print("\nper condition, 3 repetitions pooled; times in seconds; throughput per repetition (mean, min-max)")
hdr = (f"{'method':14s}{'k':>2s} | {'sent':>4s} {'Pi':>3s} {'Jet':>3s} | {'resp mean':>9s} {'p95':>6s} {'rep means':>13s} "
       f"| {'wait mean':>9s} {'p95':>6s} | {'service':>7s} | {'k/X':>6s} | {'req/h':>11s} {'img/h':>6s} | all 60: mean  p95")
print(hdr)
csv = [hdr.replace("|", ",")]
for m in METHODS:
    for k in KS:
        reqs = per[(m, k)]
        sent = [q for q in reqs if not q.get("m5_end")]
        n_rep = len({q["rep"] for q in reqs})
        resp = np.array([q["resp"] for q in sent])
        wait = np.array([q["wait"] for q in sent])
        svc = np.array([q["svc"] for q in sent])
        reps = [np.mean([q["resp"] for q in sent if q["rep"] == r]) for r in sorted({q["rep"] for q in sent})]
        thr = [len(sent) / n_rep / s * 3600 for s in span[(m, k)]]
        img = [60 / s * 3600 for s in span[(m, k)]]
        allr = np.array([q["resp"] for q in reqs])
        tiers = Counter(q.get("tier", "jetson") for q in sent)
        n_pi = len(sent) // n_rep if m != "direct_jetson" else 0
        n_jet = tiers["jetson"] // n_rep
        line = (f"{NAME[m]:14s}{k:2d} | {len(sent) // n_rep:4d} {n_pi:3d} {n_jet:3d} | {resp.mean():9.1f} {p95(resp):6.1f} "
                f"{min(reps):6.1f}-{max(reps):<6.1f} | {wait.mean():9.1f} {p95(wait):6.1f} | {svc.mean():7.1f} | "
                f"{k / (np.mean(thr) / 3600):6.1f} | {np.mean(thr):4.0f} ({min(thr):.0f}-{max(thr):.0f}) {np.mean(img):6.0f} | "
                f"{allr.mean():6.1f} {p95(allr):6.1f}")
        print(line)
        csv.append(line.replace("|", ","))
(R / "summary_table.txt").write_text("\n".join(csv) + "\n")

# ---- where the time goes in Ours, and whether routing changed under load
o = [q for k in KS for q in per[("ours", k)] if not q.get("m5_end")]
busy = sum(q["svc"] for q in o)
print(f"\nOurs, Pi_A busy time per escalated request: Pi_A VLM {np.mean([q['pi_a_vlm'] for q in o]):.1f} s, "
      f"waiting on the Jetson {np.mean([q['jet_call'] for q in o]):.1f} s (share of Pi_A busy time "
      f"{100 * sum(q['jet_call'] for q in o) / busy:.0f}%), other (Pi_B wait, routing, writing) "
      f"{np.mean([q['svc'] - q['pi_a_vlm'] - q['jet_call'] for q in o]):.1f} s")
same = sum(len({x for x in v if not x.startswith("rerun:")}) == 1 for v in routes.values())
match = sum(len({x.replace("rerun:", "") for x in v}) == 1 for v in routes.values())
print(f"Ours routing: {same}/{len(routes)} images ended at the same tier in all 9 Ours conditions; "
      f"{match}/{len(routes)} also match the MEDIC rerun's route")
