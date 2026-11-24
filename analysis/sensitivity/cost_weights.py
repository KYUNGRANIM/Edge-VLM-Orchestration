#!/usr/bin/env python3
"""When do the cost weights (0.1/0.1/0.8) change anything?

    python3 analysis/sensitivity/cost_weights.py

1. Per image, do cumulative latency, communication and energy never decrease with tier? (profiling and held-out)
2. Reference labels under other positive weightings, and against "earliest tier within the tolerance".
3. Where cost does enter: the quality tolerance. Reference tiers, Q-Sat and cost for tau = 0 (best quality) .. 0.20.
4. Where the weights enter today: the Confidence-Only thresholds matched on composite cost.
5. When weights would change labels: a skip-tier counterfactual (M5 escalates straight to the Jetson over a link of
   bandwidth b). Not measured; built from the profiling stage costs.
"""
import contextlib
import io
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(HERE.parent / "main_results/results_table.py"))
M5VLM = G["M5VLM"]
T, TI = ("m5_a", "pi_a", "jetson"), {"m5_a": 0, "pi_a": 1, "jetson": 2}
P = M5VLM / "profiling/results/pipeline_outputs"
H = M5VLM / "heldout/results/labels"
WEIGHTS = {"paper 0.1/0.1/0.8": (0.1, 0.1, 0.8), "equal": (1 / 3, 1 / 3, 1 / 3), "latency 0.98": (0.98, 0.01, 0.01),
           "comm 0.98": (0.01, 0.98, 0.01), "energy 0.98": (0.01, 0.01, 0.98)}


def jl(path, t):
    return {r["image_id"]: r for r in map(json.loads, open(path)) if r.get("type") == t}


def label(q, c, tau):
    best = max(q.values())
    feas = [t for t in T if best - q[t] <= tau + 1e-12]
    return min(feas, key=lambda t: (c[t], -q[t])), min(feas, key=TI.get)


# ---- 1-2 on the stored pipeline outputs
for name, d in (("profiling", P), ("held-out", H)):
    cost, orc = jl(d / "02_cost_components.jsonl", "cost_components"), jl(d / "03_oracle_labels.jsonl", "oracle_label")
    ids = sorted(set(cost) & set(orc))
    dec = sum(any(not (cost[i]["m5_a"][k] <= cost[i]["pi_a"][k] <= cost[i]["jetson"][k]) for k in ("latency_ms", "overhead_bytes", "energy_j"))
              for i in ids)
    print(f"[1] {name}: {len(ids)} images, cumulative cost decreasing with tier in any component: {dec}")
    for wn, w in WEIGHTS.items():
        ch = early = 0
        for i in ids:
            c = {t: w[0] * cost[i][t]["latency_norm"] + w[1] * cost[i][t]["overhead_norm"] + w[2] * cost[i][t]["energy_norm"] for t in T}
            lab, first = label(orc[i]["quality"], c, 0.10)
            ch += lab != orc[i]["oracle_label"]
            early += lab != first
        print(f"[2]   weights {wn:18s}: labels differing from the stored ones {ch}, from the earliest feasible tier {early}")

# ---- 3 tolerance: profiling labels, and the held-out reference on the run answer set (as Table I)
pq = {i: r["quality"] for i, r in jl(P / "03_oracle_labels.jsonl", "oracle_label").items()}
ids, Qr, hc = G["ids"], G["Q"]["run"], G["cost"]
print("[3] tau | profiling reference M5/Pi/Jetson, % below the best-quality tier | held-out (run answers) reference "
      "M5/Pi/Jetson, Q-Sat, latency s, energy J")
for tau in (0.0, 0.05, 0.10, 0.15, 0.20):
    pl = [label(q, {t: TI[t] for t in T}, tau)[0] for q in pq.values()]
    pb = [label(q, {t: TI[t] for t in T}, 0.0)[0] for q in pq.values()]
    hl = [label(Qr[i], {t: TI[t] for t in T}, tau)[0] for i in ids]
    c = np.array([[hc[i][t]["latency_ms"] / 1000, hc[i][t]["energy_j"]] for i, t in zip(ids, hl)]).mean(axis=0)
    pc, hcn = Counter(pl), Counter(hl)
    print(f"    {tau:.2f} | {pc['m5_a']}/{pc['pi_a']}/{pc['jetson']}, {100 * np.mean([TI[a] < TI[b] for a, b in zip(pl, pb)]):.1f}% | "
          f"{hcn['m5_a']}/{hcn['pi_a']}/{hcn['jetson']}, {100 * np.mean([Qr[i][t] >= 0.65 for i, t in zip(ids, hl)]):.1f}%, {c[0]:.1f}, {c[1]:.1f}")

# ---- 4 Confidence-Only thresholds matched on composite cost under each weighting
pcost, pids, oof, prof, NORM, AUXE = G["pcost"], G["pids"], G["oof"], G["prof"], G["NORM"], (G["AUX_M5_J"], G["AUX_PI_J"], G["AUX_PI_B"])


def comp(c, t, aux, w):
    lat, ovh, eng = c[t]["latency_ms"], c[t]["overhead_bytes"], c[t]["energy_j"]
    if aux:
        eng += AUXE[0] + (AUXE[1] if t != "m5_a" else 0)
        ovh += AUXE[2] if t != "m5_a" else 0
    nv = [min(max((v - NORM[k][0]) / NORM[k][1], 0.0), 1.0) for k, v in (("latency_ms", lat), ("overhead_bytes", ovh), ("energy_j", eng))]
    return w[0] * nv[0] + w[1] * nv[1] + w[2] * nv[2]


pick = G["conf_pick"]
for wn, w in WEIGHTS.items():
    target = np.mean([comp(pcost[i], t, True, w) for i, t in zip(pids, oof)])
    cand = sorted((abs(np.mean([comp(pcost[i], pick(prof[i]["m5_a"].get("conf") or 0, prof[i]["pi_a"].get("conf") or 0, a, b), False, w)
                                for i in pids]) - target), a, b) for a in range(1, 6) for b in range(1, 6))
    print(f"[4] weights {wn:18s}: Confidence-Only thresholds ({cand[0][1]}, {cand[0][2]})")

# ---- 5 skip-tier counterfactual on the profiling stage costs
rows = [r for r in map(json.loads, open(M5VLM / "profiling/results/collection.jsonl")) if r["image_id"] in pq]
print("[5] skip-tier: M5 -> Pi, or M5 -> Jetson directly over a link of b Mbit/s (extra transfer time = upload bytes / b)")
for bw in (None, 10.0, 5.0, 2.0, 1.0, 0.5):
    lab = {wn: [] for wn in WEIGHTS}
    conflict = 0
    raw = []
    for r in rows:
        sl, so, se = r["stage_latency_ms"], r["stage_overhead_bytes"], r["stage_energy_j"]
        up = (so["m5_pi_tx"] or so["pi_jet_tx"]) + (so["m5_pi_rx"] or 311)
        extra = 0.0 if bw is None else up * 8 / (bw * 1e6) * 1000
        raw.append({"m5_a": (sl["m5_total"], 0.0, se["m5"]),
                    "pi_a": (sl["m5_total"] + sl["m5_pi_http"] + sl["pi_total"], up, se["m5"] + se["pi"]),
                    "jetson": (sl["m5_total"] + sl["m5_pi_http"] + extra + sl["jetson_total"], up, se["m5"] + se["jetson"])})
    arr = np.array([[v[t] for t in T] for v in raw])  # image x tier x component
    lo, hi = arr.reshape(-1, 3).min(axis=0), np.percentile(arr.reshape(-1, 3), 95, axis=0)
    for r, v in zip(rows, raw):
        a, b = np.array(v["pi_a"]), np.array(v["jetson"])
        conflict += bool(np.any(a < b) and np.any(a > b))
        for wn, w in WEIGHTS.items():
            nv = {t: np.clip((np.array(v[t]) - lo) / (hi - lo), 0, 1) for t in T}
            lab[wn].append(label(pq[r["image_id"]], {t: float(np.dot(w, nv[t])) for t in T}, 0.10)[0])
    base = lab["paper 0.1/0.1/0.8"]
    d = Counter(base)
    print(f"    {'measured link' if bw is None else f'{bw:>4} Mbit/s':13s}: Pi vs Jetson costs conflict in {conflict} images; "
          f"paper-weight labels {d['m5_a']}/{d['pi_a']}/{d['jetson']}; labels changed vs paper weights: "
          + ", ".join(f"{wn} {sum(x != y for x, y in zip(lab[wn], base))}" for wn in WEIGHTS if wn != "paper 0.1/0.1/0.8"))
