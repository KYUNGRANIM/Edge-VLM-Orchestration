#!/usr/bin/env python3
"""Sensitivity to the quality weights and the reference tolerance, and Q-Sat passes with wrong labels.

All scores use the corrected severity mapping (current 01_quality_components files).

  A. Re-score: keep every method's held-out tier assignment fixed (Ours = deployed runtime run) and
     recompute Q-Sat under other quality weights; the reference is rebuilt for each weight/tolerance.
  B. Retrain: for each weight/tolerance, rebuild the profiling reference labels, retrain both routers
     (depth-4 trees, full profiling set), and replay them on the held-out 600 as the first submission did.
  C. Passes with wrong labels: held-out responses with Q >= 0.65 despite a wrong category or severity,
     and Table I methods under stricter pass rules.

Reference = earliest tier within the tolerance of the best tier (identical to the composite-cost argmin,
because cumulative costs rise with tier for every image). Costs follow the first submission's Table I.

    python3 analysis/quality_sensitivity_first_run.py
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

M5VLM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M5VLM / "profiling" / "code" / "pipeline"))
from common import make_dt, stage1_features, stage2_features, to_matrix  # noqa: E402

HELD = M5VLM / "heldout/results/labels"
TIERS = ("m5_a", "pi_a", "jetson")
Q_TH = 0.65
NOMINAL = (0.3, 0.3, 0.4)
WEIGHTS = {  # (category, severity, text)
    "nominal 0.3/0.3/0.4": NOMINAL,
    "category-heavy 0.4/0.3/0.3": (0.4, 0.3, 0.3),
    "severity-heavy 0.3/0.4/0.3": (0.3, 0.4, 0.3),
    "labels-heavy 0.4/0.4/0.2": (0.4, 0.4, 0.2),
    "equal 1/3 each": (1 / 3, 1 / 3, 1 / 3),
    "text-heavy 0.25/0.25/0.5": (0.25, 0.25, 0.5),
    "text-heavy 0.2/0.2/0.6": (0.2, 0.2, 0.6),
}
TAUS = (0.05, 0.10, 0.15)
CONF_C1, CONF_C2 = 5, 3          # Confidence-Only thresholds as in table1_final.csv
AUX_M5_ENERGY = 2 * 1.2154 * 6.08
AUX_PI_ENERGY = 7.5564 * 31.35
AUX_PI_OVERHEAD = 218.7 * 1024


def jl(path: Path, type_: str | None = None) -> dict:
    out = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            if type_ is None or r.get("type") == type_:
                out[r["image_id"]] = r
    return out


def quality(comp: dict, tier: str, w) -> float:
    c = comp[tier]
    return round(w[0] * c["cat_score"] + w[1] * c["sev_score"] + w[2] * c["text_score"], 8)


def reference(comp: dict, w, tau: float) -> str:
    q = {t: quality(comp, t, w) for t in TIERS}
    best = max(q.values())
    return next(t for t in TIERS if best - q[t] <= tau + 1e-12)


def conf_pick(m5_conf, pi_conf):
    if m5_conf >= CONF_C1:
        return "m5_a"
    return "pi_a" if pi_conf >= CONF_C2 else "jetson"


# ---------------- held-out data (same sources as the first submission) ----------------
qc = jl(HELD / "01_quality_components.jsonl", "quality_components")
cost = jl(HELD / "02_cost_components.jsonl", "cost_components")
routing = {r["image_id"]: r for r in map(json.loads, (M5VLM / "revision/first_submission/live_run/online_routing_final.jsonl").open())
           if r.get("type") == "online_routing_final" and r.get("status") == "ok"}
alltier = jl(M5VLM / "heldout/results/collection.jsonl")
ids = sorted(set(qc) & set(cost) & set(routing) & set(alltier))
n = len(ids)


def summarise(tiers, w, aux) -> dict:
    sat, lat, eng = [], [], []
    for i, t in zip(ids, tiers, strict=True):
        c = cost[i][t]
        e = c["energy_j"] + ((AUX_M5_ENERGY + (AUX_PI_ENERGY if t != "m5_a" else 0)) if aux else 0)
        sat.append(quality(qc[i], t, w) >= Q_TH)
        lat.append(c["latency_ms"] / 1000)
        eng.append(e)
    return {"qsat": 100 * float(np.mean(sat)), "lat": float(np.mean(lat)), "eng": float(np.mean(eng)),
            "dist": "/".join(str(sum(t == x for t in tiers)) for x in TIERS)}


ours = [routing[i]["route_label"] for i in ids]
conf_only = [conf_pick(routing[i]["m5"]["m5_a"].get("conf") or 0,
                       ((routing[i].get("pi") or {}).get("pi_a") or alltier[i].get("pi_a") or {}).get("conf") or 0)
             for i in ids]
fixed = {f"Fixed@{x}": [t] * n for x, t in (("M5", "m5_a"), ("Pi", "pi_a"), ("Jetson", "jetson"))}

# replay features: deployed Stage 1 inputs, deployed Stage 2 inputs where Stage 2 ran, else all-tier Pi
h1 = [stage1_features({k: routing[i]["m5"][k] for k in ("m5_a", "m5_b", "m5_c")}) for i in ids]
h2 = []
for i in ids:
    st2 = routing[i].get("stage2") or {}
    if "features" in st2:
        h2.append(st2["features"])
    else:
        a = alltier[i]
        h2.append(stage2_features({**{k: routing[i]["m5"][k] for k in ("m5_a", "m5_b", "m5_c")},
                                   "pi_a": a["pi_a"], "pi_b": a["pi_b"],
                                   "pi_a_conf_detail": a.get("pi_a_conf_detail") or {}}))

# ---------------- profiling data for retraining ----------------
pqc = jl(M5VLM / "profiling/results/pipeline_outputs/01_quality_components.jsonl", "quality_components")
prof = jl(M5VLM / "profiling/results/collection.jsonl")
pids = sorted(set(pqc) & set(prof))
x1, k1, c1, n1 = to_matrix([stage1_features(prof[i]) for i in pids])
x2, k2, c2, n2 = to_matrix([stage2_features(prof[i]) for i in pids])
hx1 = [[f[k] for k in k1] for f in h1]
hx2 = [[f[k] for k in k2] for f in h2]


def retrained_tiers(w, tau):
    y = np.array([reference(pqc[i], w, tau) for i in pids], dtype=object)
    m1 = make_dt(c1, n1, depth=4).fit(x1, (y == "m5_a").astype(int))
    s2 = [j for j in range(len(pids)) if y[j] != "m5_a"]
    m2 = make_dt(c2, n2, depth=4).fit([x2[j] for j in s2], (y[s2] == "pi_a").astype(int))
    p1, p2 = m1.predict(hx1), m2.predict(hx2)
    return ["m5_a" if p1[j] == 1 else ("pi_a" if p2[j] == 1 else "jetson") for j in range(n)], len(s2)


def main() -> None:
    # sanity: nominal re-score must reproduce Table I
    t1 = {k: summarise(v, NOMINAL, False)["qsat"] for k, v in fixed.items()}
    assert [round(t1[k], 1) for k in fixed] == [48.0, 64.3, 75.2], t1
    assert round(summarise(ours, NOMINAL, True)["qsat"], 1) == 63.8
    assert round(summarise(conf_only, NOMINAL, False)["qsat"], 1) == 68.7

    print(f"held-out N = {n}\n")
    print("A. Re-score (tier assignments fixed; Q-Sat %, reference rebuilt at tolerance 0.10)")
    print(f"{'weights':28s} {'Fx@M5':>6} {'Fx@Pi':>6} {'Fx@Jet':>6} {'ConfOnly':>8} {'Ours':>6} {'Ref':>6} {'Ours-Pi':>8}")
    for name, w in WEIGHTS.items():
        f = {k: summarise(v, w, False)["qsat"] for k, v in fixed.items()}
        o = summarise(ours, w, True)["qsat"]
        r = summarise([reference(qc[i], w, 0.10) for i in ids], w, False)["qsat"]
        print(f"{name:28s} {f['Fixed@M5']:6.1f} {f['Fixed@Pi']:6.1f} {f['Fixed@Jetson']:6.1f} "
              f"{summarise(conf_only, w, False)['qsat']:8.1f} {o:6.1f} {r:6.1f} {o - f['Fixed@Pi']:+8.1f}")
    print("\n   reference Q-Sat and tier split by tolerance (nominal weights)")
    for tau in TAUS:
        s = summarise([reference(qc[i], NOMINAL, tau) for i in ids], NOMINAL, False)
        print(f"   tolerance {tau:.2f}: Q-Sat {s['qsat']:.1f}%  M5/Pi/Jetson {s['dist']}")

    grid = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1)) for a, b in product(range(1, 9), repeat=2) if a + b <= 9]
    gaps = [summarise(ours, w, True)["qsat"] - summarise(fixed["Fixed@Pi"], w, False)["qsat"] for w in grid]
    print(f"\n   all {len(grid)} weight sets on a 0.1 grid: Ours minus Fixed@Pi ranges {min(gaps):+.1f} to {max(gaps):+.1f} pp, "
          f"Ours >= Fixed@Pi in {sum(g >= 0 for g in gaps)}")

    print("\nB. Retrain on rebuilt references and replay on held-out (Q-Sat with the same weights; aux costs included)")
    print(f"{'weights':28s} {'tol':>5} {'S2 n':>6} {'Q-Sat':>6} {'lat s':>6} {'E J':>7} {'M5/Pi/Jet':>12} {'Fx@Pi':>6}")
    for (name, w), tau in [(kv, 0.10) for kv in WEIGHTS.items()] + [(("nominal 0.3/0.3/0.4", NOMINAL), t) for t in (0.05, 0.15)]:
        tiers, n2_ = retrained_tiers(w, tau)
        s = summarise(tiers, w, True)
        print(f"{name:28s} {tau:5.2f} {n2_:6d} {s['qsat']:6.1f} {s['lat']:6.1f} {s['eng']:7.1f} {s['dist']:>12} "
              f"{summarise(fixed['Fixed@Pi'], w, False)['qsat']:6.1f}")

    print("\nC. Passes (Q >= 0.65, nominal weights) with wrong labels, held-out all-tier responses")
    print(f"{'tier':7s} {'pass':>5} {'cat wrong':>10} {'cat ok, sev off 1':>18} {'cat ok, sev off 2':>18} {'both exact':>11}")
    for t in TIERS:
        ps = [qc[i] for i in ids if quality(qc[i], t, NOMINAL) >= Q_TH]
        cw = sum(p[t]["cat_score"] == 0 for p in ps)
        s1 = sum(p[t]["cat_score"] == 1 and p[t]["sev_score"] == 0.5 for p in ps)
        s2 = sum(p[t]["cat_score"] == 1 and p[t]["sev_score"] == 0 for p in ps)
        ex = sum(p[t]["cat_score"] == 1 and p[t]["sev_score"] == 1 for p in ps)
        print(f"{t:7s} {len(ps):5d} {cw:10d} {s1:18d} {s2:18d} {ex:11d}")

    print("\n   Table I methods under stricter pass rules (Q-Sat %)")
    rules = {"Q>=0.65": lambda c: True,
             "+ category correct": lambda c: c["cat_score"] == 1,
             "+ category correct, severity within 1": lambda c: c["cat_score"] == 1 and c["sev_score"] >= 0.5,
             "+ category and severity exact": lambda c: c["cat_score"] == 1 and c["sev_score"] == 1}
    methods = {**fixed, "ConfOnly": conf_only, "Ours": ours,
               "Reference": [reference(qc[i], NOMINAL, 0.10) for i in ids]}
    print(f"{'rule':40s} " + " ".join(f"{m:>11s}" for m in methods))
    for rn, rule in rules.items():
        vals = [100 * np.mean([quality(qc[i], t, NOMINAL) >= Q_TH and rule(qc[i][t]) for i, t in zip(ids, v)])
                for v in methods.values()]
        print(f"{rn:40s} " + " ".join(f"{x:11.1f}" for x in vals))


if __name__ == "__main__":
    main()
