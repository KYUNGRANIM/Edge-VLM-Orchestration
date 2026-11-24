#!/usr/bin/env python3
"""What the all-tier offline profiling costs, and how much of it the routers need.

    /opt/anaconda3/bin/python3 analysis/router_analysis/profiling_cost.py     # scikit-learn 1.5.1 as in requirements.txt

A. Cost of profiling MEDIC (5,526 images) and EIDSeg (2,331 images): device time, energy, GPT-4o judge tokens.
   Auxiliary devices (M5_B, M5_C, Pi_B) are not timed in the collection; their per-image times are the measured
   constants of the first submission's Table I.
B. Learning curve: routers trained on a stratified fraction of the MEDIC profiling set, replayed on the 600
   held-out images with the all-tier answers (features and quality from the same answers).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

M5VLM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(M5VLM / "profiling" / "code" / "pipeline"))
from common import make_dt, stage1_features, stage2_features, to_matrix  # noqa: E402

P_M5, P_PI = 1.2154, 7.5564
AUX_M5_S, AUX_PI_S = 6.08, 31.35            # per auxiliary device, per image (first submission's Table I)
PRICE_IN, PRICE_OUT = 2.50e-6, 10.0e-6       # gpt-4o, USD per token
Q_TH, W = 0.65, (0.3, 0.3, 0.4)


def jl(path, type_=None):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if type_ is None or r.get("type") == type_:
                out[r["image_id"]] = r
    return out


def profiling_cost(name, coll, judge):
    lat = {k: np.array([r["stage_latency_ms"][k] for r in coll]) / 1000
           for k in ("m5_total", "m5_pi_http", "pi_total", "pi_jet_http", "jetson_total")}
    n = len(coll)
    hours = {"M5_A": lat["m5_total"].sum(), "M5_B": AUX_M5_S * n, "M5_C": AUX_M5_S * n,
             "Pi_A": lat["pi_total"].sum(), "Pi_B": AUX_PI_S * n, "Jetson": lat["jetson_total"].sum()}
    energy = sum(sum(r["stage_energy_j"].values()) for r in coll) + n * (2 * P_M5 * AUX_M5_S + P_PI * AUX_PI_S)
    tin = sum((j.get("usage") or {}).get("prompt_tokens", 0) for j in judge)
    tout = sum((j.get("usage") or {}).get("completion_tokens", 0) for j in judge)
    per_img = sum(v.mean() for v in lat.values())
    print(f"--- {name}: {n} images profiled, {len(judge)} judged")
    print(f"  one image through all tiers in sequence: {per_img:.1f} s "
          f"(M5 {lat['m5_total'].mean():.1f}, Pi {lat['pi_total'].mean():.1f}, Jetson {lat['jetson_total'].mean():.1f})")
    print(f"  device time (h): " + ", ".join(f"{k} {v / 3600:.0f}" for k, v in hours.items())
          + f"  | all in sequence {per_img * n / 3600:.0f} h")
    print(f"  energy {energy / 3.6e6:.2f} kWh | judge {tin / 1e6:.1f}M in + {tout / 1e6:.2f}M out tokens "
          f"= ${tin * PRICE_IN + tout * PRICE_OUT:.0f}")


# ---------------- A. profiling cost ----------------
medic = list(jl(M5VLM / "profiling/results/collection.jsonl").values())
profiling_cost("MEDIC", medic, list(jl(M5VLM / "profiling/results/pipeline_outputs/00_gpt_judge_scores.jsonl").values()))
eid_full = jl(M5VLM / "revision/eidseg/collection_eidseg_full.jsonl")
eid_prof = jl(M5VLM / "revision/eidseg/collection_eidseg_profiling.jsonl")
eid_judge = [j for i, j in jl(M5VLM / "revision/eidseg/labels/00_judge_all.jsonl").items() if i in eid_prof]
profiling_cost("EIDSeg", [eid_full[i] for i in eid_prof], eid_judge)

# ---------------- B. learning curve ----------------
oracle = jl(M5VLM / "profiling/results/pipeline_outputs/03_oracle_labels.jsonl", "oracle_label")
pids = sorted(set(oracle) & {r["image_id"] for r in medic})
prec = {r["image_id"]: r for r in medic}
y = np.array([oracle[i]["oracle_label"] for i in pids], dtype=object)
x1, k1, c1, n1 = to_matrix([stage1_features(prec[i]) for i in pids])
x2, k2, c2, n2 = to_matrix([stage2_features(prec[i]) for i in pids])

H = M5VLM / "heldout/results/labels"
hq, hc = jl(H / "01_quality_components.jsonl", "quality_components"), jl(H / "02_cost_components.jsonl", "cost_components")
hall = jl(M5VLM / "heldout/results/collection.jsonl")
hids = sorted(set(hq) & set(hc) & set(hall))
hx1 = [[stage1_features(hall[i])[k] for k in k1] for i in hids]
hx2 = [[stage2_features(hall[i])[k] for k in k2] for i in hids]
quality = lambda c: round(W[0] * c["cat_score"] + W[1] * c["sev_score"] + W[2] * c["text_score"], 8)


def replay(idx):
    m1 = make_dt(c1, n1, depth=4).fit([x1[i] for i in idx], (y[idx] == "m5_a").astype(int))
    s2 = [i for i in idx if y[i] != "m5_a"]
    m2 = make_dt(c2, n2, depth=4).fit([x2[i] for i in s2], (y[s2] == "pi_a").astype(int))
    e1, e2 = m1.predict(hx1), m2.predict(hx2)
    tiers = ["m5_a" if a else ("pi_a" if b else "jetson") for a, b in zip(e1, e2)]
    qsat = 100 * np.mean([quality(hq[i][t]) >= Q_TH for i, t in zip(hids, tiers)])
    eng = np.mean([hc[i][t]["energy_j"] + 2 * P_M5 * AUX_M5_S + (P_PI * AUX_PI_S if t != "m5_a" else 0)
                   for i, t in zip(hids, tiers)])
    lat = np.mean([hc[i][t]["latency_ms"] for i, t in zip(hids, tiers)]) / 1000
    return qsat, eng, lat, tiers


full_q, full_e, full_l, full_t = replay(np.arange(len(pids)))
print(f"--- learning curve on {len(hids)} held-out images (all-tier answers), 10 draws per fraction")
print(f"  100%  ({len(pids)} images): Q-Sat {full_q:.1f}%  energy {full_e:.0f} J  latency {full_l:.1f} s  "
      f"tiers {'/'.join(str(Counter(full_t)[t]) for t in ('m5_a', 'pi_a', 'jetson'))}")
for frac in (0.05, 0.10, 0.25, 0.50):
    res = []
    for seed in range(10):
        idx, _ = train_test_split(np.arange(len(pids)), train_size=frac, stratify=y, random_state=seed)
        q, e, l, t = replay(np.sort(idx))
        res.append((q, e, l, np.mean([a == b for a, b in zip(t, full_t)])))
    r = np.array(res)
    print(f"  {int(frac * 100):3d}%  ({int(frac * len(pids))} images): Q-Sat {r[:, 0].mean():.1f}% "
          f"[{r[:, 0].min():.1f}–{r[:, 0].max():.1f}]  energy {r[:, 1].mean():.0f} J  latency {r[:, 2].mean():.1f} s  "
          f"same tier as 100% {100 * r[:, 3].mean():.0f}%")
