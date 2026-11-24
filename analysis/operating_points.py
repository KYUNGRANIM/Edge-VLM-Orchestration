#!/usr/bin/env python3
"""Compare whole quality-cost curves, not single operating points (held-out 600, MEDIC rerun or EIDSeg).

    python3 analysis/operating_points.py [medic|eidseg]

Each method's thresholds are swept on the held-out run answers, as Fig. 4 does for Ours, to show what each policy
family can reach. Deployed points stay the ones chosen on profiling.
  Ours             the deployed trees' probability cutoffs t1, t2 in 0.05..0.95 (auxiliary devices included)
  Learned cascade  the FrugalGPT-style scorers of learned_cascade.py, thresholds 0.05..0.95 (no auxiliary devices)
  Confidence-Only  all 25 (c1, c2) pairs (no auxiliary devices)
For each Q-Sat floor: the lowest latency and the lowest energy a method reaches at or above it, then the same with
the backhaul cap (share sent to the Jetson at most Ours' deployed share). All points go to operating_points_<set>.csv.
"""
import contextlib
import csv
import io
import runpy
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "medic"
sys.argv = [str(HERE / "learned_cascade.py"), DATASET]
with contextlib.redirect_stdout(io.StringIO()):
    L = runpy.run_path(str(HERE / "learned_cascade.py"))
G, ids, run, alltier, Qr, ours, per_image, routes, GRID = (L[k] for k in
                                                          ("G", "ids", "run", "alltier", "Qr", "ours", "per_image", "routes", "GRID"))
Q_TH, M5VLM = G["Q_TH"], G["M5VLM"]
POL = M5VLM / ("deployment/routers" if DATASET == "medic" else "revision/eidseg/runtime/policies")


def point(method, tiers, aux, note=""):
    sat = np.mean([Qr[i][t] >= Q_TH for i, t in zip(ids, tiers)])
    c = per_image(list(tiers), aux).mean(axis=0)
    return {"method": method, "note": note, "qsat": 100 * sat, "lat": c[0], "kib": c[1], "j": c[2],
            "jet": 100 * np.mean(np.asarray(tiers) == "jetson")}


# ---------------- Ours: deployed trees' probabilities on the held-out features (as paper_numbers.py, Fig. 4)
p1, p2 = joblib.load(POL / "stage1_m5_router.joblib"), joblib.load(POL / "stage2_pi_router.joblib")
sf1, sf2 = G["stage1_features"], G["stage2_features"]
m5 = {i: {k: run[i]["m5"][k] for k in ("m5_a", "m5_b", "m5_c")} for i in ids}
pm5 = p1["model"].predict_proba([[sf1(m5[i])[k] for k in p1["feature_keys"]] for i in ids])[:, 1]
f2 = []
for i in ids:
    f = (run[i].get("stage2") or {}).get("features")
    if not f:
        a = alltier[i]
        f = sf2({**m5[i], "pi_a": a["pi_a"], "pi_b": a["pi_b"], "pi_a_conf_detail": a.get("pi_a_conf_detail") or {}})
    f2.append([f[k] for k in p2["feature_keys"]])
ppi = p2["model"].predict_proba(f2)[:, 1]
replay = routes(pm5, ppi, 0.5, 0.5)
assert list(replay) == list(ours), f"tree replay matches the deployed routes on {sum(a == b for a, b in zip(replay, ours))}/600 only"

pts = [point("Ours", routes(pm5, ppi, a, b), True, f"{a},{b}") for a in GRID for b in GRID]
for text, name in ((True, "Learned cascade (text+labels)"), (False, "Learned cascade (labels only)")):
    s_m5, s_pi = L["SCORES"][text]
    pts += [point(name, routes(s_m5, s_pi, a, b), False, f"{a},{b}") for a in GRID for b in GRID]
run_pi = G["run_pi"]
for a in range(1, 6):
    for b in range(1, 6):
        t = [G["conf_pick"](run[i]["m5"]["m5_a"].get("conf") or 0, (run_pi[i] or alltier[i]["pi_a"]).get("conf") or 0, a, b)
             for i in ids]
        pts.append(point("Confidence-Only", t, False, f"{a},{b}"))
singles = [point("Ours (deployed)", ours, True), point("Fixed@Pi", ["pi_a"] * len(ids), False),
           point("Fixed@Jetson", ["jetson"] * len(ids), False)]
dj = per_image(["jetson"] * len(ids), False, True).mean(axis=0)
singles.append({"method": "Direct@Jetson", "note": "", "qsat": singles[2]["qsat"], "lat": dj[0], "kib": dj[1], "j": dj[2], "jet": 100.0})
out = M5VLM / ("analysis/results" if DATASET == "medic" else "revision/eidseg/runtime/results") / f"operating_points_{DATASET}.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(pts[0]))
    w.writeheader()
    w.writerows(pts + singles)

dep = singles[0]
cap = dep["jet"]
methods = ["Ours", "Learned cascade (text+labels)", "Learned cascade (labels only)", "Confidence-Only"]
print(f"{DATASET.upper()} held-out 600: Ours deployed Q-Sat {dep['qsat']:.1f}%, {dep['lat']:.1f} s, {dep['j']:.1f} J, "
      f"{dep['jet']:.1f}% to Jetson; Direct@Jetson {singles[3]['qsat']:.1f}%, {singles[3]['lat']:.1f} s, {singles[3]['j']:.1f} J")
fmt = "{note:>9s}: {qsat:5.1f}%  {lat:5.1f} s  {j:6.1f} J  {kib:4.0f} KiB  {jet:5.1f}% to Jetson"
print("\n  thresholds chosen on profiling (deployed / pre-specified), and where they land on the held-out set")
print(f"    Ours deployed  " + fmt.format(**{**dep, "note": "0.5,0.5"}))
for text, name in ((True, "Learned cascade (text+labels)"), (False, "Learned cascade (labels only)")):
    for rule, (g, _) in L["LC"][text].items():
        p = next(p for p in pts if p["method"] == name and p["note"] == f"{g[2]},{g[3]}")
        print(f"    {name[16:]:15s} {rule:13s} " + fmt.format(**p))
p = next(p for p in pts if p["method"] == "Confidence-Only" and p["note"] == f"{G['C1']},{G['C2']}")
print(f"    Confidence-Only (matched rule) " + fmt.format(**p))
# every point below is one threshold pair; its five numbers come from the same run of the policy
floors = sorted({dep["qsat"]} | ({62.5, 65, 70} if DATASET == "medic" else {77.5, 80}))
for title, ok in (("any backhaul use", lambda p: True), (f"backhaul cap: at most {cap:.1f}% of images to the Jetson",
                                                          lambda p: p["jet"] <= cap + 1e-9)):
    print(f"\n  threshold sweep on the held-out set (sensitivity analysis, {title}): at each Q-Sat floor, the lowest-energy "
          f"point and the fastest point, each with its own values")
    for fl in floors:
        print(f"    Q-Sat >= {fl:.1f}%")
        for m in methods:
            xs = [p for p in pts if p["method"] == m and p["qsat"] >= fl - 1e-9 and ok(p)]
            if not xs:
                print(f"      {m:30s} unreachable")
                continue
            e, f = min(xs, key=lambda p: (p["j"], p["lat"])), min(xs, key=lambda p: (p["lat"], p["j"]))
            print(f"      {m:30s} lowest energy " + fmt.format(**e))
            if f is not e:
                print(f"      {'':30s} fastest       " + fmt.format(**f))
better = [p for p in pts if p["method"].startswith("Learned") and p["qsat"] >= dep["qsat"] - 1e-9 and p["lat"] <= dep["lat"]
          and p["j"] <= dep["j"] and p["jet"] <= cap + 1e-9]
print(f"\n  learned-cascade points at least as good as Ours' deployed point on Q-Sat, latency, energy and backhaul: {len(better)}"
      + ("" if not better else " (best energy: " + min(better, key=lambda p: p["j"])["method"] + " "
         + "{note}: {qsat:.1f}%, {lat:.1f} s, {j:.1f} J, {jet:.1f}% to Jetson)".format(**min(better, key=lambda p: p["j"]))))
print(f"  points written to {out.relative_to(M5VLM)}")
