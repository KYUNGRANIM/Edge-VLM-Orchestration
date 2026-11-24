#!/usr/bin/env python3
"""Re-derive the stored labels, confidences and response choices from the documented rules.

    python3 -B analysis/output_checks/verify_response_rules.py

M5 runs C++ firmware, so its parser and confidence score are ported below line by line from tier1.cpp
(classify_from_text, compute_fine_score, score_to_conf_quintile). Pi and Jetson are checked with the repo's own
engine functions (runtime copy for the final run, collection copy for the profiling data). Read-only.
"""
import glob
import json
import re
import runpy
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
M5VLM = Path(__file__).resolve().parents[2]
FINAL = M5VLM / "deployment/live_run"
PROFILING = M5VLM / "profiling/results/collection.jsonl"
EARLIER = M5VLM / "heldout/results/collection.jsonl"
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]  # noqa: E731
C = Counter()


def chk(name, ok):
    C[name, bool(ok)] += 1


# ---------------- M5: port of tier1.cpp ----------------
def has(low, *ks):
    return any(k in low for k in ks)


def m5_labels(text):
    low = text.lower()
    cat, direct = None, True
    if has(low, "earthquake", "quake"): cat = "earthquake"
    elif has(low, "hurricane", "cyclone", "typhoon", "tornado"): cat = "hurricane"
    elif has(low, "landslide", "mudslide", "rockslide", "rockfall"): cat = "landslide"
    elif has(low, "flood", "inundat"): cat = "flood"
    elif has(low, "fire", "wildfire", "blaze", "flame"): cat = "fire"
    else: direct = False
    if cat == "landslide" and not has(low, "slope", "hillside", "mud", "soil", "debris flow", "mudflow", "rockfall", "erosion",
                                      "cliff face", "embankment") \
            and has(low, "concrete", "structural", "rubble", "crumbled", "earthquake", "aftershock", "collapsed building",
                    "cracked wall", "collapsed structure"):
        cat = "earthquake"
    if not direct:
        if "burn" in low and has(low, "burning", "burned", "burnt", "smoke"):
            cat = "fire"
        else:
            e = 4 * has(low, "seismic", "tremor", "aftershock") + 3 * has(low, "rubble", "crumbled") + 4 * has(low, "fissure", "sinkhole") \
                + 3 * has(low, "collapsed building", "collapsed structure", "building collapse") \
                + 2 * ("structural" in low and has(low, "collapse", "damage")) + 2 * ("cracked" in low and has(low, "wall", "road", "foundation")) \
                + 2 * ("concrete" in low and has(low, "collapse", "damage", "crumble"))
            h = 4 * has(low, "storm surge", "storm damage", "tropical storm") + 2 * has(low, "wind", "gust") + 3 * ("uprooted" in low) \
                + 3 * has(low, "fallen tree", "tree fell", "toppled tree") + 2 * ("roof" in low and has(low, "blown", "torn")) \
                + 2 * has(low, "power line", "powerline", "downed") + 3 * ("coastal" in low and has(low, "damage", "flood"))
            f = 4 * has(low, "submerge", "waterlog") + 3 * has(low, "overflow", "water level") + 4 * ("inundat" in low) \
                + 2 * ("water" in low and has(low, "rising", "rose"))
            la = 3 * has(low, "slope", "hillside") + 3 * has(low, "cliff", "boulder") + 3 * has(low, "erosion", "eroded") \
                + 4 * has(low, "debris flow", "mudflow")
            weak = la < 3 and f < 3 and h < 3
            e += weak * has(low, "collapsed", "collapse") + weak * ("debris" in low)
            best = max(e, h, f, la)
            cat = ("other_disaster" if best < 2 else "earthquake" if e >= h and e >= f and e >= la
                   else "flood" if f >= h and f >= la else "hurricane" if h >= la else "landslide")
    if "severe" in low and not has(low, "not severe", "no severe", "not a severe"): sev = 2
    elif "moderate" in low: sev = 1
    elif has(low, "mild", "low severity", "minor", "low damage", "minimal damage", "little or no", "little to no"): sev = 0
    else:
        s = 2 * has(low, "massive", "devastating", "catastroph", "widespread") + 2 * has(low, "casualt", "killed", "fatal", "dead", "death") \
            + 2 * has(low, "significant", "major", "extensive", "destruct") + has(low, "destroy", "collaps", "rubble") \
            + has(low, "injur", "trapped", "evacuat", "rescue") + has(low, "block", "scatter", "displace")
        sev = 2 if s >= 3 else 1 if s >= 2 else 0
    return cat, sev


CAT_KW = {"landslide": ["landslide", "mudslide", "rockslide", "rockfall", "slope", "cliff", "erosion", "boulder", "mudflow"],
          "earthquake": ["earthquake", "quake", "seismic", "fissure", "sinkhole", "rubble", "crumbled", "tremor"],
          "flood": ["flood", "submerge", "inundat", "waterlog", "overflow", "water level"],
          "hurricane": ["hurricane", "cyclone", "typhoon", "storm", "wind", "gust", "surge"],
          "fire": ["fire", "wildfire", "burn", "blaze", "flame", "smoke", "ash", "scorch"]}
SPECIFIC = ["blocked", "buried", "trapped", "injured", "displaced", "evacuate", "shatter", "toppled", "scattered", "piled", "damaged", "severed"]
ANCHORS = ["road", "bridge", "building", "highway", "vehicle", "house", "wall", "roof", "river", "forest", "town", "village", "street"]
HEDGE = [" may ", " might ", " could ", " possibly ", " appears ", "appear ", " seem", "seems ", "seemed ", " perhaps ", " likely ", " unlikely "]
CUTS = {"a": (0.97, 1.20, 1.39, 1.62), "b": (0.55, 0.76, 0.93, 1.13), "c": (0.40, 0.57, 0.75, 0.98)}


def m5_conf(text, cat, role):
    low, f = text.lower(), np.float32  # the firmware computes in single precision
    contra = (cat == "flood" and has(low, "burning", "fire", "blaze", "smoke", "flame")) or \
             (cat == "fire" and has(low, "flood", "submerge", "overflow", "waterlog")) or \
             (cat == "earthquake" and "flood" in low and "quake" not in low)
    s = (f(0.30) * sum(k in low for k in CAT_KW.get(cat, [])) + f(0.25) * sum(k in low for k in SPECIFIC)
         + f(0.25) * sum(k in low for k in ANCHORS) - f(0.30) * sum(k in low for k in HEDGE) - f(0.50) * int(contra)
         + f(0.02) * min(len(re.findall(r"[A-Za-z0-9]+", text)), 40))
    return 1 + sum(bool(s > f(q)) for q in CUTS[role])


for name, rows, get in (("profiling", load(PROFILING), lambda r, k: r[k]), ("earlier held-out", load(EARLIER), lambda r, k: r[k]),
                        ("final run", load(FINAL / "online_routing_final.jsonl"), lambda r, k: r["m5"][k])):
    for role in "abc":
        for r in rows:
            x = get(r, "m5_" + role)
            cat, sev = m5_labels(x["text"])
            chk(f"M5 {name}: cat, sev, conf reproduced", (cat, sev, m5_conf(x["text"], cat, role)) == (x["cat"], x["sev"], x["conf"]))

# ---------------- Pi: the engines' own functions ----------------
A = runpy.run_path(str(M5VLM / "deployment/servers/pi_a_inference_engine.py"), run_name="verify")
B = runpy.run_path(str(M5VLM / "deployment/servers/pi_b_inference_engine.py"), run_name="verify")
A_COL = runpy.run_path(str(M5VLM / "profiling/code/collection/pi_a_inference_pipeline.py"), run_name="verify")
B_COL = runpy.run_path(str(M5VLM / "profiling/code/collection/pi_b_inference_pipeline.py"), run_name="verify")


def lp(cd):  # two tokens with the stored mean and minimum log-probability
    m, n = cd["mean_logprob"], cd["min_logprob"]
    return None if m is None else [{"logprob": n}, {"logprob": 2 * m - n}]


def replay(mod, ag, dbg):
    """run_pi_inference's decision, re-run on the logged raw outputs."""
    if ag == "pi_a":
        cand = [mod["parse_situation_action"](dbg[k]) for k in ("primary_raw", "backup_raw") if k in dbg]
        (s1, a1, h1) = cand[0]
        q1 = mod["analyze_quality"](s1, a1)
        if bool(s1) and bool(a1) and q1["passed"] and not mod["DESCRIPTIVE_ACTION_RE"].search(a1) \
                and not (mod["NEWS_STYLE_RE"].search(s1) and q1["visual_hits"] < 1):
            return "primary", f"{s1} {a1}".strip()
        score = lambda s, a, h: (5 * h + 3 * bool(s and len(s.split()) >= 5) + 3 * bool(a and len(a.split()) >= 3)  # noqa: E731
                                 - mod["analyze_quality"](s, a)["penalty"] + mod["analyze_quality"](s, a)["visual_hits"])
        (s2, a2, h2) = cand[1]
        return ("backup", f"{s2} {a2}".strip()) if score(s2, a2, h2) > score(s1, a1, h1) else ("primary", f"{s1} {a1}".strip())
    t1, h1 = mod["parse_description"](dbg["primary_raw"])
    q1 = mod["analyze_quality"](t1)
    if t1 and len(t1.split()) >= 5 and q1["passed"]:
        return "primary", t1
    t2, h2 = mod["parse_description"](dbg["backup_raw"])
    assert not (mod["is_short_landmark_fragment"](t1, dbg["primary_raw"]) or mod["is_short_landmark_fragment"](t2, dbg["backup_raw"]))
    score = lambda t, h: 5 * h + 3 * bool(t and len(t.split()) >= 5) - mod["analyze_quality"](t)["penalty"] + mod["analyze_quality"](t)["visual_hits"]  # noqa: E731
    return max((("primary", t1, score(t1, h1)), ("backup", t2, score(t2, h2))), key=lambda c: c[2])[:2]


for r in load(FINAL / "online_routing_final.jsonl"):
    if r.get("pi") is None:
        continue
    for ag, mod in (("pi_a", A), ("pi_b", B)):
        o, cd, qg, dbg = r["pi"][ag], r[f"{ag}_conf_detail"], r[f"{ag}_quality_gate"], r[f"{ag}_debug"]
        chk(f"Pi final run {ag}: cat, sev reproduced", (mod["classify_cat"](o["text"]), mod["classify_sev"](o["text"])) == (o["cat"], o["sev"]))
        pen = min(2, int(not qg["passed"]) + int(dbg["chosen"] == "backup"))
        chk(f"Pi final run {ag}: confidence reproduced", mod["compute_confidence"](lp(cd), cd["had_labels"], pen)[0] == o["conf"])
        chk(f"Pi final run {ag}: chosen attempt and text reproduced from raw outputs", replay(mod, ag, dbg) == (dbg["chosen"], o["text"]))
        C[f"Pi final run {ag}: retry generated", "backup_raw" in dbg] += 1
for r in load(PROFILING):
    for ag, mod in (("pi_a", A_COL), ("pi_b", B_COL)):
        o, cd = r[ag], r[f"{ag}_conf_detail"]
        if o["text"] is None:
            C[f"Pi profiling {ag}: no stored output", True] += 1
            continue
        chk(f"Pi profiling {ag}: cat, sev reproduced", (mod["classify_cat"](o["text"]), mod["classify_sev"](o["text"])) == (o["cat"], o["sev"]))
        chk(f"Pi profiling {ag}: confidence reproduced", mod["compute_confidence"](lp(cd), cd["had_labels"], cd["semantic_penalty"])[0] == o["conf"])

# ---------------- Jetson: the engine's own functions ----------------
J = runpy.run_path(str(M5VLM / "deployment/servers/jetson_inference_engine.py"), run_name="verify")
repair = {}
for p in sorted(glob.glob(str(M5VLM / "profiling/results/raw_per_category/jetson_*.jsonl"))):
    for o in load(p):
        repair["profiling", o["image_id"]] = o["end_jetson"]["repair"]
for o in load(FINAL / "device_logs/jetson/online_jetson_results.jsonl"):
    if o["status"] == "ok":
        repair["final run", o["image_id"]] = o["end_jetson"]["repair"]  # the latest ok row is the one kept
recs = [("profiling", r["image_id"], r["jetson"], r["jet_conf_detail"]) for r in load(PROFILING)]
recs += [("final run", r["image_id"], r["jetson"]["jetson"], r["jetson"]["conf_detail"]) for r in load(FINAL / "online_routing_final.jsonl") if r.get("jetson")]
for name, iid, out, cd in recs:
    chk(f"Jetson {name}: confidence reproduced", J["compute_confidence"](lp(cd), cd["had_labels"])[0] == out["conf"])
    chk(f"Jetson {name}: cat, sev valid", out["cat"] in J["VALID_CATS"] and out["sev"] in J["VALID_SEVS"])
    parts = re.split(r"(?<=[.!?])\s+", out["text"], maxsplit=1)
    rep = repair[name, iid]
    reason = J["needs_repair"](parts[-1])
    chk(f"Jetson {name}: text = damage + countermeasure sentence", len(parts) == 2)
    chk(f"Jetson {name}: uncorrected countermeasure passes all checks", rep["triggered"] or reason == "")
    if rep["triggered"]:
        C[f"Jetson {name}: correction triggered", True] += 1
        chk(f"Jetson {name}: kept countermeasure not ranked below the original", J["_REPAIR_SEVERITY"][reason] >= J["_REPAIR_SEVERITY"][rep["reason"]])

width = max(len(k) for k, _ in C)
for k in sorted({k for k, _ in C}):
    yes, no = C[k, True], C[k, False]
    print(f"{k:{width}s}  {yes:6d} / {yes + no:6d}")
