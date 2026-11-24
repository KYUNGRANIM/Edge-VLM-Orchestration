#!/usr/bin/env python3
"""How different are the answers of agents running the same model, and is the difference informative?

    python3 analysis/router_analysis/consistency_stats.py

Agents of one tier share the model but not the prompt (role prompts: M5_A damage + action, M5_B structural
damage, M5_C harm to people; Pi_A situation + action, Pi_B damage description). For the profiling set (5,526,
all tiers run on every image) and the MEDIC rerun (600, the run's own answers) this prints how often the agents
agree on category and severity, how correct the main agent is at each agreement level, how often the reference
ends at that tier, and how much of each deployed tree's splitting relies on the consistency features.
"""
import contextlib
import io
import json
import runpy
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(str(HERE.parent / "main_results/results_table.py"))
M5VLM = G["M5VLM"]
from common import safe_int  # noqa: E402

P = M5VLM / "profiling/results/pipeline_outputs"
q_prof = {r["image_id"]: r for r in map(json.loads, open(P / "01_quality_components.jsonl")) if r.get("type") == "quality_components"}
o_prof = {r["image_id"]: r for r in map(json.loads, open(P / "03_oracle_labels.jsonl")) if r.get("type") == "oracle_label"}
prof = G["prof"]


def m5_level(rec):
    cats = [rec[k].get("cat", "") for k in ("m5_a", "m5_b", "m5_c")]
    n = Counter(cats).most_common(1)[0][1]
    return {3: "3/3 agree", 2: "2/3 agree", 1: "all differ"}[n], cats[0] == Counter(cats).most_common(1)[0][0] or n == 1


def m5_sev_level(rec):
    sevs = [safe_int(rec[k].get("sev")) for k in ("m5_a", "m5_b", "m5_c")]
    return {3: "3/3 agree", 2: "2/3 agree", 1: "all differ"}[Counter(sevs).most_common(1)[0][1]]


def table(title, items):
    """items: (level, main-agent category correct, main-agent pass, reference ends at this tier)"""
    by = defaultdict(list)
    for lv, *v in items:
        by[lv].append(v)
    n = len(items)
    print(f"  {title}")
    for lv in sorted(by, key=lambda x: (x.startswith("all") or x.startswith("differ"), x)):
        a = np.array(by[lv], dtype=float)
        print(f"    {lv:11s} {len(a):5d} ({100 * len(a) / n:4.1f}%) | main agent category correct {100 * a[:, 0].mean():5.1f}% "
              f"| main answer passes Q-Sat {100 * a[:, 1].mean():5.1f}% | reference ends here {100 * a[:, 2].mean():5.1f}%")


# ---------------- profiling (training data)
ids = sorted(set(prof) & set(q_prof) & set(o_prof))
print(f"PROFILING {len(ids)} images")
table("M5 category agreement among M5_A/B/C (main = M5_A; 'ends here' = reference is M5)",
      [(m5_level(prof[i])[0], q_prof[i]["m5_a"]["cat_score"] == 1, o_prof[i]["quality"]["m5_a"] >= 0.65,
        o_prof[i]["oracle_label"] == "m5_a") for i in ids])
table("M5 severity agreement", [(m5_sev_level(prof[i]), q_prof[i]["m5_a"]["sev_score"] == 1, o_prof[i]["quality"]["m5_a"] >= 0.65,
                                o_prof[i]["oracle_label"] == "m5_a") for i in ids])
esc = [i for i in ids if o_prof[i]["oracle_label"] != "m5_a"]
table("Pi_A vs Pi_B category, images whose reference is Pi or Jetson (main = Pi_A; 'ends here' = reference is Pi)",
      [("match" if prof[i]["pi_a"].get("cat") == prof[i]["pi_b"].get("cat") else "differ", q_prof[i]["pi_a"]["cat_score"] == 1,
        o_prof[i]["quality"]["pi_a"] >= 0.65, o_prof[i]["oracle_label"] == "pi_a") for i in esc])

# ---------------- MEDIC rerun (the answers the devices actually produced)
run, q_run, ref = G["run"], G["q_run"], G["ref_run"]
rid = G["ids"]
print(f"\nMEDIC RERUN {len(rid)} images (run answers; reference rebuilt on them)")
table("M5 category agreement", [(m5_level(run[i]["m5"])[0], q_run[i]["m5_a"]["cat_score"] == 1, G["Q"]["run"][i]["m5_a"] >= 0.65,
                                 r == "m5_a") for i, r in zip(rid, ref)])
esc = [(i, r) for i, r in zip(rid, ref) if (run[i].get("pi") or {}).get("pi_b")]
table(f"Pi_A vs Pi_B category, the {len(esc)} images the run escalated",
      [("match" if run[i]["pi"]["pi_a"].get("cat") == run[i]["pi"]["pi_b"].get("cat") else "differ", q_run[i]["pi_a"]["cat_score"] == 1,
        G["Q"]["run"][i]["pi_a"] >= 0.65, r == "pi_a") for i, r in esc])

# ---------------- how much the deployed trees split on consistency features
CONS = {"m5_cat_agree", "m5_sev_agree", "m5_sev_range", "m5_conf_mean", "m5_conf_min", "pi_cat_match", "pi_sev_match",
        "pi_conf_mean", "pi_conf_min"}
CROSS = {"m5a_pia_cat_match", "m5a_pia_sev_diff"}
for f in ("stage1_m5_router.joblib", "stage2_pi_router.joblib"):
    p = joblib.load(M5VLM / "deployment/routers" / f)
    pipe = p["model"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = defaultdict(float)
    for nm, v in zip(names, pipe.named_steps["clf"].feature_importances_):
        # the trees were fitted on plain lists, so columns are named x<index into feature_keys>[_<category>]
        imp[p["feature_keys"][int(nm.split("__", 1)[1][1:].split("_", 1)[0])]] += v
    share = lambda s: 100 * sum(v for k, v in imp.items() if k in s)  # noqa: E731
    top = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(imp.items(), key=lambda kv: -kv[1]) if v > 0.005)
    print(f"\n{f}: splitting importance on features using auxiliary answers {share(CONS):.0f}%, cross-tier {share(CROSS):.0f}%, "
          f"main-agent only {100 - share(CONS) - share(CROSS):.0f}%\n    {top}")
