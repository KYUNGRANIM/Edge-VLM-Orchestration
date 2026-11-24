#!/usr/bin/env python3
"""What changes if the M5 parser mapped the word "mild" to severity 1, as the corrected MEDIC labels do.

    /opt/anaconda3/bin/python3 -B analysis/output_checks/severity_mild_remap.py

Read-only; needs the pinned scikit-learn 1.5.1 (the deployed routers are joblib files). Reuses instead of re-implementing:
the verified M5 parser port (verify_response_rules.py: has, m5_labels), the Table I accounting (results_table.py, run
in-process with its own output suppressed), the router helpers (export_device_router.py) and sev_soft_score
(01_quality_components.py). Alternative parse = tier1.cpp:1028-1033 with "mild" moved out of the sev-0 word list into
its own branch right before it (sev 1), so the more severe word still wins: "mild ... minor" -> 1, "mild ... severe" -> 2.
  Q1  M5 responses containing "mild" (lowercase substring of the stored text = the parser input), changed severities
  Q2  "mild" in the run's Pi and Jetson responses
  Q3  router features that change on the 600 final-run images
  Q4  deployed routers on the altered features: changed decisions, tier split, Table I costs
  Q5  M5 severity re-scored with the alternative parse: Fixed@M5, Ours, Quality-Cost Reference
"""
import ast
import contextlib
import io
import json
import re
import runpy
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
M5VLM = Path(__file__).resolve().parents[2]
PIPE = M5VLM / "analysis"
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]  # noqa: E731

# ---------------- M5 parser: the verified port and the alternative ----------------
src = (PIPE / "output_checks/verify_response_rules.py").read_text()
fn = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
OLD, NEW = 'elif has(low, "mild", ', 'elif "mild" in low: sev = 1\n    elif has(low, '
assert fn["m5_labels"].count(OLD) == 1
M5P = {}
exec("\n".join([fn["has"], fn["m5_labels"], fn["m5_labels"].replace("def m5_labels", "def m5_labels_alt").replace(OLD, NEW)]), M5P)
has, m5_labels, m5_labels_alt = M5P["has"], M5P["m5_labels"], M5P["m5_labels_alt"]
for t, old, new in (("Mild damage.", 0, 1), ("Mild and minor damage.", 0, 1), ("Minor damage.", 0, 0),
                    ("Mild here, severe there.", 2, 2), ("Moderate to mild damage.", 1, 1)):
    assert (m5_labels(t)[1], m5_labels_alt(t)[1]) == (old, new), t
LOW0 = ("low severity", "minor", "low damage", "minimal damage", "little or no", "little to no")  # tier1.cpp:1030-1033 minus "mild"
KW = ("severe", "not severe", "no severe", "not a severe", "moderate", "mild") + LOW0

# ---------------- Table I accounting, router helpers, quality score ----------------
with contextlib.redirect_stdout(io.StringIO()):
    T = runpy.run_path(str(PIPE / "main_results/results_table.py"))
E = runpy.run_path(str(M5VLM / "deployment" / "export_device_router.py"), run_name="reuse")
sev_soft_score = runpy.run_path(str(M5VLM / "profiling/code/pipeline/01_quality_components.py"), run_name="reuse")["sev_soft_score"]
run, q_run, ids, Q, cost, alltier, TIERS = (T[k] for k in ("run", "q_run", "ids", "Q", "cost", "alltier", "TIERS"))
QP = {r["image_id"]: r for r in load(M5VLM / "profiling/results/pipeline_outputs/01_quality_components.jsonl")
      if r.get("type") == "quality_components"}
GT = {i: r["gt"]["sev"] for i, r in list(QP.items()) + list(q_run.items())}
PROF = load(M5VLM / "profiling/results/collection.jsonl")
scored_ok = lambda s, x, g: s["text"] == x["text"] and s["pred_sev"] == x["sev"] and s["sev_score"] == sev_soft_score(x["sev"], g)  # noqa: E731
print("M5_A parsed severity is the scored one (text, pred_sev, sev_score of 01_quality*.jsonl): "
      f"profiling {sum(scored_ok(QP[r['image_id']]['m5_a'], r['m5_a'], GT[r['image_id']]) for r in PROF)}/{len(PROF)}, "
      f"final run {sum(scored_ok(q_run[i]['m5_a'], run[i]['m5']['m5_a'], GT[i]) for i in ids)}/{len(ids)}")

# ---------------- Q1 ----------------
print("\nQ1  M5 responses: 'mild' and changed severities (counts by ground-truth severity 0/1/2), firmware keyword counts")
SETS = (("profiling", PROF, lambda r: r),
        ("earlier all-tier held-out", load(M5VLM / "heldout/results/collection.jsonl"), lambda r: r),
        ("final run", [run[i] for i in ids], lambda r: r["m5"]))
for name, rows, m5 in SETS:
    for ag in ("m5_a", "m5_b", "m5_c"):
        c, kw = Counter(), Counter()
        for r in rows:
            x, g = m5(r)[ag], GT[r["image_id"]]
            low, new = x["text"].lower(), m5_labels_alt(x["text"])[1]
            assert m5_labels(x["text"]) == (x["cat"], x["sev"])  # the stored labels are the parse of the stored text
            assert (new != x["sev"]) == ("mild" in low and x["sev"] == 0)  # only the sev-0 word branch can change
            kw.update(k for k in KW if k in low)
            c["mild", g] += "mild" in low
            c["changed", g] += new != x["sev"]
            c["changed, no other sev-0 word", g] += new != x["sev"] and not has(low, *LOW0)
        cell = lambda k: f"{sum(c[k, g] for g in range(3))} ({c[k, 0]}/{c[k, 1]}/{c[k, 2]})"  # noqa: E731
        print(f"  {name:26s} {ag} n={len(rows)}: mild {cell('mild')}, changed {cell('changed')}, "
              f"changed with 'mild' the only sev-0 word {cell('changed, no other sev-0 word')} | keywords {dict(kw)}")

# ---------------- Q2 ----------------
print("\nQ2  'mild' in the run's Pi and Jetson responses")
for ag in ("pi_a", "pi_b"):
    sev = runpy.run_path(str(M5VLM / f"deployment/servers/{ag}_inference_engine.py"), run_name="reuse")["classify_sev"]
    outs = [run[i]["pi"][ag] for i in ids if run[i].get("pi")]
    hits = [o for o in outs if "mild" in o["text"].lower()]
    assert all(sev(o["text"]) == o["sev"] for o in outs)
    print(f"  {ag}: {len(hits)}/{len(outs)} contain 'mild', their sev {dict(Counter(o['sev'] for o in hits))}, "
          f"same sev with 'mild' deleted {sum(sev(re.sub('mild', '', o['text'], flags=re.I)) == o['sev'] for o in hits)}/{len(hits)}; "
          f"classify_sev('The damage is mild.') = {sev('The damage is mild.')}")
jet = [run[i]["jetson"]["jetson"]["text"] for i in ids if run[i].get("jetson")]
print(f"  jetson: {sum('mild' in t.lower() for t in jet)}/{len(jet)} contain 'mild' (sev is the digit on the model's 'sev:' line)")

# ---------------- Q3 ----------------
M5_RUN = {i: {ag: run[i]["m5"][ag] for ag in ("m5_a", "m5_b", "m5_c")} for i in ids}
M5_ALT = {i: {ag: {**x, "sev": m5_labels_alt(x["text"])[1]} for ag, x in M5_RUN[i].items()} for i in ids}


def pi_part(i):  # the run's Pi answers; the all-tier ones for an image the run did not send to the Pi
    s = run[i] if run[i].get("pi") else alltier[i]
    return {k: s[k] for k in ("pi_a", "pi_b", "pi_a_conf_detail")}


s1f = lambda m5s, i: T["stage1_features"](m5s[i])  # noqa: E731
s2f = lambda m5s, i: T["stage2_features"]({**m5s[i], **pi_part(i)})  # noqa: E731
pi_ids = [i for i in ids if run[i].get("pi")]
print("\nQ3  features that change under the alternative parse")
for k in ("m5a_sev", "m5_sev_range", "m5_sev_agree"):
    print(f"  Stage 1 {k}: {sum(s1f(M5_RUN, i)[k] != s1f(M5_ALT, i)[k] for i in ids)}/{len(ids)}")
print(f"  Stage 1, any feature: {sum(s1f(M5_RUN, i) != s1f(M5_ALT, i) for i in ids)}/{len(ids)}")
print(f"  Stage 2 m5a_pia_sev_diff: {sum(s2f(M5_RUN, i)['m5a_pia_sev_diff'] != s2f(M5_ALT, i)['m5a_pia_sev_diff'] for i in pi_ids)}"
      f"/{len(pi_ids)} Pi-forwarded")

# ---------------- Q4 ----------------
(P1, h1), (P2, h2) = (E["load_policy"](M5VLM / f"deployment/routers/{s}.joblib") for s in ("stage1_m5_router", "stage2_pi_router"))


def route(m5s):
    s1 = dict(zip(ids, E["joblib_end"](P1, [s1f(m5s, i) for i in ids])))
    esc = [i for i in ids if not s1[i]]
    s2 = dict(zip(esc, E["joblib_end"](P2, [s2f(m5s, i) for i in esc])))
    return s1, s2, ["m5_a" if s1[i] else "pi_a" if s2[i] else "jetson" for i in ids]


ours = [run[i]["route_label"] for i in ids]
(s1o, s2o, ro), (s1a, s2a, ra) = route(M5_RUN), route(M5_ALT)
print(f"\nQ4  deployed routers {h1}, {h2}")
print(f"  original features reproduce route_label: {sum(a == b for a, b in zip(ro, ours))}/{len(ids)}; Stage 1 decision logged by "
      f"M5_A: {sum(s1o[i] == (run[i]['stage1']['decision'] == 'END@M5') for i in ids)}/{len(ids)}; Stage 2 features logged by "
      f"Pi_A = recomputed: {sum(E['feats_equal'](s2f(M5_RUN, i), run[i]['stage2']['features']) for i in pi_ids)}/{len(pi_ids)}")
assert ro == ours
for name, P in (("Stage 1", P1), ("Stage 2", P2)):
    print(f"  {name} tree decides on {sorted(E['used_keys'](E['rules'](P)))}")
both = [i for i in s2o if i in s2a]
print(f"  changed: Stage 1 {sum(s1o[i] != s1a[i] for i in ids)}/{len(ids)}, Stage 2 {sum(s2o[i] != s2a[i] for i in both)}/{len(both)} "
      f"(+{len(set(s2a) - set(s2o))} new, -{len(set(s2o) - set(s2a))} dropped at Stage 2), routes {sum(a != b for a, b in zip(ro, ra))}")
print(T["hdr"])
T["row"]("Ours, deployed", ours, "run", aux=True)
T["row"]("Ours, routes from mild->1 features", ra, "run", aux=True)

# ---------------- Q5 ----------------
ALT = "run, mild->1"
Q[ALT] = {i: {**Q["run"][i], "m5_a": T["q"]({**q_run[i]["m5_a"], "sev_score": sev_soft_score(M5_ALT[i]["m5_a"]["sev"], GT[i])})}
          for i in ids}


def reference(qs):  # results_table.py:173-176
    return [min((t for t in TIERS if max(qs[i].values()) - qs[i][t] <= T["TAU"] + 1e-12), key=lambda t: cost[i][t]["energy_j"])
            for i in ids]


assert reference(Q["run"]) == T["ref_run"]
print(f"\nQ5  M5_A severity re-scored with the alternative parse ([{ALT}] answer set)\n{T['hdr']}")
for qs in ("run", ALT):
    T["row"](f"Fixed@M5 [{qs}]", ["m5_a"] * len(ids), qs)
    T["row"](f"Ours, deployed routes [{qs}]", ours, qs, aux=True)
    T["row"](f"Quality-Cost Ref. rebuilt [{qs}]", reference(Q[qs]), qs)
T["row"](f"Ours, mild->1 routes [{ALT}]", ra, ALT, aux=True)
flips = lambda ta, tb: sum((Q["run"][i][a] >= T["Q_TH"]) != (Q[ALT][i][b] >= T["Q_TH"]) for i, a, b in zip(ids, ta, tb))  # noqa: E731
print(f"  images whose Q-Sat status changes: M5 tier {flips(['m5_a'] * len(ids), ['m5_a'] * len(ids))}, Ours deployed {flips(ours, ours)}, "
      f"Ours mild->1 routes {flips(ours, ra)}, Reference {flips(reference(Q['run']), reference(Q[ALT]))}; "
      f"M5_A sev_score changed on {sum(Q[ALT][i]['m5_a'] != Q['run'][i]['m5_a'] for i in ids)} images")
