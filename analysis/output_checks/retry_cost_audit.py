#!/usr/bin/env python3
"""Retry/correction branches vs the Table I cost timers - how often each fired, and what the excluded part
could do to Table I.

    python3 -B analysis/output_checks/retry_cost_audit.py

Read-only; no device, network or API access. Inputs:
  May all-tier trace (the Table I cost source): heldout/results/all_tier_reference/*.jsonl, merged by
      profiling/code/collection/merge_tier_outputs.py into heldout/results/collection.jsonl
      and copied by profiling/code/pipeline/02_cost_components.py into
      heldout/results/labels/02_cost_components.jsonl
  M5 PATCH log of that trace: revision/first_submission/run_traces/oracle_patch__pilot_300_4un1.jsonl
  profiling (5,526): profiling/results/raw_per_category/{m5,pi,pi_a_v18,jetson}_*.jsonl
  November live run (route labels; Pi_B and M5_B/C auxiliary costs of Ours; not a main-path cost source):
      deployment/live_run/
Which code lines start and stop each timer is read from the code, not here. This script counts how often each branch
fired. Where a branch has no flag, the count is derived from logged fields (rule printed with the number) or bounded.
The asserts are the self-check: they fail if a derivation rule stops matching the logged data.
"""
import glob
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAY = ROOT / "heldout/results/all_tier_reference"
RUN = ROOT / "deployment/live_run"
PROF = ROOT / "profiling/results/raw_per_category"
P_M5, P_PI, P_JET = 1.2154, 7.5564, 11.199                       # W, merge_tier_outputs.py:52-54
TABLE1 = {"Ours": (39.5, 291, 376.2), "Fixed@Pi": (66.2, 230, 428.7), "Direct@Jetson": (13.4, 230, 130.9)}  # results_table.py
JET_FAIL_CAP_S = 3 * 300            # failed attempts before a success at 384 px: 3 x VLM_TIMEOUT (jetson_inference_pipeline.py:61)
sys.path.insert(0, str(ROOT / "profiling/code/collection"))
import pi_b_inference_pipeline as pib  # noqa: E402  (analyze_quality and the fixed fallback text)


def jl(path, ok_only=True):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not ok_only or r.get("status", "ok") in ("ok", "complete"):
            out.append(r)
    return out


def by_id(rows):
    return {r["image_id"]: r for r in rows if "image_id" in r}  # skips summary lines


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def share(times, flagged, label):
    """Excess of flagged records over the unflagged mean (estimate) and all time on flagged records (upper bound)."""
    base = st.mean(t for t, f in zip(times, flagged) if not f)
    extra = sum(t - base for t, f in zip(times, flagged) if f)
    upper = sum(t for t, f in zip(times, flagged) if f)
    print(f"    {label}: {sum(flagged)}/{len(times)} ({100 * sum(flagged) / len(times):.1f}%); mean {st.mean(t for t, f in zip(times, flagged) if f) / 1000 if any(flagged) else 0:.1f} s"
          f" vs {base / 1000:.1f} s otherwise; share of the tier's time: est {100 * extra / sum(times):.1f}% "
          f"(upper {100 * upper / sum(times):.1f}%)")
    return {i: (t - base if f else 0.0) for i, (t, f) in enumerate(zip(times, flagged))}


def untimed_gaps(rows, timed_ms, label, cap_s=None):
    """Wall time between consecutive records of one sequential process minus the time its stage timer recorded."""
    rows = sorted(rows, key=lambda r: ts(r["ts"]))
    u = sorted(((ts(b["ts"]) - ts(a["ts"])) * 1000 - timed_ms(b)) / 1000 for a, b in zip(rows, rows[1:]))
    big = [round(x, 1) for x in u if x > 5]
    print(f"    {label}: {len(u)} gaps, untimed median {st.median(u):.2f} s, p99 {u[int(.99 * len(u))]:.2f} s; "
          f"{len(big)} gaps > 5 s: {big if len(big) <= 8 else big[:3] + ['...'] + big[-3:]}")
    return u


# ---------------------------------------------------------------- A. Table I inputs are the traced stage timers
held = by_id(jl(ROOT / "heldout/results/labels/02_cost_components.jsonl"))
col = by_id(jl(ROOT / "heldout/results/collection.jsonl"))
raw = jl(MAY / "online_result_m5_raw.jsonl")
m5a = {r["image_id"]: r for r in raw if r["role"] == "a"}
pia = by_id(jl(MAY / "pi_a_online_routing_v2.jsonl"))
pibm = by_id(jl(MAY / "pi_b_online_routing_v2.jsonl"))
jet = by_id(jl(MAY / "online_result_jetson.jsonl"))
patch = {}
for p in jl(ROOT / "revision/first_submission/run_traces/oracle_patch__pilot_300_4un1.jsonl", ok_only=False):
    if p.get("role") == "a":
        patch.setdefault(p["image_id"], []).append(p)
ids = sorted(col)
assert len(ids) == 600 and set(ids) == set(held) == set(m5a) == set(pia) == set(jet)
for i in ids:
    c, s, o = col[i], col[i]["stage_latency_ms"], col[i]["stage_overhead_bytes"]
    for t, e in (("m5_a", "end_m5"), ("pi_a", "end_pi"), ("jetson", "end_jetson")):
        assert all(held[i][t][k] == c[e][k] for k in ("latency_ms", "overhead_bytes", "energy_j"))
    lp, gj = pia[i]["end_pi"]["cost"]["latency_ms"], pia[i]["go_jetson"]
    assert len(patch[i]) == 1
    assert s["m5_total"] == m5a[i]["latency"]["m5_total_ms"]
    assert s["m5_pi_http"] == patch[i][0]["m5_pi_http_e2e_ms"] and o["m5_pi_rx"] == patch[i][0]["m5_pi_rx_bytes"]
    assert o["m5_pi_tx"] == m5a[i]["_pi"]["content_length"]           # Pi-side Content-Length of the stored POST
    assert s["pi_total"] == lp["pi_total_ms"] and s["jetson_total"] == jet[i]["end_jetson"]["vlm_ms"]
    assert (s["pi_jet_http"], o["pi_jet_tx"], o["pi_jet_rx"]) == (gj["pi_jet_http_e2e_ms"], gj["pi_jet_tx_bytes"], gj["pi_jet_rx_bytes"])
    assert not c["_flags"]["pi_jet_handoff_imputed"]
print("A. 600/600: Table I costs (02_cost) = merged end_* = stage sums of M5_A m5_total_ms, the single role-A PATCH "
      "(http ms, rx bytes), the Pi-side Content-Length of M5_A's POST, Pi_A pi_total_ms + go_jetson, Jetson vlm_ms; "
      "0 Pi->Jetson hops imputed")
# firmware provenance: every M5 (B and C too) uploaded the full image to the Pi -> the three-role multipart flow of
# profiling/code/firmware/tier1.cpp; in heldout/code/firmware/tier1.cpp B/C send JSON to M5_A instead
assert all(r["_pi"]["content_length"] > r["image_bytes"] for r in raw)
print(f"   May M5 records: {dict(Counter(r['type'] for r in jl(MAY / 'online_result_m5.jsonl')))}; image uploaded to the Pi "
      f"by {dict(Counter(r['role'] for r in raw))} (role: uploads)")
pcol = by_id(jl(ROOT / "profiling/results/collection.jsonl"))
pcost = by_id(jl(ROOT / "profiling/results/pipeline_outputs/02_cost_components.jsonl"))
assert all(pcost[i][t][k] == pcol[i][e][k] for i in pcost for t, e in (("m5_a", "end_m5"), ("pi_a", "end_pi"),
           ("jetson", "end_jetson")) for k in ("latency_ms", "overhead_bytes", "energy_j"))
print(f"   profiling: 02_cost = merged end_* on {len(pcost)}/{len(pcol)}; Pi->Jetson hops imputed with medians: "
      f"{sum(r['_flags']['pi_jet_handoff_imputed'] for r in pcol.values())}")

# ---------------------------------------------------------------- B. M5 (VLM retry = alternative prompt)
print("\nB. M5 (RETRY_VLM: 2nd attempt with the other prompt variant; salvage re-extraction has no flag)")
for role in "abc":
    rs = [r for r in raw if r["role"] == role]
    # every vlm_query adds a fixed 250 ms pause between UART push and inference (tier1.cpp vlm_query), so
    # roundtrip - uart - infer = 250 ms x attempts: a timing signature of the attempt count
    sig = [round((r["latency"]["m5_vlm_roundtrip_ms"] - r["latency"]["m5_uart_ms"] - r["latency"]["m5_vlm_infer_ms"]) / 250) - 1
           for r in rs]
    assert sig == [r["retry_count"] for r in rs]
    dup = len(rs) - len({r["image_id"] for r in rs})
    print(f"  May M5_{role.upper()}: retry_count {dict(Counter(r['retry_count'] for r in rs))}; m5_recover_ms>0: "
          f"{sum(r['latency']['m5_recover_ms'] > 0 for r in rs)}; duplicate uploads at the Pi: {dup}; "
          f"timing signature = retry_count on {len(rs)}/{len(rs)}")
inner = [m5a[i]["latency"]["m5_total_ms"] - sum(m5a[i]["latency"][k] for k in ("m5_sd_read_ms", "m5_decode_resize_ms",
          "m5_vlm_roundtrip_ms", "m5_postprocess_ms")) - 800 * m5a[i]["retry_count"] for i in ids]  # COOL_BAD_MS = 800
assert max(inner) < 1000   # no precheck recovery (>= 8 s reset) hidden inside any M5_A m5_total_ms
print(f"  May M5_A m5_total - (SD + resize + VLM round trips + postprocess + 800 ms cool-down per retry): "
      f"median {st.median(inner)} ms, max {max(inner)} ms")
m5_t = [m5a[i]["latency"]["m5_total_ms"] for i in ids]
m5_x = share(m5_t, [m5a[i]["retry_count"] > 0 for i in ids], "May M5_A retried images")
m5_x = {i: m5_x[k] for k, i in enumerate(ids)}
prof_m5 = [r for f in sorted(PROF.glob("m5_*.jsonl")) for r in jl(f)]
pl = [r["end_m5"]["cost"]["latency_ms"] for r in prof_m5]
psig = [round((x["m5_vlm_roundtrip_ms"] - x["m5_uart_ms"] - x["m5_vlm_infer_ms"]) / 250) - 1 for x in pl]
print(f"  profiling M5_A (retry_count not stored; inferred from the timing signature): {dict(Counter(psig))} of {len(pl)}")
share([x["m5_total_ms"] for x in pl], [s > 0 for s in psig], "profiling M5_A retried (inferred)")

# ---------------------------------------------------------------- C. Pi
print("\nC. Pi (retry prompt; candidate kept only if strictly better; fixed fallback). No 'chosen' flag in May/profiling;")
print("   rule: backup kept = semantic_penalty - (gate failed); retried = backup kept, or the kept primary fails the")
print("   primary_ok test (gate failed / descriptive_action / news_style with no visual term); ambiguous = only weak_* reasons")


def pia_class(e):
    q, sp = e["quality_gate"], e["conf_detail"]["semantic_penalty"]
    backup = sp - int(not q["passed"])
    assert backup in (0, 1)
    rs = set(q["reasons"])
    if backup:
        return "backup_kept"
    if not q["passed"] or "descriptive_action" in rs or {"news_style_situation", "low_visual_specificity"} <= rs:
        return "retried_primary_kept"
    return "ambiguous" if rs & {"weak_situation", "weak_action"} else "no_retry"


for name, recs in (("May", [pia[i] for i in ids]), ("profiling", [r for f in sorted(PROF.glob("pi_a_v18_*.jsonl")) for r in jl(f)])):
    cl = [pia_class(r["end_pi"]) for r in recs]
    lat = [r["end_pi"]["cost"]["latency_ms"] for r in recs]
    print(f"  {name} Pi_A: {dict(Counter(cl))}; transport retries (pi_total - pi_vlm > 5 s): "
          f"{sum(x['pi_total_ms'] - x['pi_vlm_ms'] > 5000 for x in lat)}")
    x = share([v["pi_total_ms"] for v in lat], [c in ("backup_kept", "retried_primary_kept") for c in cl], f"{name} Pi_A retried")
    if name == "May":
        pi_x = {i: x[k] for k, i in enumerate(ids)}
untimed_gaps(list(pia.values()), lambda r: r["end_pi"]["cost"]["latency_ms"]["pi_total_ms"] + r["go_jetson"]["pi_jet_http_e2e_ms"],
             "May Pi_A untimed between records (periodic model reload every 20 images is outside the timer)")

FB = pib.PI_B_WATER_LANDMARK_FALLBACK


def pib_class(text, sp, passed):
    backup = sp - int(not passed)
    assert backup in (0, 1)
    if text == FB:
        return "fallback"
    if backup:
        return "backup_kept"
    return "retried_other" if (len(text.split()) < 5 or not passed) else "primary_or_short_retry"


cl = []
for i in ids:
    e = pibm[i]["end_pi"]
    assert e["quality_gate"]["passed"] == pib.analyze_quality(e["output"]["text"])["passed"]  # lets profiling recompute it
    cl.append(pib_class(e["output"]["text"], e["conf_detail"]["semantic_penalty"], e["quality_gate"]["passed"]))
print(f"  May Pi_B: {dict(Counter(cl))}")
prof_pi = [r for f in sorted(PROF.glob("pi_[a-z]*.jsonl")) if not f.name.startswith("pi_a_v18") for r in jl(f)]
cl = [pib_class(r["pi_b"]["text"], r["pi_b"]["conf_detail"]["semantic_penalty"], pib.analyze_quality(r["pi_b"]["text"])["passed"])
      for r in prof_pi if r["pi_b"].get("conf_detail")]
print(f"  profiling Pi_B: {dict(Counter(cl))} (no conf_detail: {sum(not r['pi_b'].get('conf_detail') for r in prof_pi)})")
run = by_id(r for r in jl(RUN / "online_routing_final.jsonl") if r.get("type") == "online_routing_final")
pbn = by_id(jl(RUN / "device_logs/pi_b/online_pi_b_results.jsonl"))
esc = [i for i in sorted(run) if run[i].get("end_pi")]
assert all(run[i]["end_pi"]["cost"]["latency_ms"]["pi_b_total_ms"] == pbn[i]["pi_b_cost"]["latency_ms"]["pi_b_total_ms"] for i in esc)
dbg = [pbn[i]["pi_b_debug"] for i in esc]
print(f"  Nov Pi_B (feeds Ours' auxiliary Pi_B energy): chosen {dict(Counter(d['chosen'] for d in dbg))}; retry prompt run "
      f"{sum('backup_raw' in d for d in dbg)}, short-fragment retry run {sum('short_retry_raw' in d for d in dbg)}; "
      f"transport retries {sum(pbn[i]['pi_b_cost']['latency_ms']['pi_b_total_ms'] - pbn[i]['pi_b_cost']['latency_ms']['pi_vlm_ms'] > 5000 for i in esc)}")
pb_t = [pbn[i]["pi_b_cost"]["latency_ms"]["pi_b_total_ms"] for i in esc]
pb_x = share(pb_t, ["backup_raw" in d or "short_retry_raw" in d for d in dbg], "Nov Pi_B retried")
b2a = st.mean(pbn[i]["pi_b_to_pi_a"]["pi_b_pi_a_tx_bytes"] + pbn[i]["pi_b_to_pi_a"]["pi_b_pi_a_rx_bytes"] for i in esc)
print(f"    Pi_B -> Pi_A result message (not in Table I bytes): {b2a:.0f} B per escalated image")

# ---------------------------------------------------------------- D. Jetson
print("\nD. Jetson (one corrective call, max 80 tokens; resize ladder 768->640->512->384 has no flag in the record)")
for name, recs in (("May", [jet[i] for i in ids]), ("profiling", [r for f in sorted(PROF.glob("jetson_*.jsonl")) for r in jl(f)])):
    rep = [r["end_jetson"]["repair"] for r in recs]
    print(f"  {name}: repair {sum(x['triggered'] for x in rep)}/{len(rep)}, reasons {dict(Counter(x['reason'] for x in rep if x['triggered']))}")
    x = share([r["end_jetson"]["vlm_ms"] for r in recs], [x["triggered"] for x in rep], f"{name} Jetson repaired")
    if name == "May":
        jet_x = {i: x[k] for k, i in enumerate(ids)}
        u = untimed_gaps(recs, lambda r: r["end_jetson"]["vlm_ms"], "May Jetson untimed between records")
    else:
        u = []
        for f in sorted(PROF.glob("jetson_*.jsonl")):
            u += untimed_gaps(jl(f), lambda r: r["end_jetson"]["vlm_ms"], f"profiling {f.stem} untimed")
    hid = sum(min(max(v, 0), JET_FAIL_CAP_S) for v in u) / len(recs)
    print(f"    bound on untimed Jetson work (each gap capped at {JET_FAIL_CAP_S} s = 3 timed-out attempts): "
          f"{hid:.2f} s = {hid * P_JET:.1f} J per Jetson request")
    if name == "May":
        jet_hidden_s = hid

# ---------------------------------------------------------------- E. retry/correction time already inside Table I
lab = {i: run[i]["route_label"] for i in ids}
rows = {"Ours": lambda i: (m5_x[i] / 1000 * P_M5 + (pi_x[i] / 1000 * P_PI if lab[i] != "m5_a" else 0)
                           + (jet_x[i] / 1000 * P_JET if lab[i] == "jetson" else 0),
                           m5_x[i] + (pi_x[i] if lab[i] != "m5_a" else 0) + (jet_x[i] if lab[i] == "jetson" else 0)),
        "Fixed@Pi": lambda i: (m5_x[i] / 1000 * P_M5 + pi_x[i] / 1000 * P_PI, m5_x[i] + pi_x[i]),
        "Direct@Jetson": lambda i: (jet_x[i] / 1000 * P_JET, jet_x[i])}
pb_extra_j = sum(pb_x.values()) / len(pb_t) / 1000 * P_PI * len(esc) / 600
print("\nE. estimated retry/correction time already inside Table I (main path; Ours also its Pi_B retries):")
for k, f in rows.items():
    e = sum(f(i)[0] for i in ids) / 600 + (pb_extra_j if k == "Ours" else 0)
    lt = sum(f(i)[1] for i in ids) / 600 / 1000
    print(f"  {k:14s} {lt:.2f} s of {TABLE1[k][0]} s, {e:.1f} J of {TABLE1[k][2]} J")

# ---------------------------------------------------------------- F. November run: what was re-run (excluded)
print("\nF. November live run (excluded from Table I costs)")
lists = [[x.strip() for x in open(RUN / f) if x.strip()]
         for f in ("retry_ids.txt", "retry2_ids.txt", "retry3_ids.txt", "retry4_ids.txt", "retry5_ids.txt")]
print(f"  image-level M5 re-run rounds {[len(v) for v in lists]} -> {sum(map(len, lists))} re-runs of "
      f"{len({x for v in lists for x in v})} images")
every = Counter()                                 # all rounds, all three M5s (retry folders start with the id list)
for f in glob.glob(str(RUN / "device_logs/m5*_sd*/failed_ids.jsonl")):
    every.update(r.get("reason") for r in jl(f, ok_only=False) if r.get("reason"))
for dev in "abc":
    reasons = Counter(r.get("reason") for r in jl(RUN / f"device_logs/m5{dev}_sd/failed_ids.jsonl", ok_only=False) if r.get("reason"))
    print(f"    M5_{dev.upper()} main-round failure log: {dict(reasons.most_common(6))}")
print(f"    all rounds, all M5s: {dict(every.most_common(7))}")
wd = {f: Counter(json.loads(x).get("kind") for x in open(RUN / "device_logs/watchdog" / f) if x.strip())
      for f in ("watchdog_medic_events_a.jsonl", "watchdog_medic_events_bc.jsonl")}
print(f"    watchdog power cycles: M5_A {wd['watchdog_medic_events_a.jsonl']['cycle']}, M5_B/C {wd['watchdog_medic_events_bc.jsonl']['cycle']}")
res = {i for i in ids if run[i].get("jetson_answer_source") == "resent_after_run"}
log = jl(RUN / "jetson_resend_log.jsonl")
srv = jl(RUN / "device_logs/jetson/online_jetson_results.jsonl", ok_only=False)
pre = [r for r in srv if r["image_id"] in res and r["ts"] < "2026-11-04T18:50"]
pre_ok = [r["end_jetson"]["cost"]["latency_ms"]["jetson_wall_ms"] / 1000 for r in pre if r["status"] == "ok"]
print(f"  Jetson-routed {sum(lab[i] == 'jetson' for i in ids)}; answer re-sent after the run {len(res)} "
      f"(resend log {len(log)} lines, {sum(x['attempts'] == 1 and x['ok'] for x in log)} ok at the 1st attempt, "
      f"mean {st.mean(x['seconds'] for x in log):.1f} s; missing from log: {sorted(res - {x['image_id'] for x in log})})")
print(f"    before the re-send the Jetson server logged {len(pre)} records for {len({r['image_id'] for r in pre})} of them: "
      f"{len(pre_ok)} finished after Pi_A's 60 s read timeout (mean {st.mean(pre_ok):.0f} s, max {max(pre_ok):.0f} s, "
      f"sum {sum(pre_ok):.0f} s); failed {dict(Counter(r.get('error') for r in pre if r['status'] != 'ok'))} (no timing)")

# ---------------------------------------------------------------- G. bounds (labelled as bounds, not measurements)
print("\nG. BOUNDS on Table I (latency s / KiB / J)")
L, K, J = TABLE1["Ours"]
h = jet_hidden_s * len(ids) / 600            # worst case: all untimed Jetson work sits on Ours' 140 Jetson images
print(f"  G1 untimed Jetson work, worst case all on Ours: Ours <= {L + h:.1f} s / {J + h * P_JET:.1f} J; "
      f"Direct@Jetson <= {TABLE1['Direct@Jetson'][0] + jet_hidden_s:.1f} s / {TABLE1['Direct@Jetson'][2] + jet_hidden_s * P_JET:.1f} J")
aux_kib = (2 * st.mean(len(json.dumps({k: v for k, v in r.items() if k != "_pi"})) + 200 for r in raw if r["role"] != "a")
           + b2a * len(esc) / 600) / 1024
print(f"  G2 auxiliary result messages (M5_B/C->M5_A est. from May record size + 200 B header; Pi_B->Pi_A logged): "
      f"Ours <= {K + aux_kib:.0f} KiB (+{aux_kib:.1f})")
n = len(res)
rate = n / sum(lab[i] == "jetson" for i in ids)
print(f"  G3 Nov fault charged to Ours ({n} timed-out Jetson calls): +60 s wait each -> {L + n * 60 / 600:.1f} s; "
      f"Jetson busy 60 s each -> {J + n * 60 * P_JET / 600:.1f} J; logged abandoned Jetson work only ({sum(pre_ok):.0f} s, "
      f"a lower bound of the waste) -> {J + sum(pre_ok) * P_JET / 600:.1f} J (vs Fixed@Pi {TABLE1['Fixed@Pi'][2]} J, no Jetson)")
dl, _, dj = TABLE1["Direct@Jetson"]
print(f"      same fault rate ({100 * rate:.0f}% of Jetson calls, 60 s each) on Direct@Jetson: {dl + rate * 60:.1f} s, "
      f"{dj + rate * 60 * P_JET:.1f} J")
aux = defaultdict(list)                          # M5_B/C times as results_table.py reads them
for f in glob.glob(str(RUN / "device_logs/m5a_sd*/m5a_aux_cache__*.jsonl")) + glob.glob(str(RUN / "device_logs/m5a_sd*/*.extracted.jsonl")):
    for d in jl(f, ok_only=False):
        ms = d.get("m5_total_ms") or (d.get("latency") or {}).get("m5_total_ms") or 0
        if d.get("event") == "store" and d.get("role") in ("b", "c") and ms > 0:
            aux[d["role"]].append(ms)
trio_j = st.mean(held[i]["m5_a"]["energy_j"] for i in ids) + P_M5 * sum(st.mean(v) for v in aux.values()) / 1000
print(f"  G4 Nov M5 answers lost to power cuts (stale_missing_part, {every['stale_missing_part']} entries), each at most "
      f"one M5 trio's work ({trio_j:.1f} J): Ours +{every['stale_missing_part'] * trio_j / 600:.1f} J; "
      f"VLM timeouts ({every['timeout_empty']} entries, <= 50 s wait each at {P_M5} W) hit any method running the "
      f"M5 VLM: +{every['timeout_empty'] * 50 * P_M5 / 600:.1f} J (durations of recoveries/reboots not logged)")
