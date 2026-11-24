#!/usr/bin/env python3
"""Prompt order: which M5 prompt (primary or backup) produced each M5 answer set.

    /opt/anaconda3/bin/python3 analysis/output_checks/prompt_order_provenance.py

Read-only; prints only. Sections:
  1. Firmware: the first-attempt rule and RETRY_VLM in the collection, old online-routing and runtime firmware, and
     whether the retry loop and the text/quality/category/severity/confidence code are the same text in all three.
  2. Profiling (5,526): the first-prompt rule reconstructed as the per-category sorted index (odd = backup first),
     checked against the order of the device timestamps; images re-collected in a later pass are flagged (their index
     in that pass is unknown).
  3. May all-tier held-out (600): same check; retry_count is logged there, so the adopted prompt is known per answer.
  4. Final run (600): which log fields exist; a timing estimate of two-attempt answers, calibrated on May.
     Profiling: the same timing estimate (no retry_count is stored).
  5. Wording markers calibrated on May (adopted prompt known): estimated backup share per answer set, and a windowed
     check that the even/odd rule holds along every profiling category.
profiling_first_prompt() is imported by prompt_order_router.py.
"""
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

M5VLM = Path(__file__).resolve().parents[2]
RAW = M5VLM / "profiling/results/raw_per_category"
MAY_RAW = M5VLM / "heldout/results/all_tier_reference/online_result_m5_raw.jsonl"
MAY_ALLTIER = M5VLM / "heldout/results/collection.jsonl"
HELD_Q = M5VLM / "heldout/results/labels/01_quality_components.jsonl"
RUN = M5VLM / "deployment/live_run"
FIRMWARE = {"collection": "profiling/code/firmware", "old online routing": "heldout/code/firmware",
            "runtime (final run)": "deployment/m5_firmware"}
SAME_CODE = ("vlm_query", "parse_structured_output", "extract_unlabeled_text_block", "postprocess_text_output",
             "sanitize_a_final", "ensure_a_two_sentences", "sanitize_b_text", "sanitize_c_text", "text_quality_ok",
             "is_garbage_text", "has_non_ascii", "is_disaster_relevant", "salvage_vlm_text", "classify_from_text",
             "compute_fine_score", "score_to_conf_quintile", "compute_conf_from_text", "resize_jpeg_for_vlm")
# Wording markers (word present in the answer), chosen on the May answers whose adopted prompt is known.
MARKERS = {"a": ({"urgent"}, {"include"}), "b": ({"collapse"}, {"blocked", "blocking", "partially"}),
           "c": ({"could"}, {"risk", "scene"})}
# May calibration of the timing rules (Section 4 prints the separation they rest on)
A_TOTAL_MS, A_POST_MS, A_INFER_MS = 15000, 32, 11500


def jl(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # SD logs can hold a line cut by a power cut
            pass
    return out


def late_ids(rows, rank, ts):
    """Images processed after an image with a higher index: re-collected in a later pass."""
    late, seen = set(), -1
    for r in sorted(rows, key=ts):
        k = rank[r["image_id"]]
        if k < seen:
            late.add(r["image_id"])
        seen = max(seen, k)
    return late


def ascending_pairs(rows, rank, ts):
    ks = [rank[r["image_id"]] for r in sorted(rows, key=ts)]
    return sum(b > a for a, b in zip(ks, ks[1:])), len(ks) - 1


def profiling_index():
    """image_id -> (category, index in the category's sorted file names, re-collected later)."""
    out = {}
    for f in sorted(RAW.glob("m5_*.jsonl")):
        rows = jl(f)
        rank = {i: k for k, i in enumerate(sorted(r["image_id"] for r in rows))}
        late = set().union(*(late_ids(rows, rank, lambda r, d=d: r["device_ts"][d]) for d in "abc"))
        out.update({i: (f.stem[3:], k, i in late) for i, k in rank.items()})
    return out


def profiling_first_prompt():
    """image_id -> 'primary' | 'backup' (first attempt), None if re-collected in a later pass."""
    return {i: None if late else ("backup" if k % 2 else "primary") for i, (_, k, late) in profiling_index().items()}


def may_records():
    """May all-tier held-out: index, re-collected ids, and per agent the raw record of the answer Table I uses."""
    gt = {r["image_id"]: r["gt"]["cat"] for r in jl(HELD_Q) if r.get("type") == "quality_components"}
    rank = {i: k for k, i in enumerate(sorted(gt, key=lambda i: (gt[i], i)))}  # /images_sampled/<category>/<file>
    raw = jl(MAY_RAW)
    alltier = {r["image_id"]: r for r in jl(MAY_ALLTIER)}
    late = set().union(*(late_ids([r for r in raw if r["device"] == d], rank, lambda r: r["ts"])
                         for d in ("M5_A", "M5_B", "M5_C")))
    rec = {}
    for r in raw:
        ag = r["device"][-1].lower()
        if r["text"] == alltier[r["image_id"]]["m5_" + ag]["text"]:
            rec[(ag, r["image_id"])] = r
    return rank, late, raw, rec, alltier


def adopted(first, retry_count):
    return first if retry_count == 0 else {"primary": "backup", "backup": "primary"}[first]


def marker(text, ag):
    w = set(re.findall(r"[a-z]+", text.lower()))
    p, b = MARKERS[ag]
    return int(bool(w & p)) - int(bool(w & b))


def funcs(src):
    out = {}
    for m in re.finditer(r"^\s*(?:static\s+)?[\w:<>\*&\s]+?\b(\w+)\s*\([^;{]*\)\s*\{", src, re.M):
        depth = 0
        for j in range(m.end() - 1, len(src)):
            depth += {"{": 1, "}": -1}.get(src[j], 0)
            if depth == 0:
                break
        out.setdefault(m.group(1), re.sub(r"\s+", " ", src[m.start():j + 1]))
    return out


def main():
    print("1. FIRMWARE")
    loops, fns = {}, {}
    for name, d in FIRMWARE.items():
        lines = (M5VLM / d / "tier1.cpp").read_text(encoding="utf-8", errors="replace").splitlines()
        cfg = (M5VLM / d / "config.h").read_text(encoding="utf-8", errors="replace").splitlines()
        flip = [(n + 1, s.strip()) for n, s in enumerate(lines) if "flip_prompt_order =" in s]
        retry = [(n + 1, s.strip()) for n, s in enumerate(cfg) if "RETRY_VLM =" in s]
        a = next(n for n, s in enumerate(lines) if "int infer_attempts = 0;" in s)
        b = next(n for n in range(a, len(lines)) if lines[n].strip() == "if (!vlm_ok) {")
        loops[name] = "\n".join(lines[a:b])
        fns[name] = funcs("\n".join(lines))
        sha = hashlib.sha256((M5VLM / d / "prompts.h").read_bytes()).hexdigest()[:12]
        print(f"   {name:20s} {d}/tier1.cpp:{flip[0][0]} {flip[0][1]}  | config.h:{retry[0][0]} {retry[0][1]}  | "
              f"retry loop tier1.cpp:{a + 1}-{b} | prompts.h sha256 {sha}")
    ref = next(iter(FIRMWARE))
    print(f"   retry loop text identical in all three: {len(set(loops.values())) == 1}; text/quality/cat/sev/conf "
          f"functions identical: {all(len({fns[k][f] for k in fns}) == 1 for f in SAME_CODE)} ({len(SAME_CODE)} functions)")

    print("\n2. PROFILING: per-category sorted index vs device-timestamp order (adjacent pairs in ascending index)")
    idx = profiling_index()
    for f in sorted(RAW.glob("m5_*.jsonl")):
        rows = jl(f)
        rank = {i: k for k, i in enumerate(sorted(r["image_id"] for r in rows))}
        asc = [ascending_pairs(rows, rank, lambda r, d=d: r["device_ts"][d]) for d in "abc"]
        late = sorted(rank[i] for i in rank if idx[i][2])
        print(f"   {f.stem[3:]:15s} n={len(rows)}  A {asc[0][0]}/{asc[0][1]}  B {asc[1][0]}/{asc[1][1]}  "
              f"C {asc[2][0]}/{asc[2][1]}  re-collected later (index in category): {late}")
    fp = profiling_first_prompt()
    print(f"   first prompt: {dict(Counter(fp.values()))}  (None = re-collected, excluded from the split)")
    print(f"   one batch over all 5,526 instead: {sum(k % 2 for k in range(len(idx)))} backup-first")

    print("\n3. MAY ALL-TIER HELD-OUT (600; heldout/results/all_tier_reference/online_result_m5_raw.jsonl)")
    rank, late, raw, rec, alltier = may_records()
    for d in ("M5_A", "M5_B", "M5_C"):
        rows = [r for r in raw if r["device"] == d]
        a, n = ascending_pairs(rows, rank, lambda r: r["ts"])
        print(f"   {d}: {len(rows)} records, (category, file name) order {a}/{n} ascending pairs; "
              f"uploads its own image to the Pi: {all(r['_pi']['content_length'] > r['image_bytes'] for r in rows)}")
    print(f"   re-collected later: index {sorted(rank[i] for i in late)} (excluded below); "
          f"100 per category, so the 600-image index has the same parity as the per-category index")
    for ag in "abc":
        t = Counter()
        for i, k in rank.items():
            if i in late:
                continue
            r = rec[(ag, i)]
            first = "backup" if k % 2 else "primary"
            t[(first, r["retry_count"], adopted(first, r["retry_count"]))] += 1
        print(f"   M5_{ag.upper()} (first prompt, retry_count, adopted): " +
              ", ".join(f"{f}/{rc}/{a}: {n}" for (f, rc, a), n in sorted(t.items())))

    print("\n4. ATTEMPTS IN THE LOGS")
    keys = Counter()
    for f in glob.glob(str(RUN / "**/*.jsonl"), recursive=True):
        for r in jl(f):
            stack = [("", r)]
            while stack:
                p, d = stack.pop()
                for k, v in (d.items() if isinstance(d, dict) else []):
                    if re.search("retry|prompt|attempt", k, re.I):
                        keys[Path(f).relative_to(RUN).as_posix() + ":" + p + k] += 1
                    stack.append((p + k + ".", v))
    print(f"   final run: JSON keys containing retry/prompt/attempt in any of its JSONL logs: {dict(keys) or 'none'}")
    may_a = [r for r in raw if r["device"] == "M5_A"]
    for rc in (0, 1):
        lat = np.array([[r["latency"][k] for k in ("m5_total_ms", "m5_postprocess_ms", "m5_vlm_infer_ms")]
                        for r in may_a if r["retry_count"] == rc])
        print(f"   May M5_A retry_count={rc} (n={len(lat)}): total {lat[:, 0].min()}-{lat[:, 0].max()} ms, "
              f"postprocess {lat[:, 1].min()}-{lat[:, 1].max()} ms, VLM inference {lat[:, 2].min()}-{lat[:, 2].max()} ms")
    fp_, fn_ = (sum(r["latency"]["m5_total_ms"] >= A_TOTAL_MS for r in may_a if r["retry_count"] == 0),
                sum(r["latency"]["m5_total_ms"] < A_TOTAL_MS for r in may_a if r["retry_count"] == 1))
    print(f"   rule total >= {A_TOTAL_MS} ms on May M5_A: {fp_} single-attempt answers flagged, {fn_} two-attempt missed")
    for d in ("M5_B", "M5_C"):
        print(f"   May {d}: retry_count {dict(Counter(r['retry_count'] for r in raw if r['device'] == d))}, "
              f"single-attempt total max {max(r['latency']['m5_total_ms'] for r in raw if r['device'] == d)} ms")

    run = {r["image_id"]: r for r in jl(RUN / "online_routing_final.jsonl")}
    a_ms = {i: (r.get("cost") or r["end_pi"]["cost"])["latency_ms"]["m5_total_ms"] for i, r in run.items()}
    flag = [i for i, v in a_ms.items() if v >= A_TOTAL_MS]
    print(f"   final run M5_A (runtime: primary first): {len(flag)}/600 answers with total >= {A_TOTAL_MS} ms "
          f"(possible second attempt = backup adopted); wording marker of those: "
          f"{dict(Counter(marker(run[i]['m5']['m5_a']['text'], 'a') for i in flag))} (+1 primary-like, -1 backup-like)")
    store = defaultdict(list)
    for f in glob.glob(str(RUN / "device_logs/m5a_sd*/m5a_aux_cache__*.jsonl")) + glob.glob(str(RUN / "device_logs/m5a_sd*/*.extracted.jsonl")):
        for d in jl(f):
            if d.get("event") == "store" and d.get("role") in ("b", "c"):
                store[(d["role"], d["image_id"], d.get("text"))].append(d.get("m5_total_ms") or 0)
    for ag in "bc":
        ms = [max(store[(ag, i, r["m5"]["m5_" + ag]["text"])]) for i, r in run.items()
              if store.get((ag, i, r["m5"]["m5_" + ag]["text"]))]
        print(f"   final run M5_{ag.upper()}: timing found for {len(ms)}/600 adopted answers (M5_A cache logs), "
              f"max total {max(ms)} ms")

    prows = [r for f in sorted(RAW.glob("m5_*.jsonl")) for r in jl(f)]
    print("   profiling: no retry_count or prompt field is stored (raw_per_category, merged); timing estimate:")
    for first in ("primary", "backup", None):
        sel = [r for r in prows if fp[r["image_id"]] == first]
        lat = [r["end_m5"]["cost"]["latency_ms"] for r in sel]
        two_a = [r for r, c in zip(sel, lat) if c["m5_postprocess_ms"] >= A_POST_MS and c["m5_vlm_infer_ms"] >= A_INFER_MS]
        either = sum(c["m5_postprocess_ms"] >= A_POST_MS or c["m5_vlm_infer_ms"] >= A_INFER_MS for c in lat)
        bc = [sum((r["aux_latency_ms"][f"m5_{g}_total_ms"] or 0) > 12600 for r in sel) for g in "bc"]
        print(f"     first={str(first):8s} n={len(sel):4d}  M5_A two attempts: {len(two_a)} (postprocess>={A_POST_MS} and "
              f"inference>={A_INFER_MS} ms; {either} if either), wording of those "
              f"{dict(Counter(marker(r['artifacts']['m5_a']['text'], 'a') for r in two_a))}; "
              f"M5_B/M5_C total > 12.6 s: {bc[0]}/{bc[1]}")

    print("\n5. WORDING MARKERS (+1 primary word, -1 backup word; calibrated on May answers whose prompt is known)")
    ref = {}
    for ag in "abc":
        for p in ("primary", "backup"):
            ref[(ag, p)] = np.mean([marker(rec[(ag, i)]["text"], ag) for i, k in rank.items() if i not in late and
                                    adopted("backup" if k % 2 else "primary", rec[(ag, i)]["retry_count"]) == p])
    sets = {"May primary-first": [alltier[i] for i, k in rank.items() if i not in late and k % 2 == 0],
            "May backup-first": [alltier[i] for i, k in rank.items() if i not in late and k % 2 == 1],
            "final run (600)": [r["m5"] for r in run.values()],
            "profiling primary-first": [r["artifacts"] for r in prows if fp[r["image_id"]] == "primary"],
            "profiling backup-first": [r["artifacts"] for r in prows if fp[r["image_id"]] == "backup"]}
    print("   estimated backup share = (mean_primary - mean) / (mean_primary - mean_backup); mean marker in brackets")
    for ag in "abc":
        print(f"   M5_{ag.upper()} reference means: primary {ref[(ag, 'primary')]:+.3f}, backup {ref[(ag, 'backup')]:+.3f}")
        parts = []
        for name, rows in sets.items():
            m = np.mean([marker(r["m5_" + ag]["text"], ag) for r in rows])
            parts.append(f"{name} {(ref[(ag, 'primary')] - m) / (ref[(ag, 'primary')] - ref[(ag, 'backup')]):.2f} [{m:+.3f}]")
        print("     " + " | ".join(parts))
    print("   windowed parity check, profiling: mean marker(even index) - mean marker(odd index) per 100-index window")
    for ag in "abc":
        diffs = []
        for cat in sorted({c for c, _, _ in idx.values()}):
            sc = {idx[r["image_id"]][1]: marker(r["artifacts"]["m5_" + ag]["text"], ag) for r in prows
                  if idx[r["image_id"]][0] == cat and not idx[r["image_id"]][2]}
            for w in range(0, 921, 100):
                ev = [v for k, v in sc.items() if w <= k < w + 100 and k % 2 == 0]
                od = [v for k, v in sc.items() if w <= k < w + 100 and k % 2 == 1]
                diffs.append(np.mean(ev) - np.mean(od))
        print(f"   M5_{ag.upper()}: {len(diffs)} windows, even-minus-odd > 0 in {sum(d > 0 for d in diffs)}, "
              f"min {min(diffs):+.2f}, median {np.median(diffs):+.2f}")


if __name__ == "__main__":
    main()
