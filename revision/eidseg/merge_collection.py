#!/usr/bin/env python3
"""Merge the EIDSeg tier outputs into one collection file, with profiling/code/collection/merge_tier_outputs.py.

    python3 revision/eidseg/merge_collection.py full      # files from revision/eidseg/results_inference/, see RUN.md "끝나면"
    python3 revision/eidseg/merge_collection.py pilot50

Before merging, it fills the M5_A->Pi_A transfer (m5_pi_http_e2e_ms, m5_pi_rx_bytes) from Pi_A's
oracle_patch__<run>.jsonl. M5_A sends these in a PATCH that usually reaches Pi_A after the merge, and the
collection pi_a_inference_pipeline.py reads late patches from a file name fixed to the MEDIC run
(oracle_patch__pilot_300_run1.jsonl), so under run_id eidseg_* it found none: 39 of 48 pilot images had 0.
Only zero/missing values are filled; anything Pi_A recorded itself is kept.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "profiling" / "code" / "collection"))
import merge_tier_outputs as m  # noqa: E402

RUNS = {  # run -> (folder, M5 merged, Pi_A+Pi_B as Jetson received them, Pi_A own output, Jetson output, patches)
    "pilot50": (HERE, "merged_oracle__eidseg_pilot50.jsonl", "jetson_collected__eidseg_pilot50.jsonl",
                "pi_a_eidseg_pilot50.jsonl", "jetson_review_pilot.jsonl", "oracle_patch__eidseg_pilot50.jsonl"),
    "full": (HERE / "results_inference","merged_oracle__eidseg_full.jsonl", "jetson_collected__eidseg_full.jsonl",
             "pi_a_eidseg_full.jsonl", "jetson_eidseg_full.jsonl", "oracle_patch__eidseg_full.jsonl"),
}
FILL = ("m5_pi_http_e2e_ms", "m5_pi_rx_bytes")


def main(run: str) -> None:
    root, m5, pi, pi_a, jet, patch_file = RUNS[run]
    patches = {}
    for line in (root / patch_file).open(encoding="utf-8"):
        p = json.loads(line)
        if p.get("role") == "a":
            patches[p["image_id"]] = p  # last patch per image wins
    filled = Counter()

    def build_record(iid, m5_rec, pi_rec, *rest):
        costs, p = pi_rec["pi_a"]["costs"], patches.get(iid, {})
        for k in FILL:
            if not costs.get(k) and p.get(k):
                costs[k] = p[k]
                filled[k] += 1
        return build_record_orig(iid, m5_rec, pi_rec, *rest)

    build_record_orig = m.build_record
    m.build_record = build_record
    m.ROOT, m.M5_GLOB, m.PI_GLOB, m.PI_A_ORIG_GLOB, m.JET_GLOB = root, m5, pi, pi_a, jet
    m.OUT_PATH = HERE / f"collection_eidseg_{run}.jsonl"
    m.main()
    print(f"filled from {patch_file} ({len(patches)} M5_A patches): {dict(filled)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
