#!/usr/bin/env python3
"""Judge input for a routing run: the answers the run itself produced, in the form 00_gpt_judge.py scores.

    python3 build_runtime_judge_input.py --runtime <online_routing_final.jsonl> --all-tier <collection>.jsonl \
        --out <judge_input>.jsonl
    python3 profiling/code/pipeline/00_gpt_judge.py --collection <judge_input>.jsonl --image-root <images> --out <judge>.jsonl
    python3 profiling/code/pipeline/01_quality_components.py --collection <judge_input>.jsonl --judge <judge>.jsonl ...

The judge always scores three reports of one image together. The tiers the run reached (M5_A always, Pi_A
after Stage 1 escalation, Jetson after Stage 2 escalation) are the run's own answers; a tier the run never
reached is filled from the all-tier collection of the same image, so the prompt has the same form as for the
baselines. `source` says which slots came from the run. The routed answer is the `route_label` slot, which is
always from the run. (MEDIC's Table I scored route_label on the all-tier run's answers instead.)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

TIERS = ("m5_a", "pi_a", "jetson")
KEEP = ("text", "cat", "sev", "conf")


def run_answers(r: dict) -> dict:
    return {"m5_a": (r.get("m5") or {}).get("m5_a"),
            "pi_a": (r.get("pi") or {}).get("pi_a"),
            "jetson": (r.get("jetson") or {}).get("jetson")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--all-tier", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    all_tier = {r["image_id"]: r for r in map(json.loads, args.all_tier.open(encoding="utf-8"))}
    rows, n = [], Counter()
    for r in map(json.loads, args.runtime.open(encoding="utf-8")):
        if r.get("type") != "online_routing_final" or r.get("status") != "ok":
            n["skipped_not_ok"] += 1
            continue
        i, ans = r["image_id"], run_answers(r)
        if not (ans[r["route_label"]] or {}).get("text"):
            raise SystemExit(f"{i}: routed to {r['route_label']} but the run has no answer for it")
        row = {"image_id": i, "status": "ok", "route_label": r["route_label"], "source": {}}
        for t in TIERS:
            if (ans[t] or {}).get("text"):
                row[t], row["source"][t] = {k: ans[t].get(k) for k in KEEP}, "run"
            elif i in all_tier:
                row[t], row["source"][t] = {k: all_tier[i][t].get(k) for k in KEEP}, "all_tier"
            else:
                raise SystemExit(f"{i}: no {t} answer in the run or the all-tier collection")
        n[f"route={r['route_label']} " + " ".join(f"{t}:{row['source'][t]}" for t in TIERS)] += 1
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}")
    for k, v in sorted(n.items()):
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
