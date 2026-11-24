#!/usr/bin/env python3
"""Clean the pulled EIDSeg result files before putting them back on the devices (2026-10-30).

- Each output keeps one ok line per image. `fail` lines (one per failed attempt) are dropped.
- Jetson records that must be redone are dropped too, so the Jetson pipeline (resume) redoes them at
  768 px once the file is back. From 2026-10-28 17:57 to 10-29 06:42 vLLM on the Jetson kept timing out
  (memory swap) until it was restarted:
    - 64 answers from that window took over 20 s (outside it: median about 10 s, max 20.4 s), which
      inflates Jetson latency and energy;
    - an image that timed out at 768 px was retried at 640/512/384 px, which only jetson_infer.out shows.
      Pass its "[N] <image>" and "recovered at dim" lines as the second argument.

    python3 revision/eidseg/clean_for_rerun.py revision/eidseg/full/pulled_20261030_1401 [retry_lines.txt]

Writes full/jetson_eidseg_full.jsonl, full/pi_b/pi_b_eidseg_full.jsonl and full/jetson_redo_ids.txt.
"""
import json
import re
import sys
from pathlib import Path

WINDOW = ("2026-10-28T17:57", "2026-10-29T06:42")
SLOW_MS = 20000
OUT = Path(__file__).resolve().parent / "full"


def ok_lines(path: Path) -> list[str]:
    return [l for l in path.open(encoding="utf-8") if json.loads(l).get("status") == "ok"]


def main(src: Path, retry_lines: str | None) -> None:
    jet = ok_lines(src / "jetson_eidseg_full.jsonl")
    redo = set()
    for l in jet:
        r = json.loads(l)
        if WINDOW[0] <= r["ts"] < WINDOW[1] and r["end_jetson"]["vlm_ms"] > SLOW_MS:
            redo.add(r["image_id"])
    n_slow = len(redo)
    if retry_lines:
        cur, seen, smaller = None, set(), set()
        for line in open(retry_lines, encoding="utf-8", errors="ignore"):
            if m := re.match(r"\[\d+\] (\S+\.jpg)", line):
                cur = m.group(1)
                seen.add(cur)
            elif "recovered at dim" in line and cur:
                smaller.add(cur)
        print(f"log: {len(seen)} images, covers {len(redo & seen)}/{n_slow} slow ones; "
              f"recovered at a smaller size {len(smaller)}, of which not slow {len(smaller - redo)}")
        redo |= smaller

    keep = [l for l in jet if json.loads(l)["image_id"] not in redo]
    (OUT / "jetson_eidseg_full.jsonl").write_text("".join(keep), encoding="utf-8")
    (OUT / "jetson_redo_ids.txt").write_text("".join(i + "\n" for i in sorted(redo)), encoding="utf-8")
    pib = ok_lines(src / "pi_b_eidseg_full.jsonl")
    (OUT / "pi_b").mkdir(exist_ok=True)
    (OUT / "pi_b" / "pi_b_eidseg_full.jsonl").write_text("".join(pib), encoding="utf-8")
    print(f"Jetson: {len(jet)} ok, {len(redo)} to redo ({n_slow} slow) -> {len(keep)} kept;  Pi_B: {len(pib)} ok")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
