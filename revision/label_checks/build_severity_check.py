"""Build the blind severity check: can MMBDisasterSeverity labels be judged from
the image alone, and does its 5-level scale map onto MEDIC's 3 levels?

Mixes 60 MMBD images (12 per MMBD level, from the five categories we keep) with
60 MEDIC held-out images (20 per MEDIC level). The annotator rates each image on
MEDIC's three levels or marks it as not judgeable; source, original label,
category and text stay out of the page (they are written to the key file).

  severity_check.html      self-contained page (images embedded; gitignored)
  severity_check_key.json  image -> source and original label
"""
import csv
import json
import random
from pathlib import Path

from build_human_eval import HERE, JUDGE_SCORES, jpeg_data_uri

MMBD = Path("/Volumes/CORSAIR/MMBDisasterSeverity_raw")
MEDIC_QUALITY = HERE.parents[0] / "heldout/results/labels/01_quality_components.jsonl"
OUT_HTML = HERE / "severity_check.html"
OUT_KEY = HERE / "severity_check_key.json"

SEED = 2027
MMBD_CATEGORIES = ("Earthquake", "Flood", "Landslides", "Wildfire", "Tropical Storm")
MMBD_LEVELS = ("Minimal", "Mild", "Moderate", "Severe", "Catastrophic")
MEDIC_LEVELS = ("little_or_none", "mild", "severe")
PER_MMBD_LEVEL, PER_MEDIC_LEVEL = 12, 20


def main():
    rng = random.Random(SEED)
    files = {p.name: p for p in (MMBD / "images").rglob("*") if p.is_file() and not p.name.startswith("._")}
    rows = [r for s in ("train", "validation", "test")
            for r in csv.DictReader(open(MMBD / "csv" / f"{s}.csv", encoding="utf-8-sig"))
            if r["category"] in MMBD_CATEGORIES]
    picked = []
    for level in MMBD_LEVELS:
        pool = sorted((r for r in rows if r["label"] == level), key=lambda r: r["image_name"])
        for r in rng.sample(pool, PER_MMBD_LEVEL):
            picked.append(("mmbd", r["image_name"], files[r["image_name"]], level, r["category"]))

    paths = {}
    for line in JUDGE_SCORES.open():
        r = json.loads(line)
        if r.get("type") == "gpt_text_judge" and r.get("status") == "ok":
            paths[r["image_id"]] = Path(r["image_path"])
    medic = [r for r in map(json.loads, MEDIC_QUALITY.open()) if r.get("type") == "quality_components"]
    for level in MEDIC_LEVELS:
        pool = sorted((r for r in medic if r["gt"]["sev_raw"] == level), key=lambda r: r["image_id"])
        for r in rng.sample(pool, PER_MEDIC_LEVEL):
            picked.append(("medic", r["image_id"], paths[r["image_id"]], level, r["gt"]["cat"]))
    rng.shuffle(picked)

    items, key = [], {}
    for n, (source, name, path, level, cat) in enumerate(picked):
        item_id = f"s{n:03d}"
        key[item_id] = {"source": source, "image": name, "label": level, "category": cat}
        items.append({"id": item_id, "img": jpeg_data_uri(path)})

    assert len(items) == 120
    assert sum(v["source"] == "mmbd" for v in key.values()) == 60

    page = (HERE / "severity_check_template.html").read_text()
    page = page.replace("__STORE__", "severity_check_v1")
    OUT_HTML.write_text(page.replace("__ITEMS__", json.dumps(items).replace("</", "<\\/")))
    OUT_KEY.write_text(json.dumps({"seed": SEED, "key": key}, indent=1, ensure_ascii=False))
    print(f"wrote {OUT_HTML.name} ({OUT_HTML.stat().st_size / 1e6:.1f} MB), {OUT_KEY.name}")


if __name__ == "__main__":
    main()
