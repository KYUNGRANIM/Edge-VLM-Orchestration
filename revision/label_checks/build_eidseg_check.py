"""Build the blind check for a mask-derived EIDSeg severity rule.

Samples 20 non-duplicate EIDSeg images from each mask group and asks the
annotator for a MEDIC 3-level severity (or "not judgeable"), using the same page
as the MMBD/MEDIC check. Agreement with the rule (debris -> severe,
damaged-only -> mild, little -> little_or_none) is then compared with the MEDIC
baseline from that check.

  eidseg_check.html      self-contained page (images embedded; gitignored)
  eidseg_check_key.json  item -> image and mask group
"""
import csv
import json
import random
from pathlib import Path

from build_human_eval import HERE, jpeg_data_uri

STATS = HERE.parents[1] / "revision" / "eidseg" / "dataset" / "eidseg_stats.csv"
IMAGES = Path("/Volumes/CORSAIR/EIDSeg_raw/data")
OUT_HTML = HERE / "eidseg_check.html"
OUT_KEY = HERE / "eidseg_check_key.json"
SEED, PER_GROUP, MAX_DUP_DIST = 2028, 20, 10


def group(r):
    deb = float(r["frac_Debris"])
    dmg = float(r["frac_D_Building"]) + float(r["frac_D_Road"])
    if deb >= 0.05:
        return "debris"
    if deb < 0.01 and dmg >= 0.05:
        return "damaged_only"
    if deb < 0.01 and dmg < 0.01:
        return "little"
    return None


def main():
    rng = random.Random(SEED)
    rows = [r for r in csv.DictReader(STATS.open()) if int(r["medic_hash_dist"]) > MAX_DUP_DIST]
    picked = []
    for g in ("debris", "damaged_only", "little"):
        pool = sorted((r for r in rows if group(r) == g), key=lambda r: r["image"])
        picked += [(g, r) for r in rng.sample(pool, PER_GROUP)]
    rng.shuffle(picked)

    items, key = [], {}
    for n, (g, r) in enumerate(picked):
        item_id = f"e{n:03d}"
        key[item_id] = {"image": r["image"], "split": r["split"], "group": g, "event": r["event"]}
        items.append({"id": item_id, "img": jpeg_data_uri(IMAGES / r["split"] / "images" / "default" / r["image"])})
    assert len(items) == 3 * PER_GROUP

    page = (HERE / "severity_check_template.html").read_text().replace("__STORE__", "eidseg_check_v1")
    OUT_HTML.write_text(page.replace("__ITEMS__", json.dumps(items).replace("</", "<\\/")))
    OUT_KEY.write_text(json.dumps({"seed": SEED, "key": key}, indent=1))
    print(f"wrote {OUT_HTML.name} ({OUT_HTML.stat().st_size / 1e6:.1f} MB), {OUT_KEY.name}")


if __name__ == "__main__":
    main()
