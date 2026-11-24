"""Prepare EIDSeg (post-earthquake social-media images with pixel damage labels)
as the external evaluation set.

stats: writes eidseg_stats.csv with, per image, the area fraction of each
       damage class (polygons rasterized per class), the earthquake inferred
       from the file name, and whether the image duplicates one of the 6,126
       MEDIC images used in the study (64-bit difference hash).
build: writes the SD-card image folder and a MEDIC-format manifest for every
       image that is neither a MEDIC near-duplicate nor one of the 60 images
       used to validate the severity rule.
split: marks the first HELDOUT images of the shuffled order as held-out (the
       rest are profiling data for training the EIDSeg router) and copies them
       into their own folder for the routing run.
category: sets the ground-truth category by visible evidence, the same rule
       every tier's prompt uses: images without annotated damage become
       other_disaster ("people, responders, equipment, vehicles, or aftermath
       without a clear visible hazard"); the rest stay earthquake. The event
       label is kept in category_event for the all-earthquake sensitivity run.
       Decided on the pilot, before any full-run results (2026-10-27).

Usage:
  python3 revision/eidseg/dataset/prepare_eidseg.py stats /Volumes/CORSAIR/EIDSeg_raw/data
  python3 revision/eidseg/dataset/prepare_eidseg.py build /Volumes/CORSAIR/EIDSeg_raw/data /Volumes/CORSAIR/EIDSeg_eval
  python3 revision/eidseg/dataset/prepare_eidseg.py split /Volumes/CORSAIR/EIDSeg_eval
  python3 revision/eidseg/dataset/prepare_eidseg.py category /Volumes/CORSAIR/EIDSeg_eval
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "eidseg_stats.csv"
MEDIC_USED = [Path("/Volumes/CORSAIR/MEDIC_v2_train/images_sampled"),
              Path("/Volumes/CORSAIR/MEDIC_v2_test/images_sampled")]
CLASSES = ("Debris", "D_Building", "D_Road", "UD_Building", "UD_Road")
DUP_BITS = 6

# (event, first day, last day) for tweet-ID dating; Haiti 2010 and Wenchuan 2008
# predate snowflake IDs and are matched by keyword only.
EVENTS = [("nepal_2015", "2015-04-25", "2015-07-31"), ("illapel_2015", "2015-09-16", "2015-11-15"),
          ("ecuador_2016", "2016-04-16", "2016-07-15"), ("mexico_2017", "2017-09-07", "2017-11-05"),
          ("iran_iraq_2017", "2017-11-12", "2018-01-31"), ("turkiye_2023", "2023-02-06", "2023-05-31"),
          ("morocco_2023", "2023-09-08", "2023-11-30")]
KEYWORDS = {"morocco": "morocco_2023", "turk": "turkiye_2023", "syria": "turkiye_2023", "haiti": "haiti_2010",
            "wenchuan": "wenchuan_2008", "sichuan": "wenchuan_2008", "nepal": "nepal_2015",
            "ecuador": "ecuador_2016", "mexico": "mexico_2017", "chile": "illapel_2015", "illapel": "illapel_2015",
            "iran": "iran_iraq_2017", "iraq": "iran_iraq_2017"}
IN_MEDIC = {"nepal_2015", "ecuador_2016", "mexico_2017", "iran_iraq_2017"}  # via ASONAM17 / CrisisMMD


def event_of(name: str):
    low = name.lower()
    for k, ev in KEYWORDS.items():
        if k in low:
            return ev
    head = low.split("_")[0].split(".")[0]
    if head.isdigit() and int(head) > 10**16:  # Twitter snowflake ID
        day = datetime.fromtimestamp(((int(head) >> 22) + 1288834974657) / 1000, tz=timezone.utc).date().isoformat()
        for ev, a, b in EVENTS:
            if a <= day <= b:
                return ev
    return None


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        px = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def stats(root: Path):
    medic_hashes = np.array([dhash(p) for d in MEDIC_USED for p in d.rglob("*.jpg") if not p.name.startswith("._")],
                            dtype=np.uint64)
    print(f"hashed {len(medic_hashes)} MEDIC images")
    rows = []
    for split in ("train", "val", "test"):
        for im in ET.parse(root / split / f"{split}.xml").getroot().findall("image"):
            name, w, h = im.get("name"), int(im.get("width")), int(im.get("height"))
            frac = {}
            for cls in CLASSES:
                mask = Image.new("1", (w, h), 0)
                draw = ImageDraw.Draw(mask)
                for poly in im.findall(f"polygon[@label='{cls}']"):
                    pts = [tuple(map(float, p.split(","))) for p in poly.get("points").split(";")]
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=1)
                frac[cls] = np.asarray(mask).mean()
            path = root / split / "images" / "default" / name
            dist = int(np.unpackbits((medic_hashes ^ np.uint64(dhash(path))).view(np.uint8)).reshape(-1, 64).sum(1).min())
            ev = event_of(name)
            rows.append({"image": name, "split": split, "width": w, "height": h,
                         **{f"frac_{c}": round(float(v), 5) for c, v in frac.items()},
                         "event": ev or "", "event_in_medic": int(ev in IN_MEDIC) if ev else "",
                         "medic_dup": int(dist <= DUP_BITS), "medic_hash_dist": dist})
    assert len(rows) == 3266, len(rows)
    with OUT_CSV.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {OUT_CSV.name}: {len(rows)} images")


MANIFEST = HERE / "eidseg_manifest.csv"
CHECK_KEY = HERE.parents[2] / "revision" / "label_checks" / "eidseg_check_key.json"
EXCLUDE_DUP_DIST = 10  # stricter than DUP_BITS, to be safe
SEVERE_DEBRIS = 0.05   # fixed before the 60-image blind check (kappa 0.56 vs 0.43 for MEDIC's own labels);
                       # 0.15/0.30 (best fit to that single annotator, kappa 0.71) kept for sensitivity
MAX_SIDE, MAX_KEEP_BYTES = 1280, 2 * 1024 * 1024  # MEDIC held-out: 90% of images <= 1200 px
SEED, PILOT = 2029, 50


def severity(r, severe_at):
    deb = float(r["frac_Debris"])
    if deb >= severe_at:
        return "severe"
    if deb + float(r["frac_D_Building"]) + float(r["frac_D_Road"]) < 0.01:
        return "little_or_none"
    return "mild"


def to_sd_jpeg(src: Path, dst: Path):
    """Copy JPEGs that already look like MEDIC images; downscale/convert the rest.
    JPEG by content, not by name: 6 EIDSeg files named .jpg are PNG, which the firmware skips."""
    out = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(src)],
                         capture_output=True, text=True, check=True).stdout.split()
    w, h = int(out[-3]), int(out[-1])
    if src.read_bytes()[:2] == b"\xff\xd8" and max(w, h) <= MAX_SIDE and src.stat().st_size <= MAX_KEEP_BYTES:
        shutil.copyfile(src, dst)
    else:
        cmd = ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "90"]
        if max(w, h) > MAX_SIDE:
            cmd += ["-Z", str(MAX_SIDE)]
        subprocess.run(cmd + [str(src), "--out", str(dst)], check=True, capture_output=True)
    out = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(dst)],
                         capture_output=True, text=True, check=True).stdout.split()
    return int(out[-3]), int(out[-1])


def build(root: Path, out: Path):
    validation = {v["image"] for v in json.loads(CHECK_KEY.read_text())["key"].values()}
    rows = [r for r in csv.DictReader(OUT_CSV.open())
            if int(r["medic_hash_dist"]) > EXCLUDE_DUP_DIST and r["image"] not in validation]
    random.Random(SEED).shuffle(rows)

    full, pilot = out / "images_sampled" / "earthquake", out / "pilot50" / "earthquake"
    full.mkdir(parents=True)
    pilot.mkdir(parents=True)
    manifest = []
    for idx, r in enumerate(rows, start=1):
        src = root / r["split"] / "images" / "default" / r["image"]
        name = f"{idx:06d}__{hashlib.sha1(src.read_bytes()).hexdigest()[:10]}.jpg"
        w, h = to_sd_jpeg(src, full / name)
        assert (full / name).stat().st_size < 5 * 1024 * 1024  # firmware MAX_JPG_SIZE
        if idx <= PILOT:
            shutil.copyfile(full / name, pilot / name)
        manifest.append({"split": r["split"], "image_id": "EIDSeg", "event_name": r["event"] or "unknown",
                         "image_path": f"data/{r['split']}/images/default/{r['image']}",
                         "informative": "", "humanitarian": "", "disaster_types": "earthquake",
                         "damage_severity": severity(r, SEVERE_DEBRIS), "width": w, "height": h,
                         "raw_bytes": (full / name).stat().st_size, "group": "earthquake",
                         "local_image_path": f"images/earthquake/{name}",
                         "severity_debris15": severity(r, 0.15), "severity_debris30": severity(r, 0.30),
                         "event_in_medic": r["event_in_medic"]})
    with MANIFEST.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0]))
        wr.writeheader()
        wr.writerows(manifest)
    shutil.copyfile(MANIFEST, out / "eidseg_manifest.csv")
    assert len({m["local_image_path"] for m in manifest}) == len(manifest)
    print(f"wrote {len(manifest)} images to {full} (+{PILOT} pilot) and {MANIFEST.name}")


HELDOUT = 600  # same size as the MEDIC held-out set; a learning curve shows 2,331 training images suffice


def split(out: Path):
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    assert "eval_split" not in rows[0], "manifest is already split"
    heldout = out / f"heldout{HELDOUT}" / "earthquake"
    heldout.mkdir(parents=True)
    for r in rows:
        name = Path(r["local_image_path"]).name
        r["eval_split"] = "heldout" if int(name.split("__")[0]) <= HELDOUT else "profiling"
        if r["eval_split"] == "heldout":
            shutil.copyfile(out / "images_sampled" / "earthquake" / name, heldout / name)
    with MANIFEST.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    shutil.copyfile(MANIFEST, out / "eidseg_manifest.csv")
    assert sum(r["eval_split"] == "heldout" for r in rows) == HELDOUT
    print(f"held-out {HELDOUT} -> {heldout}; profiling {len(rows) - HELDOUT}")


def category(out: Path):
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    assert "category_event" not in rows[0], "categories already assigned"
    for r in rows:
        r["category_event"] = r["group"]
        visible = "other_disaster" if r["damage_severity"] == "little_or_none" else "earthquake"
        r["group"] = r["disaster_types"] = visible
    with MANIFEST.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    shutil.copyfile(MANIFEST, out / "eidseg_manifest.csv")
    n_other = sum(r["group"] == "other_disaster" for r in rows)
    print(f"category: earthquake {len(rows) - n_other}, other_disaster {n_other}")


SD_CHUNK = 300  # the firmware opens every directory entry while scanning; 2,931 files in one folder takes minutes per boot


def sdcard(out: Path):
    """SD-card copy split into sub-folders of SD_CHUNK images (b00 = 000001-000300, ...).
    The firmware scans recursively and sorts by full path, so the processing order is unchanged."""
    src = out / "images_sampled" / "earthquake"
    dst = out / "sd_card" / "EIDSeg_eval" / "images_sampled" / "earthquake"
    names = sorted(p.name for p in src.glob("[0-9]*__*.jpg"))
    assert len(names) == 2931, len(names)
    for name in names:
        part = dst / f"b{(int(name.split('__')[0]) - 1) // SD_CHUNK:02d}"
        part.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / name, part / name)  # data only: no xattrs, so no ._ files on the card
    print(f"wrote {len(names)} images in {len(list(dst.iterdir()))} folders under {dst}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats").add_argument("root", type=Path)
    b = sub.add_parser("build")
    b.add_argument("root", type=Path)
    b.add_argument("out", type=Path)
    sub.add_parser("split").add_argument("out", type=Path)
    sub.add_parser("category").add_argument("out", type=Path)
    sub.add_parser("sdcard").add_argument("out", type=Path)
    args = p.parse_args()
    if args.cmd == "stats":
        stats(args.root)
    elif args.cmd == "build":
        build(args.root, args.out)
    elif args.cmd == "split":
        split(args.out)
    elif args.cmd == "category":
        category(args.out)
    else:
        sdcard(args.out)


if __name__ == "__main__":
    main()
