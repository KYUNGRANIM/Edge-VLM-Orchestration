#!/usr/bin/env python3
"""Sample a held-out MEDIC test subset: 100 images per disaster type."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

CLASSES = ("earthquake", "fire", "flood", "hurricane", "landslide", "other_disaster")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/Volumes/CORSAIR/MEDIC_v2_test/images"))
    ap.add_argument("--dst", type=Path, default=Path("/Volumes/CORSAIR/MEDIC_v2_test/images_sampled"))
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.dst.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for cls in CLASSES:
        src_dir = args.src / cls
        dst_dir = args.dst / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = images_in(src_dir)
        if len(files) < args.per_class:
            raise RuntimeError(f"{cls}: only {len(files)} images, need {args.per_class}")
        picked = sorted(rng.sample(files, args.per_class), key=lambda p: p.name)
        for idx, src in enumerate(picked):
            dst = dst_dir / src.name
            if args.copy:
                shutil.copy2(src, dst)
            elif not dst.exists():
                dst.symlink_to(src)
            manifest_rows.append({
                "class": cls,
                "index": idx,
                "src": str(src),
                "dst": str(dst),
                "filename": src.name,
            })
        print(f"[OK] {cls}: {len(picked)}")

    manifest = args.dst / "sample_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "index", "src", "dst", "filename"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[DONE] total={len(manifest_rows)} manifest={manifest}")


if __name__ == "__main__":
    main()

