"""Build the blind human-evaluation page for validating the GPT-4o text judge.

Samples 100 held-out images stratified by disaster category (the first 60 are
exactly 10 per category, so annotation may stop there), shuffles the
three tier responses per image under neutral labels (A/B/C), and writes:

  human_eval.html      self-contained annotation page (images embedded; gitignored)
  human_eval_key.json  candidate label -> tier mapping, kept out of the page

The rubric shown to the annotator is read from the judge script itself, so the
human and GPT-4o score against the same wording.
"""
import base64
import importlib.util
import json
import random
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
M5VLM = HERE.parents[2]
JUDGE_SCRIPT = M5VLM / "profiling" / "code" / "pipeline" / "00_gpt_judge.py"
JUDGE_SCORES = M5VLM / "heldout" / "results" / "gpt_judge_scores.jsonl"
OUT_HTML = HERE / "human_eval.html"
OUT_KEY = HERE / "human_eval_key.json"
OUT_KO = HERE / "human_eval_ko.json"

SEED = 2026
N_PER_CAT = {"earthquake": 17, "fire": 17, "flood": 17, "hurricane": 17,
             "landslide": 16, "other_disaster": 16}
FIRST_BLOCK_PER_CAT = 10
TIERS = ("m5_a", "pi_a", "jetson")
LABELS = ("A", "B", "C")


def load_judge_module():
    spec = importlib.util.spec_from_file_location("gpt_judge", JUDGE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jpeg_data_uri(path: Path) -> str:
    # Downscale for display only; GPT-4o saw the original at detail=high.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x.jpg"
        subprocess.run(["sips", "-Z", "1024", "-s", "format", "jpeg",
                        "-s", "formatOptions", "85", str(path), "--out", str(out)],
                       check=True, capture_output=True)
        return "data:image/jpeg;base64," + base64.b64encode(out.read_bytes()).decode()


def main():
    judge = load_judge_module()
    tmpl = judge.USER_PROMPT_TEMPLATE
    criteria = tmpl[tmpl.index("Score each candidate independently"):tmpl.index("Also provide a short reason")]

    records = {}
    for line in JUDGE_SCORES.open():
        r = json.loads(line)
        if r.get("type") == "gpt_text_judge" and r.get("status") == "ok":
            records[r["image_id"]] = r
    by_cat = {}
    for r in records.values():
        by_cat.setdefault(Path(r["image_path"]).parent.name, []).append(r)

    # The first FIRST_BLOCK_PER_CAT * 6 images are balanced by category, so
    # annotation can stop there and still be stratified.
    rng = random.Random(SEED)
    first, rest = [], []
    for cat, k in N_PER_CAT.items():
        s = rng.sample(sorted(by_cat[cat], key=lambda r: r["image_id"]), k)
        first += s[:FIRST_BLOCK_PER_CAT]
        rest += s[FIRST_BLOCK_PER_CAT:]
    rng.shuffle(first)
    rng.shuffle(rest)
    picked = first + rest

    items, key = [], {}
    for r in picked:
        tiers = list(TIERS)
        rng.shuffle(tiers)
        key[r["image_id"]] = dict(zip(LABELS, tiers))
        items.append({
            "id": r["image_id"],
            "img": jpeg_data_uri(Path(r["image_path"])),
            "cands": [{"c": c, "text": r["tier_outputs"][t]["text"]} for c, t in zip(LABELS, tiers)],
        })

    cats = Counter(Path(records[i]["image_path"]).parent.name for i in key)
    assert cats == Counter(N_PER_CAT), cats
    assert len(items) == len(key) == 100
    first_cats = Counter(Path(records[it["id"]]["image_path"]).parent.name for it in items[:6 * FIRST_BLOCK_PER_CAT])
    assert set(first_cats.values()) == {FIRST_BLOCK_PER_CAT}, first_cats
    assert all(sorted(m.values()) == sorted(TIERS) for m in key.values())
    assert all(c["text"].strip() for it in items for c in it["cands"])

    page = (HERE / "human_eval_template.html").read_text()
    page = page.replace("__ITEMS__", json.dumps(items).replace("</", "<\\/"))
    page = page.replace("__SYSTEM__", json.dumps(judge.SYSTEM_PROMPT).replace("</", "<\\/"))
    page = page.replace("__CRITERIA__", json.dumps(criteria).replace("</", "<\\/"))
    # Korean reading aid shown under each candidate ("<image_id>|<label>" -> text); scoring stays on the English original.
    ko = json.loads(OUT_KO.read_text()) if OUT_KO.exists() else {}
    page = page.replace("__KO__", json.dumps(ko, ensure_ascii=False).replace("</", "<\\/"))
    OUT_HTML.write_text(page)
    OUT_KEY.write_text(json.dumps({"seed": SEED, "judge_scores": str(JUDGE_SCORES.relative_to(M5VLM)),
                                   "key": key}, indent=1))
    print(f"wrote {OUT_HTML.name} ({OUT_HTML.stat().st_size / 1e6:.1f} MB), {OUT_KEY.name}; categories {dict(cats)}")


if __name__ == "__main__":
    main()
