"""Judge input for re-scoring the 180 human-rated answers with the final evaluation's snapshot.

The 60 images every rater finished, with exactly the tier texts the raters saw (tier_outputs of the earlier
judge file that build_human_eval.py read). Written as a 00_gpt_judge.py --collection file.
"""
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
M5VLM = HERE.parents[2]
sys.path.insert(0, str(M5VLM / "revision/human_evaluation"))
import analyze_human_eval as H  # noqa: E402

key = json.load(open(M5VLM / "revision/human_evaluation/rating_page/human_eval_key.json"))["key"]
old = {r["image_id"]: r for r in map(json.loads, open(H.JUDGE)) if r.get("type") == "gpt_text_judge" and r.get("status") == "ok"}
raters = [H.load_rater(p, key, old) for p in sorted(glob.glob(str(M5VLM / "revision/human_evaluation/ratings/rater*.json")))]
ids = sorted(set.intersection(*(set(r) for r in raters)))
assert len(raters) == 10 and len(ids) == 60
with open(HERE / "input_he60.jsonl", "w") as f:
    for i in ids:
        f.write(json.dumps({"image_id": i, **{t: old[i]["tier_outputs"][t] for t in H.TIERS}}, ensure_ascii=False) + "\n")
print(len(ids), "images ->", HERE / "input_he60.jsonl")
