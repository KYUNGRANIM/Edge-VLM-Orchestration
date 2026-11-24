"""Agreement between human raters and the GPT-4o text judge (held-out 600 sample, 3 tier answers per image).

    python3 analyze_human_eval.py <ratings/rater*.json> [<second rater>.json ...]

Human sub-scores go through the judge's own text-score formula and the paper's quality / Q-Sat / reference-tier
rules, so every number compares like with like. CIs are 95% percentile bootstraps over images (the 3 answers of
one image move together). With two or more raters it also reports human-human agreement as the ceiling.
"""
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

M5VLM = Path(__file__).resolve().parents[2]
JUDGE = M5VLM / "heldout/results/gpt_judge_scores.jsonl"
OUT = M5VLM / "heldout/results/labels"
ROUTING = M5VLM / "revision/first_submission/live_run/online_routing_final.jsonl"
ALL_TIER = M5VLM / "heldout/results/collection.jsonl"
TIERS = ("m5_a", "pi_a", "jetson")
CRIT = ("damage_grounding", "damage_overclaim", "response_appropriateness",
        "response_operationality", "response_target_alignment")
Q_TH, TAU = 0.65, 0.10          # Q-Sat threshold (first submission) and reference tolerance (03_oracle_labels)
W_CAT, W_SEV, W_TEXT = 0.3, 0.3, 0.4

spec = importlib.util.spec_from_file_location("j", M5VLM / "profiling/code/pipeline/00_gpt_judge.py")
judge_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge_mod)
text_score = judge_mod.compute_text_judge_score


def kappa(a, b, weighted=False, k=None):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.arange(k) if k else np.unique(np.concatenate([a, b]))
    idx = {c: n for n, c in enumerate(cats)}
    m = len(cats)
    obs = np.zeros((m, m))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((m, m))
    w = ((i - j) / (m - 1)) ** 2 if weighted else (i != j).astype(float)
    return 1 - (w * obs).sum() / (w * exp).sum() if (w * exp).sum() else 1.0


def boot(fn, ids, n=2000, seed=0):
    rng = random.Random(seed)
    vals = sorted(fn([rng.choice(ids) for _ in ids]) for _ in range(n))
    return vals[int(.025 * n)], vals[int(.975 * n)]


def reference_tier(q, cost):
    best = max(q.values())
    feasible = [t for t in TIERS if best - q[t] <= TAU + 1e-12]
    return min(feasible, key=lambda t: (cost[t], -q[t]))


def load_rater(path, key, judge):
    """image_id -> tier -> {crit: int} (labels A/B/C mapped back to tiers)."""
    s = json.load(open(path))["scores"]
    out = {}
    for img, by_label in s.items():
        if all(all(isinstance(by_label.get(c, {}).get(k), int) for k in CRIT) for c in "ABC"):
            out[img] = {key[img][c]: by_label[c] for c in "ABC"}
    return out


def main(paths):
    key = json.load(open(Path(__file__).parent / "rating_page" / "human_eval_key.json"))["key"]
    judge = {r["image_id"]: r for r in map(json.loads, open(JUDGE)) if r.get("type") == "gpt_text_judge" and r.get("status") == "ok"}
    qc = {r["image_id"]: r for r in map(json.loads, open(OUT / "01_quality_components.jsonl")) if r["type"] == "quality_components"}
    oracle = {r["image_id"]: r for r in map(json.loads, open(OUT / "03_oracle_labels.jsonl")) if r.get("type") == "oracle_label"}
    cost = {i: r["cost"] for i, r in oracle.items()}
    oracle_q = {i: r["quality"] for i, r in oracle.items()}
    raters = [load_rater(p, key, judge) for p in paths]
    ids = sorted(set.intersection(*(set(r) for r in raters)))
    words = {(i, t): len(judge[i]["tier_outputs"][t]["text"].split()) for i in ids for t in TIERS}

    def quality(i, t, txt):
        # 03_oracle_labels stores round(q, 8); without it 0.3+0.15+0.2 = 0.6499... fails the 0.65 threshold
        return round(W_CAT * qc[i][t]["cat_score"] + W_SEV * qc[i][t]["sev_score"] + W_TEXT * txt, 8)

    # GPT side, rebuilt from the judge record and checked against the pipeline's stored values
    gpt_sub = {i: {t: judge[i]["scores"][t] for t in TIERS} for i in ids}
    gpt_txt = {(i, t): text_score(gpt_sub[i][t]) for i in ids for t in TIERS}
    for i in ids:
        for t in TIERS:
            assert abs(gpt_txt[i, t] - qc[i][t]["text_score"]) < 1e-6, (i, t)
            assert quality(i, t, gpt_txt[i, t]) == oracle_q[i][t], (i, t)

    def report(name, sub_a, txt_a, sub_b, txt_b):
        """a = reference side (human), b = compared side."""
        q_a = {(i, t): quality(i, t, txt_a[i, t]) for i in ids for t in TIERS}
        q_b = {(i, t): quality(i, t, txt_b[i, t]) for i in ids for t in TIERS}
        pairs = lambda s: [(i, t) for i in s for t in TIERS]
        sat = lambda q, s: [int(q[p] >= Q_TH) for p in pairs(s)]
        agree = lambda s: float(np.mean(np.equal(sat(q_a, s), sat(q_b, s))))
        kap = lambda s: kappa(sat(q_a, s), sat(q_b, s), k=2)
        ref = lambda q, i: reference_tier({t: q[i, t] for t in TIERS}, cost[i])
        ref_agree = lambda s: float(np.mean([ref(q_a, i) == ref(q_b, i) for i in s]))
        spear = lambda s: float(np.corrcoef(*[np.argsort(np.argsort(v)) for v in ([txt_a[p] for p in pairs(s)], [txt_b[p] for p in pairs(s)])])[0, 1])

        print(f"\n=== {name}  ({len(ids)} images, {3 * len(ids)} answers)")
        print(f"Q-Sat pass rate        {np.mean(sat(q_a, ids)):.1%} vs {np.mean(sat(q_b, ids)):.1%}")
        print(f"Q-Sat agreement        {agree(ids):.1%}  CI {boot(agree, ids)[0]:.1%}–{boot(agree, ids)[1]:.1%}")
        print(f"Q-Sat kappa            {kap(ids):.2f}  CI {boot(kap, ids)[0]:.2f}–{boot(kap, ids)[1]:.2f}")
        print(f"reference tier same    {ref_agree(ids):.1%}  CI {boot(ref_agree, ids)[0]:.1%}–{boot(ref_agree, ids)[1]:.1%}")
        print(f"text score Spearman    {spear(ids):.2f}  CI {boot(spear, ids)[0]:.2f}–{boot(spear, ids)[1]:.2f}")
        diff = {p: txt_b[p] - txt_a[p] for p in pairs(ids)}
        print(f"text score mean diff   {np.mean(list(diff.values())):+.3f} (compared minus reference)")
        for t in TIERS:
            tdiff = lambda s: float(np.mean([diff[i, t] for i in s]))
            lo, hi = boot(tdiff, ids)
            pa, pb = (np.mean([q[i, t] >= Q_TH for i in ids]) for q in (q_a, q_b))
            print(f"  {t:7s} mean text {np.mean([txt_a[i, t] for i in ids]):.3f} vs {np.mean([txt_b[i, t] for i in ids]):.3f}"
                  f"  diff {tdiff(ids):+.3f} CI {lo:+.3f}–{hi:+.3f}  Q-Sat {pa:.1%} vs {pb:.1%}"
                  f"  words {np.mean([words[i, t] for i in ids]):.0f}")
        w, d = [words[p] for p in pairs(ids)], [diff[p] for p in pairs(ids)]
        print(f"length bias: corr(words, diff) {np.corrcoef(w, d)[0, 1]:+.2f}  (positive = compared side favours longer answers)")
        order_same = np.mean([np.sign(txt_a[i, s] - txt_a[i, u]) == np.sign(txt_b[i, s] - txt_b[i, u])
                              for i in ids for s, u in (("m5_a", "pi_a"), ("m5_a", "jetson"), ("pi_a", "jetson"))])
        print(f"tier pair order same   {order_same:.1%}  (per image, 3 pairs, ties count only if both tie)")
        print("per criterion: quadratic-weighted kappa / exact match")
        for k in CRIT:
            a = [sub_a[i][t][k] for i, t in pairs(ids)]
            b = [sub_b[i][t][k] for i, t in pairs(ids)]
            print(f"  {k:26s} {kappa(a, b, weighted=True, k=6):.2f} / {np.mean(np.equal(a, b)):.1%}")

    htxt = [{(i, t): text_score(r[i][t]) for i in ids for t in TIERS} for r in raters]
    for n, (r, txt) in enumerate(zip(raters, htxt), 1):
        report(f"rater {n} ({Path(paths[n - 1]).name}) vs GPT-4o", r, txt, gpt_sub, gpt_txt)
    for a in range(len(raters)):
        for b in range(a + 1, len(raters)):
            report(f"rater {a + 1} vs rater {b + 1} (human ceiling)", raters[a], htxt[a], raters[b], htxt[b])
    scorers = {"GPT-4o": gpt_txt, **{f"rater {n}": t for n, t in enumerate(htxt, 1)}}
    if len(raters) >= 3:
        # per-criterion median of the raters is the human reference; median_low keeps integer scores
        med = {i: {t: {k: statistics.median_low(r[i][t][k] for r in raters) for k in CRIT} for t in TIERS} for i in ids}
        mtxt = {(i, t): text_score(med[i][t]) for i in ids for t in TIERS}
        report(f"median of {len(raters)} raters vs GPT-4o", med, mtxt, gpt_sub, gpt_txt)
        scorers["median"] = mtxt

    # Table I methods on these images under each scorer: does the comparison between methods hold?
    # Same tier choices and Confidence-Only rule (5, 2) as analysis/review_existing_data_audit.py.
    routing = {r["image_id"]: r for r in map(json.loads, open(ROUTING))
               if r.get("type") == "online_routing_final" and r.get("status") == "ok"}
    allt = {r["image_id"]: r for r in map(json.loads, open(ALL_TIER))}

    def conf_only(i):
        m = routing[i]["m5"]["m5_a"].get("conf") or 0
        p = ((routing[i].get("pi") or {}).get("pi_a") or allt[i].get("pi_a") or {}).get("conf") or 0
        return "m5_a" if m >= 5 else "pi_a" if p >= 2 else "jetson"

    choice = {"Fixed@M5": lambda i, txt: "m5_a", "Fixed@Pi": lambda i, txt: "pi_a",
              "Fixed@Jetson": lambda i, txt: "jetson", "Confidence-Only": lambda i, txt: conf_only(i),
              "Ours": lambda i, txt: routing[i]["route_label"],
              "Reference": lambda i, txt: reference_tier({t: quality(i, t, txt[i, t]) for t in TIERS}, cost[i])}
    print(f"\n=== Q-Sat (%) of Table I methods on these {len(ids)} images, by scorer (Reference is rebuilt per scorer)")
    print(" " * 16 + "".join(f"{s:>10s}" for s in scorers))
    for name, f in choice.items():
        print(f"{name:16s}" + "".join(
            f"{100 * np.mean([quality(i, f(i, txt), txt[i, f(i, txt)]) >= Q_TH for i in ids]):10.1f}"
            for txt in scorers.values()))


if __name__ == "__main__":
    main(sys.argv[1:])
