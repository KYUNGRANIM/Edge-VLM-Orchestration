"""Leave-one-event-out evaluation of the routing policy on MEDIC.

Pools the 5,526 profiling and 600 held-out images (all tiers were executed on
both), holds out each disaster event with >= MIN_EVENT images, retrains both
stages on everything else with the deployed recipe (depth-4 trees, Stage 2 on
non-Tier-1 references), and replays the held-out event from the stored tier
outputs. Images without a recoverable event are used for training only. The
same images are also scored under random stratified 5-fold CV to show how much
of the result relies on having seen the event.

Usage: python3 analysis/event_split.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

M5VLM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M5VLM / "profiling" / "code" / "pipeline"))
from common import make_dt, stage1_features, stage2_features, to_matrix  # noqa: E402

HELD = M5VLM / "heldout/results/labels"
SETS = [  # (collection with all tier outputs, oracle labels, cost components, MEDIC manifest)
    (M5VLM / "profiling/results/collection.jsonl",
     M5VLM / "profiling/results/pipeline_outputs/03_oracle_labels.jsonl",
     M5VLM / "profiling/results/pipeline_outputs/02_cost_components.jsonl",
     M5VLM / "profiling/results/medic_manifest.csv"),
    (M5VLM / "heldout/results/collection.jsonl",
     HELD / "03_oracle_labels.jsonl", HELD / "02_cost_components.jsonl",
     M5VLM / "heldout/results/medic_manifest_test.csv"),
]
MIN_EVENT = 100
Q_TH = 0.65
TIERS = ("m5_a", "pi_a", "jetson")
# Same auxiliary-device accounting as the first submission's Table I.
AUX_M5_ENERGY = 2 * 1.2154 * 6.08
AUX_PI_ENERGY = 7.5564 * 31.35
AUX_PI_OVERHEAD = 218.7 * 1024

# Source folders that are crawls or generic searches rather than one event.
GENERIC = {"natural_disaster", "landslide_crawled_images_v2", "human-induced_disaster",
           "security-incidents-activities", "ggImage"}
ALIASES = {"nepal_earthquake": "nepal_eq", "harvey": "hurricane_harvey",
           "irma": "hurricane_irma", "maria": "hurricane_maria"}


def event_of(path: str):
    parts = path.split("/")
    if "crisismmd" in parts:
        return parts[parts.index("data_image") + 1]
    for root in ("ASONAM17_Damage_Image_Dataset", "aidr_disaster_types", "aidr_info"):
        if root in parts:
            e = parts[parts.index(root) + 1]
            return None if e in GENERIC else ALIASES.get(e, e)
    return None  # Mouzannar multimodal set carries no event


def jl(path, row_type=None):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if row_type is None or r.get("type") == row_type:
            out[r["image_id"]] = r
    return out


def load():
    recs, orc, cst, events, cats = [], [], [], [], []
    for coll_p, orc_p, cst_p, man_p in SETS:
        coll, o, c = jl(coll_p), jl(orc_p, "oracle_label"), jl(cst_p, "cost_components")
        # The held-out manifest lives on the CORSAIR drive. Without it those images are left out entirely: their
        # events are real but unknown, so training on them could leak a held-out event.
        if not man_p.exists():
            print(f"no manifest at {man_p}: its {len(set(coll) & set(o) & set(c))} images are left out")
            continue
        man = {Path(r["local_image_path"]).name: r for r in csv.DictReader(open(man_p))}
        for i in sorted(set(coll) & set(o) & set(c)):
            recs.append(coll[i]); orc.append(o[i]); cst.append(c[i])
            m = man.get(i)
            events.append(event_of(m["image_path"]) if m else None); cats.append(m["group"] if m else None)
    return recs, orc, cst, events, cats


def fit_predict(x1, x2, c1, c2, y, tr, te):
    y1 = np.array([int(v == "m5_a") for v in y])
    y2 = np.array([int(v == "pi_a") for v in y])
    m1 = make_dt(c1[0], c1[1], depth=4).fit([x1[i] for i in tr], y1[tr])
    tr2 = [i for i in tr if y[i] != "m5_a"]
    m2 = make_dt(c2[0], c2[1], depth=4).fit([x2[i] for i in tr2], y2[tr2])
    s1 = m1.predict([x1[i] for i in te]).astype(int)
    s2 = m2.predict([x2[i] for i in te]).astype(int)
    return {g: ("m5_a" if a == 1 else ("pi_a" if b == 1 else "jetson")) for g, a, b in zip(te, s1, s2)}


def summary(idx, tiers, orc, cst, aux):
    q, lat, eng = [], [], []
    for i in idx:
        t = tiers[i] if isinstance(tiers, dict) else tiers
        c = cst[i][t]
        e = c["energy_j"] + ((AUX_M5_ENERGY + (AUX_PI_ENERGY if t != "m5_a" else 0)) if aux else 0)
        q.append(orc[i]["quality"][t] >= Q_TH); lat.append(c["latency_ms"] / 1000); eng.append(e)
    return 100 * np.mean(q), np.mean(lat), np.mean(eng)


def main():
    recs, orc, cst, events, cats = load()
    y = np.array([o["oracle_label"] for o in orc], dtype=object)
    x1, _, c1i, n1i = to_matrix([stage1_features(r) for r in recs])
    x2, _, c2i, n2i = to_matrix([stage2_features(r) for r in recs])
    c1, c2 = (c1i, n1i), (c2i, n2i)

    counts = Counter(e for e in events if e)
    held_events = [e for e, n in counts.most_common() if n >= MIN_EVENT]
    loeo = {}
    for ev in held_events:
        te = [i for i, e in enumerate(events) if e == ev]
        tr = [i for i, e in enumerate(events) if e != ev]
        loeo.update(fit_predict(x1, x2, c1, c2, y, tr, te))

    rand = {}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y):
        rand.update(fit_predict(x1, x2, c1, c2, y, tr, te))

    print(f"pooled N={len(recs)}, unknown event {sum(e is None for e in events)} (training only)")
    print(f"{'held-out event':24s} {'n':>5s} | {'LOEO':>18s} | {'random CV':>18s} | {'Fixed@Pi':>18s}")
    rows = [(ev, [i for i, e in enumerate(events) if e == ev]) for ev in held_events]
    rows.append(("ALL held-out events", [i for ev, idx in rows for i in idx]))
    for ev, idx in rows:
        a = summary(idx, loeo, orc, cst, True)
        b = summary(idx, rand, orc, cst, True)
        f = summary(idx, "pi_a", orc, cst, False)
        fmt = lambda s: f"{s[0]:5.1f}% {s[1]:5.1f}s {s[2]:4.0f}J"
        print(f"{ev:24s} {len(idx):5d} | {fmt(a)} | {fmt(b)} | {fmt(f)}")

    idx = rows[-1][1]
    o = np.array([orc[i]["quality"][loeo[i]] >= Q_TH for i in idx], float)
    p = np.array([orc[i]["quality"]["pi_a"] >= Q_TH for i in idx], float)
    boot = np.random.default_rng(0).integers(0, len(idx), (10000, len(idx)))
    d = (o[boot].mean(1) - p[boot].mean(1)) * 100
    print(f"LOEO vs Fixed@Pi on all held-out events: {100 * (o.mean() - p.mean()):+.1f} pp, "
          f"95% CI [{np.percentile(d, 2.5):.1f}, {np.percentile(d, 97.5):.1f}], one-sided lower {np.percentile(d, 5):.1f}")
    print("held-out images by category:", dict(Counter(cats[i] for i in idx)))


if __name__ == "__main__":
    main()
