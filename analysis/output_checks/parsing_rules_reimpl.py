#!/usr/bin/env python3
"""R2 'exact definitions': re-implement the category/severity parsers from the reply-letter listing only.

    python3 -B analysis/output_checks/parsing_rules_reimpl.py [parsing_rules.tex]

Reads the four lstlisting blocks of parsing_rules.tex (M5; Pi category; Pi severity; Jetson), builds the rules
from the printed term lists, weights and regexes (nothing is imported from the device code), and replays them
on the stored responses: M5 profiling / earlier held-out / final run, Pi profiling / earlier held-out / final
run. Jetson raw responses were not retained, so only label validity is checked here (see parsing_rules_difftest.py).
Read-only. Prints match counts and every mismatch.
"""
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

M5VLM = Path(__file__).resolve().parents[2]
TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("parsing_rules.tex")  # the reply letter's listing
M5_BLOCK, PI_CAT_BLOCK, PI_SEV_BLOCK, JET_BLOCK = re.findall(
    r"\\begin\{lstlisting\}\[style=reviewprompt[^\]]*\]\n(.*?)\\end\{lstlisting\}", TEX.read_text(), re.S)


# ---------------- M5: substring DSL ({a, b} any-of, "a" term, & and, ! not) ----------------
def m5_cond(expr):
    expr = re.sub(r"\{([^}]*)\}", lambda m: "A(%r)" % [t.strip() for t in m.group(1).split(",")], expr)
    expr = re.sub(r'"([^"]*)"', lambda m: "A(%r)" % [m.group(1)], expr)
    return expr.replace("&", " and ").replace("!", " not ")


def m5_rules():
    lines = []
    for ln in M5_BLOCK.splitlines():  # a row continues while a { is open or the next line starts with &
        if lines and (ln.strip().startswith("&") or lines[-1].count("{") > lines[-1].count("}")):
            lines[-1] += " " + ln.strip()
        else:
            lines.append(ln)
    R, sec, step = {}, None, None
    for ln in lines:
        s = ln.strip()
        if s in ("CATEGORY", "SEVERITY"):
            sec = s
        elif s.startswith("Step "):
            step = s[5]
        elif s.startswith("Then, only if"):
            step = "weak"
        elif s.startswith("negated = "):
            R["negated"] = m5_cond(s[len("negated = "):])
        elif (m := re.match(r"^\s+(.+?)\s+:\s+(.+)$", ln)):
            R.setdefault((sec, step), []).append((m.group(1), m5_cond(m.group(2))))
    return R


M5R = m5_rules()


def m5_labels(text):
    low = text.lower()  # stored M5 texts are ASCII
    env = {"A": lambda ts: any(t in low for t in ts)}
    ev = lambda c: eval(c, env)  # noqa: E731
    cat = next((k for k, c in M5R["CATEGORY", "1"] if ev(c)), None)
    info = ""
    if cat == "landslide":
        (k, c), = M5R["CATEGORY", "2"]
        if ev(c):
            cat, info = k, "landslide->earthquake correction"
    elif cat is None:
        (k, c), = M5R["CATEGORY", "3"][:1]
        if ev(c):
            cat = k
        else:
            sc = dict(EQ=0, HU=0, FL=0, LA=0)
            for key, c in M5R["CATEGORY", "3"][1:]:
                name, w = key.split()
                sc[name] += int(w) * ev(c)
            if sc["HU"] < 3 and sc["FL"] < 3 and sc["LA"] < 3:
                for key, c in M5R["CATEGORY", "weak"]:
                    name, w = key.split()
                    sc[name] += int(w) * ev(c)
            best = max(sc.values())
            if best < 2:
                cat = "other_disaster"
            else:
                order = [("earthquake", "EQ"), ("flood", "FL"), ("hurricane", "HU"), ("landslide", "LA")]
                tied = [c for c, n in order if sc[n] == best]
                cat, info = tied[0], ("score tie " + "/".join(tied)) if len(tied) > 1 else ""
    env["negated"] = ev(M5R["negated"])
    sev = next((int(v) for v, c in M5R["SEVERITY", "1"] if ev(c)), None)
    if sev is None:
        score = sum(int(w) * ev(c) for w, c in M5R["SEVERITY", "2"])
        sev = 2 if score >= 3 else 1 if score == 2 else 0
    return cat, sev, info


# ---------------- Pi: regex rows ----------------
def join_wrapped(block):
    out = []
    for ln in block.splitlines():
        if out and ln[:1] == " ":  # indented line continues the row; a break after | is not part of the pattern
            out[-1] += ("" if out[-1].endswith("|") else " ") + ln.strip()
        else:
            out.append(ln)
    return [l for l in out if l.strip()]


def pi_cond(tokens, names):
    """tokens: PATTERN [AND [NOT] PATTERN]...  ->  list of (negate, compiled regex)."""
    out, neg, expect = [], False, True
    for tok in tokens:
        if tok == "AND":
            expect = True
        elif tok == "NOT":
            neg = True
        else:
            assert expect, tokens
            out.append((neg, re.compile(names.get(tok, tok))))
            neg, expect = False, False
    return out


def pi_rules():
    names, rows, fallback, sec = {}, [], [], None
    for ln in join_wrapped(PI_CAT_BLOCK):
        if ln.startswith("CATEGORY ROWS"):
            sec = rows
        elif ln.startswith("FALLBACK ROWS"):
            sec = fallback
        elif (m := re.match(r"^(\w+) = (\S+)$", ln)):
            names[m.group(1)] = m.group(2)
        elif sec is rows:
            t = ln.split()
            rows.append((t[0], int(t[1]), pi_cond(t[2:], names)))
        elif sec is fallback:
            t = ln.split()
            fallback.append((t[0], [] if t[1:] == ["(always)"] else pi_cond(t[1:], names)))
    sev = {}
    for ln in join_wrapped(PI_SEV_BLOCK):
        if (m := re.match(r"^(S1|S2|S3|L)\s+= (.+)$", ln)):
            sev[m.group(1)] = pi_cond(m.group(2).split(), {})
    return rows, fallback, sev


PI_ROWS, PI_FALLBACK, PI_SEV = pi_rules()
match = lambda conds, t: all(bool(rx.search(t)) != neg for neg, rx in conds)  # noqa: E731


def pi_labels(text):
    t = text.lower()
    scores = OrderedDict()  # insertion order = order of first points
    for cat, w, conds in PI_ROWS:
        if match(conds, t):
            scores[cat] = scores.get(cat, 0) + w
    info = ""
    if scores:
        best = max(scores.values())
        tied = [c for c, v in scores.items() if v == best]
        cat = tied[0]
        info = ("tie " + "/".join(tied)) if len(tied) > 1 else ""
    else:
        cat = next(c for c, conds in PI_FALLBACK if match(conds, t))
        info = "fallback " + cat
    S = {k: match(v, t) for k, v in PI_SEV.items()}
    sev = 2 if (S["S1"] or S["S2"] or S["S3"]) else 0 if S["L"] else 1
    return cat, sev, info


# ---------------- Jetson: label lines ----------------
JET_LABEL = re.search(r"^label = (\S+)", JET_BLOCK, re.M).group(1)
JET_CATS = set(re.search(r"v in \{([^}]*)\}", JET_BLOCK).group(1).replace(" ", "").split(","))


def jetson_labels(raw):
    cat, sev = "other_disaster", 1
    for ln in (x.strip() for x in raw.splitlines()):
        m = ln and re.match(JET_LABEL, ln, re.IGNORECASE)
        if not m:
            continue
        label, value = m.group(1).lower(), m.group(2).strip().strip("*").strip()
        if label == "cat":
            v = value.lower().split()[0].strip(".,:;")  # IndexError when value is empty, as documented
            if v in JET_CATS:
                cat = v
        elif label == "sev":
            d = next((ch for ch in value if ch in "012"), None)
            if d is not None:
                sev = int(d)
    return cat, sev


# ---------------- replay on stored labels ----------------
if __name__ == "__main__":
    load = lambda p: [json.loads(l) for l in open(p) if l.strip()]  # noqa: E731
    PROF = load(M5VLM / "profiling/results/collection.jsonl")
    EARLIER = load(M5VLM / "heldout/results/collection.jsonl")
    FINAL = load(M5VLM / "deployment/live_run/online_routing_final.jsonl")
    C, cov, bad = Counter(), Counter(), []
    sets = [("profiling", PROF, lambda r, k: r.get(k)), ("earlier held-out", EARLIER, lambda r, k: r.get(k)),
            ("final run", FINAL, lambda r, k: (r["m5"] if k.startswith("m5") else (r.get("pi") or {})).get(k))]
    for name, rows, get in sets:
        for ag in ("m5_a", "m5_b", "m5_c", "pi_a", "pi_b"):
            fn = m5_labels if ag.startswith("m5") else pi_labels
            for r in rows:
                o = get(r, ag)
                if not o or o.get("text") is None:
                    continue
                cat, sev, info = fn(o["text"])
                ok = (cat, sev) == (o["cat"], o["sev"])
                C[f"{name:16s} {ag}", ok] += 1
                if info:
                    cov[f"{'M5' if ag.startswith('m5') else 'Pi'}: {info}"] += 1
                if not ok:
                    bad.append((name, ag, r["image_id"], (cat, sev), (o["cat"], o["sev"]), o["text"]))
    for name, recs in (("profiling", [r["jetson"] for r in PROF]), ("earlier held-out", [r["jetson"] for r in EARLIER]),
                       ("final run", [r["jetson"]["jetson"] for r in FINAL if r.get("jetson")])):
        for o in recs:
            C[f"{name:16s} jetson (label valid only)", o["cat"] in JET_CATS and o["sev"] in (0, 1, 2)] += 1
    for k in sorted({k for k, _ in C}):
        print(f"{k:45s} {C[k, True]:6d} / {C[k, True] + C[k, False]:6d}")
    print("\nrule-branch coverage on the replayed responses:")
    for k, v in sorted(cov.items()):
        print(f"  {k:45s} {v:6d}")
    print(f"\nmismatches: {len(bad)}")
    for b in bad:
        print(" ", b)
    assert not bad
