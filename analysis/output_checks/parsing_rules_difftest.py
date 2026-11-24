#!/usr/bin/env python3
"""R2 'exact definitions': differential test of the listing-based parsers against the deployed code.

    python3 -B analysis/output_checks/parsing_rules_difftest.py [parsing_rules.tex]

The stored labels (parsing_rules_reimpl.py) leave some rules unexercised and Jetson raw responses were not kept,
so this compares the listing-based re-implementation with the deployed parsers on many more inputs:
  M5     : firmware classify_from_text, extracted verbatim from deployment/m5_firmware/tier1.cpp and
           compiled on the host with a minimal Arduino-String shim (indexOf / toLowerCase / ==).
  Pi     : classify_cat / classify_sev of deployment/servers/pi_a_inference_engine.py (Pi B and the
           collection copies are AST-identical).
  Jetson : parse_output of deployment/servers/jetson_inference_engine.py.
Inputs: every stored MEDIC response text (profiling / earlier held-out / final run, all agents) plus seeded random
texts built from the listed terms; random raw label blocks for Jetson. Read-only; temp files only.
"""
import json
import random
import re
import runpy
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import parsing_rules_reimpl as S  # noqa: E402  (listing-based; reads parsing_rules.tex)

M5VLM = S.M5VLM
rng = random.Random(20261120)

# ---------------- reference implementations ----------------
SHIM = r"""
#include <string>
#include <cctype>
#include <iostream>
struct String {
  std::string s;
  String(const char* c = "") : s(c) {}
  int indexOf(const char* t) const { size_t p = s.find(t); return p == std::string::npos ? -1 : (int)p; }
  void toLowerCase() { for (auto& c : s) c = (char)std::tolower((unsigned char)c); }
  bool operator==(const char* o) const { return s == o; }
};
"""
MAIN = r"""
int main() {
  std::string rec;
  while (std::getline(std::cin, rec, '\0')) {
    String t; t.s = rec; String cat; int sev = -1;
    classify_from_text(t, cat, sev);
    if (sev > 2) sev = 2;
    if (sev < 0) sev = 0;
    std::cout << cat.s << '\t' << sev << '\n';
  }
}
"""


def firmware_labels(texts):
    fw = (M5VLM / "deployment/m5_firmware/tier1.cpp").read_text(encoding="utf-8")
    i = fw.index("static void classify_from_text(const String& text")
    fn = fw[i:fw.index("static float compute_fine_score", i)]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "m5.cpp").write_text(SHIM + fn + MAIN, encoding="utf-8")
        subprocess.run(["c++", "-std=c++17", "-O1", "-o", f"{d}/m5", f"{d}/m5.cpp"], check=True)
        out = subprocess.run([f"{d}/m5"], input="\0".join(texts) + "\0", capture_output=True, text=True, check=True).stdout
    return [(c, int(s)) for c, s in (ln.split("\t") for ln in out.splitlines())]


PI = runpy.run_path(str(M5VLM / "deployment/servers/pi_a_inference_engine.py"), run_name="difftest")
JET = runpy.run_path(str(M5VLM / "deployment/servers/jetson_inference_engine.py"), run_name="difftest")


def outcome(fn, x):
    try:
        return fn(x)
    except Exception as e:  # the documented empty-cat-value crash must be reproduced too
        return type(e).__name__


# ---------------- inputs ----------------
def stored_texts():
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "text" and isinstance(v, str):
                    out.add(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for p in ["profiling/results/collection.jsonl", "heldout/results/collection.jsonl",
              "deployment/live_run/online_routing_final.jsonl"]:
        for ln in open(M5VLM / p):
            if ln.strip():
                walk(json.loads(ln))
    return sorted(out)


def vocabulary():
    words = set()
    for blk in (S.M5_BLOCK,):
        for grp in re.findall(r"\{([^}]*)\}", blk):
            words.update(t.strip() for t in grp.split(","))
        words.update(re.findall(r'"([^"]*)"', blk))
    for blk in (S.PI_CAT_BLOCK, S.PI_SEV_BLOCK):
        for pat in re.findall(r"\\b\S*\\b", "\n".join(S.join_wrapped(blk))):
            p = re.sub(r"\\s[+*]", " ", pat)
            p = re.sub(r"\\b|\(\?:|\)|\?|\\w\*", "", p)
            words.update(w.strip() for w in p.split("|") if w.strip())
    words.discard("")
    return sorted(words)


FILL = ["the", "a", "area", "with", "and", "near", "of", "is", "not", "no", "very", "large", "people", "car", "street", ".",
        ",", "severe", "moderate", "visible", "damage", "water", "trees", "roof", "building"]


def synth_texts(n, vocab, ascii_only):
    out = []
    for _ in range(n):
        toks = []
        for _ in range(rng.randint(1, 12)):
            w = rng.choice(vocab) if rng.random() < 0.7 else rng.choice(FILL)
            w += rng.choice(["", "", "", "s", "ed", "ing", "ly", "-", "."])
            if rng.random() < 0.05:
                w = w.upper()
            toks.append(w)
        t = rng.choice([" ", "  ", " ", "\t"]).join(toks)
        if not ascii_only and rng.random() < 0.05:
            t += " caf\u00e9 \u0130stanbul"
        out.append(t)
    return out


def synth_jetson(n):
    labels = ["cat", "CAT", "Cat", "sev", "Sev", "SEV", "damage", "countermeasure", "category", "severity", " cat", "- cat", "**cat**", "cat sev"]
    seps = [":", "\uff1a", " :", ": ", ":  ", "::", ":\t", ""]
    cats = ["fire", "flood", "hurricane", "landslide", "earthquake", "other_disaster", "other disaster", "Flood.", "*flood*",
            "**Earthquake**", "fire,", "hurricane damage", "none", "landslide;", "Wildfire", "*", "**", ";flood", "fire:", "x"]
    sevs = ["0", "1", "2", "3", "10", "2/2", "severe", "level 2", "*1*", "-1", "two", "**2**", "(1)", " 0 "]
    rows = []
    for _ in range(n):
        lines = []
        for _ in range(rng.randint(0, 7)):
            lab = rng.choice(labels)
            if rng.random() < 0.15:
                lines.append(rng.choice(["", "   ", "A collapsed building.", "Clear the road now please."]))
                continue
            val = rng.choice(cats) if "cat" in lab.lower() else rng.choice(sevs) if "sev" in lab.lower() else "Some text here."
            lines.append(rng.choice(["", " ", "  "]) + lab + rng.choice(seps) + rng.choice(["", " "]) + val + rng.choice(["", " ", "*"]))
        rows.append(rng.choice(["\n", "\r\n", "\n\n"]).join(lines))
    return rows


# ---------------- compare ----------------
if __name__ == "__main__":
    C, bad = Counter(), {}

    def cmp(name, xs, ref, mine):
        for x, a, b in zip(xs, ref, mine):
            C[name, a == b] += 1
            if a != b:
                bad.setdefault(name, []).append((x, a, b))

    stored, vocab = stored_texts(), vocabulary()
    ascii_stored = [t for t in stored if t.isascii() and "\0" not in t]
    m5_inputs = [("M5 vs firmware: stored texts (ASCII)", ascii_stored), ("M5 vs firmware: synthetic", synth_texts(60000, vocab, True))]
    for name, xs in m5_inputs:
        cmp(name, xs, firmware_labels(xs), [S.m5_labels(x)[:2] for x in xs])
    for name, xs in (("Pi vs engine: stored texts", stored), ("Pi vs engine: synthetic", synth_texts(60000, vocab, False))):
        cmp(name, xs, [(PI["classify_cat"](x), PI["classify_sev"](x)) for x in xs], [S.pi_labels(x)[:2] for x in xs])
    js = synth_jetson(30000)
    cmp("Jetson vs engine: synthetic raw label blocks", js,
        [outcome(lambda r: (lambda d: (d["cat"], d["sev"]))(JET["parse_output"](r)), x) for x in js],
        [outcome(S.jetson_labels, x) for x in js])
    C["Jetson synthetic cases that raise (both)", True] = sum(isinstance(outcome(S.jetson_labels, x), str) for x in js)

    print(f"stored distinct texts: {len(stored)} (ASCII: {len(ascii_stored)}); vocabulary terms: {len(vocab)}")
    for k in sorted({k for k, _ in C}):
        print(f"{k:50s} {C[k, True]:7d} / {C[k, True] + C[k, False]:7d}")
    for name, rows in bad.items():
        print(f"\n{name}: {len(rows)} mismatches, first 10:")
        for r in rows[:10]:
            print("  ", r)
    assert not bad
