#!/usr/bin/env python3
"""
Jetson Oracle Builder v60 (single-pass + conditional repair)
============================================================
Final hybrid combining the strengths of v58_codex (visual-detail damage
descriptions, strong other_disaster cat) with v58/v59 (CM imperative-form
repair safety net, refined hurricane/landslide/earthquake cat definitions).

From v58_codex:
  - damage prompt: explicit visual-detail list (material, color, count,
    structure, position, water, smoke, flames, debris, people)
  - damage anti-hallucination list (no inferred injuries/deaths/trapped
    people/causes; no damage to vehicles/boats/aircraft unless visible)
  - other_disaster cat guidance (responders, shelters, equipment,
    aftermath without a clear visible hazard from the named categories)
  - countermeasure target rule (no actions on undamaged responders/equipment)

From v58:
  - Stronger hurricane/landslide/earthquake cat definitions
  - is_should_form / is_past_tense_cm safety nets in needs_repair()
  - Imperative-focused REPAIR_SYSTEM_PROMPT

Kept overall:
  - Four labeled output lines (damage, countermeasure, cat, sev)
  - strip_meta_preamble with purpose-clause + adverb-opener + label-leak
  - is_prompt_leak safety net
  - Combined logprobs across main + repair call (Pi-identical)
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


# ------------------------------- Defaults ------------------------------- 

DEFAULT_COLLECTED = "tier3_earthquake/pi_earthquake.jsonl"  #flood
DEFAULT_OUT       = "tier3_earthquake/jetson_earthquake.jsonl"
DEFAULT_RUN_ID    = "pilot_300_run1"

DEFAULT_LIMIT         = 0
DEFAULT_RESUME        = True
DEFAULT_MAX_IMAGE_DIM = 768
RETRY_DIMS            = [640, 512, 384]

VLLM_URL        = os.getenv("VLLM_URL",   "http://127.0.0.1:9000/v1/chat/completions")
VLLM_MODEL      = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
VLM_TIMEOUT     = int(os.getenv("TIMEOUT", "300"))
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.3"))
VLM_MAX_TOKENS        = int(os.getenv("VLM_MAX_TOKENS",        "220"))
VLM_REPAIR_MAX_TOKENS = int(os.getenv("VLM_REPAIR_MAX_TOKENS", "80"))
KST             = timezone(timedelta(hours=9))

VALID_CATS = {"fire", "flood", "hurricane", "landslide", "earthquake", "other_disaster"}
VALID_SEVS = {0, 1, 2}


# ------------------------------- Main prompt (4 labeled lines) -------------------------------
SYSTEM_PROMPT = (
    "You are a disaster response specialist writing a concise field report\n"
    "from visual evidence only. Look carefully at one disaster photograph.\n"
    "Output exactly four labeled lines.\n"
    "\n"
    "damage: one natural English sentence describing the visible disaster\n"
    "  situation. Include concrete visual details such as material, color,\n"
    "  count, structure, position, water, smoke, flames, debris, or people.\n"
    "  Describe only what is plainly visible. Do not infer hidden injuries,\n"
    "  deaths, trapped people, causes, or damage to vehicles, boats,\n"
    "  aircraft, or equipment unless clearly visible. Do not name a building's\n"
    "  function unless signage confirms it. Match the full visible scene, not\n"
    "  just a foreground fragment.\n"
    "\n"
    "countermeasure: one natural English sentence prescribing one\n"
    "  concrete first-response action. Begin with a specific action verb\n"
    "  (e.g., Deploy, Clear, Pump, Cordon, Search, Extinguish, Shore,\n"
    "  Evacuate, Remove, Stabilize, Cover, Seal, Drain, Reroute). Name\n"
    "  the specific target visible in the damage line (the collapsed\n"
    "  structure, the flooded street, the burning building, the fallen\n"
    "  tree). Do not start with 'Immediate countermeasure', 'Immediate\n"
    "  action', 'Immediate response', or any label prefix. Do not use\n"
    "  generic safety phrases such as 'ensure safety', 'prevent further\n"
    "  damage', 'evacuate to a safer place', or 'secure the area' as the\n"
    "  whole sentence — bind the action to a specific visible target.\n"
    "  Avoid 'should', 'must', and past-tense narration.\n"
    "\n"
    "cat: choose one based on the dominant visible evidence:\n"
    "  fire — visible flames, active smoke, or charred surfaces.\n"
    "  flood — pooled or flowing water, submersion, or waterlines on\n"
    "          structures.\n"
    "  hurricane — wind-torn roofs, snapped or uprooted trees, laterally\n"
    "              blown debris. Storm-surge flooding may be present;\n"
    "              classify hurricane when wind damage is the dominant\n"
    "              visible signal.\n"
    "  landslide — hillside earth, mud, or rocks displaced downslope;\n"
    "              hillside debris burying a road or structure at a\n"
    "              slope base.\n"
    "  earthquake — cracked, leaning, tilted, pancaked, or collapsed\n"
    "               masonry; structural collapse without active fire or\n"
    "               dominant water.\n"
    "  other_disaster — when the image mainly shows people, shelters,\n"
    "                   responders, equipment, vehicles, or aftermath\n"
    "                   without a clear visible hazard from the categories\n"
    "                   above.\n"
    "\n"
    "sev: 0 = minor or superficial damage; 1 = one section seriously damaged;\n"
    "  2 = widespread or total destruction.\n"
    "\n"
    "No markdown. No blank lines. Output:\n"
    "damage: ...\n"
    "countermeasure: ...\n"
    "cat: ...\n"
    "sev: ..."
)

USER_PROMPT = "Write the 4-line disaster field report for this photograph."


# ------------------------------- Repair prompt (countermeasure only) -------------------------------
REPAIR_SYSTEM_PROMPT = (
    "You are a disaster response specialist looking at a disaster photograph.\n"
    "Write one concrete first-response action as an imperative sentence.\n"
    "\n"
    "Begin with a specific action verb (Deploy, Clear, Pump, Cordon,\n"
    "Search, Extinguish, Shore, Evacuate, Remove, Stabilize, Cover, Seal,\n"
    "Drain, Reroute). Name the specific visible target from the damage\n"
    "description.\n"
    "\n"
    "Do not start with 'Immediate countermeasure', 'Immediate action',\n"
    "'Immediate response', or any label prefix. Do not use generic safety\n"
    "phrases ('ensure safety', 'prevent further damage', 'evacuate to a\n"
    "safer place', 'secure the area') as the whole sentence. Avoid\n"
    "'should', 'must', and past-tense narration.\n"
    "\n"
    "Output one sentence only. No labels. No markdown."
)


def repair_user_prompt(damage: str) -> str:
    d = damage.strip() or "(no damage description provided)"
    return f'Damage: "{d}"\n\nWrite one countermeasure sentence for this photograph.'


# ------------------------------- IO helpers -------------------------------
def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="milliseconds")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def backup_existing_output(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.pre_run_{stamp}.bak")
    shutil.copy2(path, backup)
    print(f"[BACKUP] existing out copied to {backup}")


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("status") == "ok" and obj.get("image_id"):
                done.add(obj["image_id"])
    return done


def resolve_image(rec: dict, collected_dir: Path) -> Path | None:
    image_id = str(rec.get("image_id", ""))
    saved    = str(rec.get("saved_image", ""))
    for p in (Path(saved), collected_dir / "images" / image_id, collected_dir / saved):
        if p.is_file():
            return p
    return None


def b64_image(path: Path, max_dim: int) -> str:
    img = Image.open(path)
    w, h = img.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ------------------------------- Power sampling -------------------------------
def _find_power_node() -> str | None:
    patterns = [
        "/sys/bus/i2c/devices/*/hwmon/hwmon*/power1_input",
        "/sys/bus/i2c/devices/*/ina3221*/hwmon/hwmon*/power1_input",
        "/sys/devices/platform/*/i2c-*/*/hwmon/hwmon*/power1_input",
        "/sys/class/hwmon/hwmon*/power1_input",
    ]
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return sorted(hits)[0]
    return None


POWER_NODE = _find_power_node()


def _read_power_mw() -> float | None:
    if not POWER_NODE:
        return None
    try:
        with open(POWER_NODE, "r", encoding="utf-8") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def sample_power_during(fn, *args, **kwargs):
    samples: list[float] = []
    done = threading.Event()

    def _sampler():
        while not done.is_set():
            v = _read_power_mw()
            if v is not None:
                samples.append(v)
            time.sleep(0.2)

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        done.set()
        t.join(timeout=1.0)

    if samples:
        avg_mw   = round(sum(samples) / len(samples), 2)
        energy_j = round(avg_mw * dt / 1000.0, 4)
    else:
        avg_mw   = None
        energy_j = None
    return result, energy_j, avg_mw


# ------------------------------- vLLM -------------------------------
def call_vllm(system: str, user_text: str, img_b64: str,
              max_tokens: int) -> tuple[str, int, list | None]:
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    payload = {
        "model":        VLLM_MODEL,
        "messages":     messages,
        "max_tokens":   max_tokens,
        "temperature":  VLM_TEMPERATURE,
        "logprobs":     True,
        "top_logprobs": 1,
    }
    t0 = time.perf_counter()
    r  = requests.post(VLLM_URL, json=payload, timeout=VLM_TIMEOUT)
    ms = int((time.perf_counter() - t0) * 1000)
    r.raise_for_status()
    data   = r.json()
    choice = data["choices"][0]
    text   = (choice.get("message") or {}).get("content", "") or ""
    lp_obj = choice.get("logprobs") or {}
    logprobs = lp_obj.get("content") if isinstance(lp_obj, dict) else None
    return text, ms, logprobs


# ------------------------------- Text helpers -------------------------------
def _clean(s: str) -> str:
    s = (s or "").replace("**", "")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def _first_sentence(s: str) -> str:
    s = _clean(s)
    m = re.match(r"(.+?[.!?])(\s|$)", s)
    return (m.group(1) if m else s).strip()


def _finalize_sentence(s: str) -> str:
    s = _clean(s)
    if not s:
        return s
    if s[0].islower():
        s = s[0].upper() + s[1:]
    if s[-1] not in ".!?":
        s += "."
    return s


# ------------------------------- strip_meta_preamble -------------------------------
_META_PREAMBLE_PATTERNS: list[re.Pattern] = [
    re.compile(p, flags=re.IGNORECASE)
    for p in [
        r"^\s*the\s+immediate\s+countermeasures?\s+should\s+involve\s+",
        r"^\s*the\s+immediate\s+countermeasure\s+should\s+be\s+to\s+",
        r"^\s*the\s+immediate\s+countermeasure\s+is\s+to\s+",
        r"^\s*immediate\s+countermeasures?\s+should\s+involve\s+",
        r"^\s*immediate\s+countermeasures?\s+should\s+include\s+",
        r"^\s*immediate\s+action\s+should\s+be\s+taken\s+to\s+",
        r"^\s*immediate\s+action\s+is\s+required\s+to\s+",
        r"^\s*immediate\s+action\s+must\s+be\s+taken\s+to\s+",
        r"^\s*the\s+first\s+action\s+should\s+be\s+to\s+",
        r"^\s*the\s+first\s+action\s+is\s+to\s+",
        r"^\s*the\s+first\s+step\s+(?:would|should)\s+be\s+to\s+",
        r"^\s*the\s+first\s+step\s+is\s+to\s+",
        r"^\s*the\s+next\s+step\s+(?:should|would)\s+be\s+to\s+",
        r"^\s*the\s+next\s+step\s+is\s+to\s+",
        r"^\s*the\s+most\s+appropriate\s+action\s+(?:would\s+be|is)\s+to\s+",
        r"^\s*the\s+countermeasure\s+should\s+be\s+to\s+",
        r"^\s*countermeasures?\s+should\s+immediately\s+include\s+",
        r"^\s*a\s+temporary\s+countermeasure\s+(?:would\s+involve|would\s+be|is|should\s+be)\s+to\s+",
        r"^\s*temporary\s+countermeasures?\s+(?:would\s+involve|should\s+(?:involve|include|be\s+to))\s+",
        r"^\s*a\s+countermeasure\s+(?:would\s+be|is|should\s+be)\s+to\s+",
        r"^\s*one\s+(?:immediate|appropriate)\s+action\s+(?:would\s+be|is|should\s+be)\s+to\s+",
        r"^\s*immediate\s+actions?\s+(?:would\s+involve|should\s+involve|required\s+to)\s+",
        r"^\s*immediate\s+actions?\s+to\s+(?:secure|address|stabilize|prevent|reinforce)\s+",
        # Purpose-clause opener.
        r"^\s*to\s+prevent\s+further\s+(?:damage|collapse|harm|injury|flooding|erosion)\s*,\s*",
        r"^\s*to\s+ensure\s+(?:safety|structural|the\s+safety)\s*",
        # Label-leak: Imperative:/Declarative:/Counter:/Action:/Immediate action:
        r"^\s*(?:imperative|declarative|countermeasure|counter|action)\s*[:：\-\–]\s*",
        r"^\s*immediate\s+action\s*[:：]\s*",
        r"^\s*immediate\s+countermeasures?\s*[:：]\s*",
        r"^\s*immediate\s+response\s*[:：]\s*",
        r"^\s*immediate\s+first[\-\s]?response(?:\s+action)?s?\s*[:：]\s*",
        r"^\s*first[\-\s]?response\s+actions?\s*[:：]\s*",
        # Note: complex "Immediate X should involve Y-ing" patterns are NOT
        # stripped here — stripping them creates -ing fragments that fail QC.
        # Instead, those are caught by is_should_form → triggers repair.
        # Bare adverb openers before an imperative.
        r"^\s*(?:immediately|urgently|quickly|promptly)\s*[,]?\s+",
        # Ordering openers — require comma to preserve "First responders".
        r"^\s*(?:first|firstly|initially)\s*,\s+",
        r"^\s*first\s+of\s+all\s*,?\s+",
    ]
]


def strip_meta_preamble(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    for pat in _META_PREAMBLE_PATTERNS:
        m = pat.match(s)
        if m:
            rest = s[m.end():].lstrip()
            if rest:
                rest = rest[0].upper() + rest[1:]
                return rest
            return s
    return s


# ------------------------------- QC detectors -------------------------------
_DESC_FORM_RE = re.compile(r"\b(is|are)\s+\w+ing\b", re.IGNORECASE)

_SIMPLE_PRES_NARR_VERBS = {
    "is", "are", "was", "were",
    "uses", "takes", "operates", "holds", "applies", "performs",
    "places", "moves", "pulls", "pushes", "lifts", "carries",
    "searches", "sifts", "climbs", "walks", "stands", "sits",
    "drives", "digs", "clears", "checks", "examines", "assesses",
}
_NARRATIVE_MODAL_EXEMPTIONS = {"should", "must", "need", "needs"}


def is_desc_form(s: str) -> bool:
    if not s:
        return False
    head = " ".join(s.split()[:8])
    return bool(_DESC_FORM_RE.search(head))


def is_simple_pres_narrative(s: str) -> bool:
    if not s:
        return False
    toks = [t.lower().rstrip(".,;:!?") for t in s.split()[:10]]
    if any(m in toks for m in _NARRATIVE_MODAL_EXEMPTIONS):
        return False
    return any(v in toks for v in _SIMPLE_PRES_NARR_VERBS)


def is_fragment(s: str) -> bool:
    if not s:
        return False
    first = s.strip().split()[0].lower().rstrip(".,;:!?")
    return first.endswith("ing") and first != "being"


# Safety net: prompt-guideline leakage patterns.
_PROMPT_LEAK_PATTERNS: list[re.Pattern] = [
    re.compile(p, flags=re.IGNORECASE)
    for p in [
        r"^\s*unstable\s+(?:remaining\s+)?structure\s*[-—–:]\s*",
        r"^\s*piled\s+debris\s+or\s+rubble\s*[-—–:]\s*",
        r"^\s*trapped\s+(?:visible\s+)?person\s*[-—–:]\s*",
        r"^\s*exposed\s+(?:live\s+)?line\s*[-—–:]\s*",
        r"^\s*fire\s+or\s+smoke\s*[-—–:]\s*",
        r"^\s*water\s*[-—–:]\s*(?:pump|drain|divert)",
        r"\b(?:shore|brace|stabilize)\s*,\s*(?:shore|brace|stabilize)\s*,\s*(?:shore|brace|stabilize)\b",
        r"\b(?:clear|remove|extract)\s*,\s*(?:clear|remove|extract)\s*,\s*(?:clear|remove|extract)\b",
    ]
]


_SHOULD_RE = re.compile(r"\bshould\b", re.IGNORECASE)

_PAST_TENSE_CM_RE = re.compile(
    r"\b(began|organized|arrived|evacuated|started|conducted|rushed|"
    r"mobilized|cleared|worked|dispatched|surveyed|searched|boarded|"
    r"responded|coordinated|initiated|completed|managed|deployed|"
    r"secured|removed|assessed|established|cordoned)\b",
    re.IGNORECASE,
)


def is_should_form(s: str) -> bool:
    return bool(_SHOULD_RE.search(s)) if s else False


def is_past_tense_cm(s: str) -> bool:
    return bool(_PAST_TENSE_CM_RE.search(s)) if s else False


def is_prompt_leak(s: str) -> bool:
    if not s:
        return False
    for pat in _PROMPT_LEAK_PATTERNS:
        if pat.search(s):
            return True
    return False


def needs_repair(counter: str) -> str:
    if not counter or not counter.strip():
        return "empty"
    if len(counter.split()) < 6:
        return "too_short"
    if is_prompt_leak(counter):
        return "prompt_leak"
    if is_fragment(counter):
        return "fragment"
    if is_should_form(counter):
        return "should_form"
    if is_past_tense_cm(counter):
        return "past_tense"
    if is_desc_form(counter):
        return "desc_form"
    if is_simple_pres_narrative(counter):
        return "simple_pres_narr"
    return ""


# Severity rank for repair-vs-original arbitration.
# Lower number = worse. Used to keep the better of the two when both fail.
_REPAIR_SEVERITY = {
    "empty": 0,
    "too_short": 1,
    "fragment": 2,
    "prompt_leak": 3,
    "desc_form": 4,
    "simple_pres_narr": 5,
    "past_tense": 6,
    "should_form": 7,
    "": 99,  # OK / no failure
}


def _better_counter(orig: str, orig_reason: str,
                    repaired: str, repaired_reason: str) -> tuple[str, str]:
    """Pick whichever CM is less broken. Repair wins ties."""
    if not repaired:
        return orig, orig_reason
    if _REPAIR_SEVERITY.get(repaired_reason, -1) >= _REPAIR_SEVERITY.get(orig_reason, -1):
        return repaired, repaired_reason
    return orig, orig_reason


# ------------------------------- Parsing -------------------------------
_LABEL_RE = re.compile(
    r"^\s*(damage|countermeasure|cat|sev)\s*[:：]",
    re.IGNORECASE,
)


def parse_output(raw: str) -> dict:
    damage = ""
    counter = ""
    cat = "other_disaster"
    sev: int | None = None
    had_labels = False

    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    for ln in lines:
        m = re.match(
            r"^\s*(damage|countermeasure|cat|sev)\s*[:：]\s*(.+)$",
            ln, flags=re.IGNORECASE,
        )
        if not m:
            continue
        label = m.group(1).lower()
        value = m.group(2).strip().strip("*").strip()
        had_labels = True
        if label == "damage" and not damage:
            damage = _first_sentence(value)
        elif label == "countermeasure" and not counter:
            counter = _first_sentence(value)
        elif label == "cat":
            v = value.lower().split()[0].strip(".,:;")
            if v in VALID_CATS:
                cat = v
        elif label == "sev":
            mn = re.search(r"[0-2]", value)
            if mn:
                sev = int(mn.group(0))

    if had_labels and damage and not counter:
        body_lines = [ln for ln in lines if not _LABEL_RE.match(ln)]
        for ln in body_lines:
            candidate = _first_sentence(ln.strip().strip("*").strip())
            if candidate and candidate != damage and len(candidate.split()) >= 6:
                counter = candidate
                break

    if not had_labels:
        flat = _clean(raw)
        sents = re.split(r"(?<=[.!?])\s+", flat)
        if len(sents) >= 1:
            damage = _first_sentence(sents[0])
        if len(sents) >= 2:
            counter = _first_sentence(sents[1])

    damage = _finalize_sentence(damage)
    counter = _finalize_sentence(counter)
    if sev not in VALID_SEVS:
        sev = 1
    return {
        "damage": damage, "countermeasure": counter,
        "cat": cat, "sev": sev, "had_labels": had_labels,
    }


def parse_repair_counter(raw: str) -> str:
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in lines:
        m = re.match(r"^\s*countermeasure\s*[:：]\s*(.+)$", ln, flags=re.IGNORECASE)
        if m:
            return _first_sentence(m.group(1).strip().strip("*"))
    for ln in lines:
        if _LABEL_RE.match(ln):
            continue
        candidate = _first_sentence(ln.strip().strip("*"))
        if candidate and len(candidate.split()) >= 4:
            return candidate
    return _first_sentence(_clean(raw))


# ------------------------------- Confidence (Pi-identical) -------------------------------
def _mean_logprob(logprobs: list | None) -> float | None:
    if not logprobs:
        return None
    vals = [e["logprob"] for e in logprobs
            if isinstance(e, dict) and "logprob" in e]
    return sum(vals) / len(vals) if vals else None


def _min_logprob(logprobs: list | None) -> float | None:
    if not logprobs:
        return None
    vals = [e["logprob"] for e in logprobs
            if isinstance(e, dict) and "logprob" in e]
    return min(vals) if vals else None


def compute_confidence(logprobs: list | None, had_labels: bool,
                       semantic_penalty: int = 0) -> tuple[int, dict]:
    mean_lp = _mean_logprob(logprobs)
    min_lp  = _min_logprob(logprobs)

    if mean_lp is None:      base = 3
    elif mean_lp > -0.62:    base = 5
    elif mean_lp > -0.67:    base = 4
    elif mean_lp > -0.72:    base = 3
    elif mean_lp > -0.76:    base = 2
    else:                    base = 1

    conf = base
    if min_lp is not None and min_lp < -3.0:
        conf -= 1
    conf -= max(0, semantic_penalty)
    conf = max(1, min(5, conf))

    detail = {
        "mean_logprob":     round(mean_lp, 4) if mean_lp is not None else None,
        "min_logprob":      round(min_lp, 4)  if min_lp  is not None else None,
        "base":             base,
        "had_labels":       had_labels,
        "semantic_penalty": semantic_penalty,
    }
    return conf, detail


# ------------------------------- Pipeline -------------------------------
def run_one_image(img_path: Path, base_dim: int):
    """Returns (result_dict, err_str)."""
    dims = [base_dim] + RETRY_DIMS
    last_err = ""

    for i, dim in enumerate(dims):
        try:
            img_b64 = b64_image(img_path, max_dim=dim)
        except Exception as e:
            return None, f"image_encode_fail:{e}"

        def _do_pipeline():
            # Main pass: damage + countermeasure + cat + sev in one call.
            raw_main, ms1, lp1 = call_vllm(
                SYSTEM_PROMPT, USER_PROMPT, img_b64, VLM_MAX_TOKENS,
            )
            parsed = parse_output(raw_main)
            parsed["countermeasure"] = _finalize_sentence(
                strip_meta_preamble(parsed["countermeasure"])
            )

            # Conditional repair on the countermeasure only.
            reason = needs_repair(parsed["countermeasure"])
            raw_repair, ms2, lp2 = "", 0, None
            if reason:
                raw_repair, ms2, lp2 = call_vllm(
                    REPAIR_SYSTEM_PROMPT,
                    repair_user_prompt(parsed["damage"]),
                    img_b64, VLM_REPAIR_MAX_TOKENS,
                )
                new_c = _finalize_sentence(strip_meta_preamble(parse_repair_counter(raw_repair)))
                new_reason = needs_repair(new_c) if new_c else "empty"
                # Keep whichever CM is less broken; repair wins ties.
                chosen, _ = _better_counter(
                    parsed["countermeasure"], reason,
                    new_c, new_reason,
                )
                parsed["countermeasure"] = chosen

            combined_lp = (lp1 or []) + (lp2 or [])
            total_ms = int((ms1 or 0) + (ms2 or 0))

            return {
                "damage":         parsed["damage"],
                "countermeasure": parsed["countermeasure"],
                "cat":            parsed["cat"],
                "sev":            parsed["sev"],
                "had_labels":     parsed["had_labels"],
                "total_ms":       total_ms,
                "combined_lp":    combined_lp or None,
                "raw_main":       raw_main,
                "raw_repair":     raw_repair,
                "repair_reason":  reason or "",
            }

        try:
            result, energy_j, avg_mw = sample_power_during(_do_pipeline)
            result["energy_j"] = energy_j
            result["avg_mw"]   = avg_mw
            if i > 0:
                print(f"  [RETRY] recovered at dim={dim}")
            return result, ""
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "unknown"
            last_err = f"http_{code}"
            if code == 400 and i < len(dims) - 1:
                print(f"  [RETRY] 400 at dim={dim}, retry smaller")
                continue
            return None, last_err
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            last_err = f"timeout_dim_{dim}"
            if i < len(dims) - 1:
                print(f"  [RETRY] timeout at dim={dim}, retry smaller")
                continue
            return None, last_err
        except Exception as e:
            return None, str(e)

    return None, last_err or "unknown"


# ------------------------------- Main loop -------------------------------
def main(args):
    collected_path = Path(args.collected)
    out_path       = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    collected_dir  = collected_path.parent
    backup_existing_output(out_path)

    go_jetson_map: dict[str, dict] = {}
    if args.pi_oracle and Path(args.pi_oracle).exists():
        with open(args.pi_oracle, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                except Exception:
                    continue
                gj = obj.get("go_jetson")
                if gj and gj.get("status") == "ok":
                    go_jetson_map[obj["image_id"]] = gj
        print(f"[PI_ORACLE] loaded go_jetson for {len(go_jetson_map)} images")
    elif args.pi_oracle:
        print(f"[PI_ORACLE] WARNING: not found: {args.pi_oracle}")

    print(f"[CONFIG] collected={collected_path}")
    print(f"[CONFIG] out={out_path}")
    print(f"[CONFIG] vllm={VLLM_URL} model={VLLM_MODEL}")
    print(f"[CONFIG] temperature={VLM_TEMPERATURE} "
          f"main_max={VLM_MAX_TOKENS} repair_max={VLM_REPAIR_MAX_TOKENS}")
    print(f"[POWER]  sensor={POWER_NODE or 'not found'}")

    try:
        r = requests.get(VLLM_URL.replace("/chat/completions", "/models"), timeout=5)
        r.raise_for_status()
        models = [m.get("id") for m in r.json().get("data", [])]
        print(f"[CHECK] vLLM OK models={models}")
    except Exception as e:
        print(f"[CHECK] vLLM unreachable: {e}")
        return

    done = load_done_ids(out_path) if args.resume else set()
    if args.resume:
        print(f"[RESUME] skip existing ok records: {len(done)}")

    n_ok = n_skip = n_fail = n_repair = 0
    with collected_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "jetson_collected":
                continue

            image_id = str(rec.get("image_id", ""))
            run_id   = args.run_id or str(rec.get("run_id", DEFAULT_RUN_ID))

            if args.resume and image_id in done:
                n_skip += 1
                continue

            img_path = resolve_image(rec, collected_dir)
            if img_path is None:
                append_jsonl(out_path, {
                    "type": "jetson_oracle", "run_id": run_id,
                    "image_id": image_id, "ts": now_kst(),
                    "status": "fail", "error": "image_not_found",
                })
                n_fail += 1
                continue

            pi_a = rec.get("pi_a") or {}
            costs = pi_a.get("costs") or {}
            m5_total_ms       = int(costs.get("m5_total_ms", 0)       or 0)
            m5_pi_http_e2e_ms = int(costs.get("m5_pi_http_e2e_ms", 0) or 0)
            m5_pi_tx_bytes    = int(costs.get("m5_pi_tx_bytes", 0)    or 0)
            m5_pi_rx_bytes    = int(costs.get("m5_pi_rx_bytes", 0)    or 0)
            pi_total_ms       = int(costs.get("pi_total_ms", 0)       or 0)

            gj = go_jetson_map.get(image_id, {})
            pi_jet_http_e2e_ms = int(gj.get("pi_jet_http_e2e_ms", 0) or 0)
            pi_jet_tx_bytes    = int(gj.get("pi_jet_tx_bytes", 0)    or 0)
            pi_jet_rx_bytes    = int(gj.get("pi_jet_rx_bytes", 0)    or 0)

            seq = n_ok + n_fail + 1
            print(f"[{seq}] {image_id}")

            result, err = run_one_image(img_path, args.max_image_dim)

            if result is None:
                append_jsonl(out_path, {
                    "type": "jetson_oracle", "run_id": run_id,
                    "image_id": image_id, "ts": now_kst(),
                    "status": "fail", "error": f"vlm_fail:{err}",
                })
                n_fail += 1
                print(f"  [FAIL] {err}")
                if args.limit and (n_ok + n_fail) >= args.limit:
                    break
                continue

            if result["repair_reason"]:
                n_repair += 1

            damage  = result["damage"]
            counter = result["countermeasure"]
            text = (damage + " " + counter).strip() if damage and counter else (damage or counter)

            conf_final, conf_detail = compute_confidence(
                result["combined_lp"], result["had_labels"],
            )

            jet_ms = result["total_ms"]
            end_jet_lat = (
                m5_total_ms + m5_pi_http_e2e_ms + pi_total_ms
                + pi_jet_http_e2e_ms + jet_ms
            )
            end_jet_ovhd = (
                m5_pi_tx_bytes + m5_pi_rx_bytes
                + pi_jet_tx_bytes + pi_jet_rx_bytes
            )

            out_rec = {
                "type":        "jetson_oracle",
                "run_id":      run_id,
                "image_id":    image_id,
                "ts":          now_kst(),
                "status":      "ok",
                "saved_image": str(img_path),
                "end_jetson": {
                    "rep_output": "jetson",
                    "output": {
                        "text": text,
                        "cat":  result["cat"],
                        "sev":  result["sev"],
                        "conf": conf_final,
                    },
                    "conf_detail": conf_detail,
                    "conf_final":  conf_final,
                    "vlm_ms":      jet_ms,
                    "repair": {
                        "triggered": bool(result["repair_reason"]),
                        "reason":    result["repair_reason"],
                    },
                    "cost": {
                        "latency_ms": {
                            "total":              end_jet_lat,
                            "m5_total_ms":        m5_total_ms,
                            "m5_pi_http_e2e_ms":  m5_pi_http_e2e_ms,
                            "pi_total_ms":        pi_total_ms,
                            "pi_jet_http_e2e_ms": pi_jet_http_e2e_ms,
                            "jetson_total_ms":    jet_ms,
                        },
                        "signal_overhead_bytes": {
                            "total":           end_jet_ovhd,
                            "m5_pi_tx_bytes":  m5_pi_tx_bytes,
                            "m5_pi_rx_bytes":  m5_pi_rx_bytes,
                            "pi_jet_tx_bytes": pi_jet_tx_bytes,
                            "pi_jet_rx_bytes": pi_jet_rx_bytes,
                        },
                        "power_mw": result["avg_mw"],
                        "energy_j": result["energy_j"],
                    },
                },
                "_pi_b_status": rec.get("_pi_b_status", "unknown"),
            }
            if args.keep_raw:
                out_rec["raw_debug"] = {
                    "raw_main":   result["raw_main"],
                    "raw_repair": result["raw_repair"],
                }

            append_jsonl(out_path, out_rec)
            n_ok += 1

            mean_lp = conf_detail["mean_logprob"]
            min_lp  = conf_detail["min_logprob"]
            flag = f" REPAIR[{result['repair_reason']}]" if result["repair_reason"] else ""
            print(f"  cat={result['cat']} sev={result['sev']} conf={conf_final} "
                  f"mean_lp={mean_lp} min_lp={min_lp} vlm={jet_ms}ms "
                  f"energy={result['energy_j']}J{flag}")
            print(f"  END@Jetson latency={end_jet_lat}ms overhead={end_jet_ovhd}B")
            print(f"  text: {text}")

            if args.limit and (n_ok + n_fail) >= args.limit:
                break

    rate = (n_repair / max(n_ok, 1)) * 100.0
    print(f"\n[DONE] ok={n_ok} skip={n_skip} fail={n_fail} "
          f"repair_triggered={n_repair} ({rate:.1f}%) out={out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Jetson oracle builder v60 (single-pass)")
    p.add_argument("--collected",     default=DEFAULT_COLLECTED)
    p.add_argument("--out",           default=DEFAULT_OUT)
    p.add_argument("--run_id",        default=DEFAULT_RUN_ID)
    p.add_argument("--limit",         type=int, default=DEFAULT_LIMIT)
    p.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
    p.add_argument("--resume",        action="store_true", default=DEFAULT_RESUME)
    p.add_argument("--no-resume",     action="store_true")
    p.add_argument("--keep-raw",      action="store_true",
                   help="Include raw VLM outputs in each record for debugging.")
    p.add_argument("--pi-oracle", default=None,
                   help="pi_a_oracle.jsonl path; supplies go_jetson metrics.")
    a = p.parse_args()
    if a.no_resume:
        a.resume = False
    return a


if __name__ == "__main__":
    main(parse_args())
