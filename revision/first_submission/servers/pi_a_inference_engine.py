#!/usr/bin/env python3
"""
Pi_A Inference Pipeline v18 - Oracle Dataset Collection
=======================================================
Same as v17 but with a revised prompt:

- Role added: "You are a disaster response specialist."
- Situation/Action described by purpose, not by template:
    Situation: visible damage, blockage, or hazard
    Action: immediate field command to address that damage
- Removed <...> template placeholders
- Removed "never start with Rescue workers are" (handled by DESCRIPTIVE_ACTION_RE)
- Kept minimal negative constraints: no news summaries, no cause guessing,
  no organization names
- All heuristics (cat, sev, quality gate) identical to v17
"""

import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────── Config ───────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_PATH = str(_SCRIPT_DIR / "tier2_online_a_oracle/online_result_m5.jsonl")
OUT_PATH    = str(_SCRIPT_DIR / "tier2_online_a_oracle/pi_a_online_routing_v3.jsonl")

IMAGE_ROOT  = ""
JETSON_URL  = "http://192.168.0.92:8000/ingest_pi_a"
RUN_ID      = "pilot_300_run1"
LIMIT       = 0
RESUME      = True

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL       = os.getenv("MODEL", "ahmadwaqar/smolvlm2-2.2b-instruct")
VLM_TIMEOUT = int(os.getenv("TIMEOUT", "600"))
RELOAD_EVERY = 20
MAX_RETRIES = 2
KST = timezone(timedelta(hours=9))
JETSON_TIMEOUT = (5, 60)

VALID_CATS = {"landslide", "flood", "fire", "earthquake", "hurricane", "other_disaster"}

NUM_PREDICT = 80

# Keep the same temperature as the 5526-image collection and only add an
# optional seed for reproducibility. Do not set top_k/top_p here; leaving them
# to the backend defaults best matches the original v18 collection.
TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.2"))
SEED        = os.getenv("VLM_SEED", "42")

# ─────────────────────── Helpers ───────────────────────
def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="milliseconds")

def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

# ─────────────────────── Prompt ───────────────────────
PI_A_PROMPT = (
    "You are a disaster response specialist. Look at this disaster photograph.\n"
    "Write exactly two labeled lines:\n"
    "Situation: one sentence describing the visible damage, blockage, or hazard.\n"
    "Action: one immediate field command to address that damage.\n"
    "\n"
    "Each line is 12-18 words. Name specific structures, locations, or materials. "
    "Action starts with an imperative verb. "
    "No news summaries, no cause guessing, no organization names."
)

PI_A_RETRY_PROMPT_TEMPLATE = (
    "Same image. Write exactly two labeled lines (12-18 words each):\n"
    "Situation: visible damage, blockage, or hazard.\n"
    "Action: immediate field command starting with a verb.\n\n"
    "Previous output:\n{prev_raw}"
)


# ─────────────────────── Text parsing ───────────────────────
def strip_leading_boilerplate(text: str) -> str:
    text = normalize_ws(text)
    for _ in range(3):
        text = re.sub(
            r"^(?:"
            r"the image (?:shows|depicts|captures|provided shows)"
            r"|this image (?:shows|depicts)"
            r"|based on the image"
            r"|in the image"
            r"|in this image"
            r"|the (?:damage|situation|scene|disaster) visible in this image (?:includes?|is|shows?)"
            r"|the (?:damage|situation|scene|disaster) visible in this image"
            r")[,:]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
    return text


def _strip_trailing_boilerplate(text: str) -> str:
    return re.sub(
        r"\s*(?:This description is based on|This is based on|Note:|Please note)\b.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def extract_sentences(text: str, max_sentences: int = 1) -> str:
    text = strip_leading_boilerplate(text)
    text = _strip_trailing_boilerplate(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    kept = parts[:max_sentences] if parts else []
    if not kept:
        return ""
    res = " ".join(kept)
    if res and res[-1] not in ".!?":
        res += "."
    if res:
        res = res[0].upper() + res[1:]
    return res


def _strip_cause_clauses(text: str) -> str:
    """Strip 'due to/caused by ...' tails that are speculative cause-guesses."""
    text = re.sub(
        r",?\s*(?:due to|caused by|because of|as a result of|likely due to|possibly due to)\s+[^.]+[.]?",
        ".",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\.", ".", text)
    return text


def _strip_leading_label(text: str) -> str:
    """Remove any leftover '<Word>:' prefix that may have leaked through parsing."""
    if not text:
        return text
    for _ in range(2):
        new = re.sub(r"^\s*[A-Za-z][A-Za-z _-]{0,25}\s*:\s*", "", text)
        if new == text:
            break
        text = new
    if text:
        text = text[0].upper() + text[1:]
    return text


# Match "Situation: ...", "Action: ...", and common aliases / "Disaster:" etc.
_LABEL_RE = re.compile(
    r"^(?:\*{0,2})\s*"
    r"(?P<label>Situation|Disaster|Damage|Scene|Description|Observation|Action|cat|category|sev|severity|severe)"
    r"\s*(?:line)?\s*:\s*(?:\*{0,2})\s*(?P<rest>.*)$",
    re.IGNORECASE,
)


def parse_situation_action(raw_text: str) -> tuple[str, str, bool]:
    """Return (situation, action, had_both_labels). Robust against:
      - single-line output: "Situation: ... Action: ..."
      - bullet lists under a label
      - blank line right after label, content on next line
      - VLM using "Disaster:"/"Damage:"/"Scene:" instead of "Situation:"
      - extra unwanted lines (cat:, sev:, etc.) — ignored gracefully
    """
    # Force newline before any inline label so single-line output parses correctly.
    normalized = re.sub(
        r"(?<!^)\s*(?<![A-Za-z])"
        r"(Situation|Disaster|Damage|Scene|Description|Observation|Action|cat|category|sev|severity|severe)"
        r"\s*:",
        r"\n\1:",
        raw_text,
        flags=re.IGNORECASE,
    )

    sit_aliases = {"situation", "disaster", "damage", "scene", "description", "observation"}

    # Collect content per label across multiple lines (until next label).
    buckets = {"situation": [], "action": []}
    current = None
    found = {"situation": False, "action": False}

    for raw_line in normalized.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        m = _LABEL_RE.match(line)
        if m:
            lab = m.group("label").lower()
            if lab in sit_aliases:
                current = "situation"
            elif lab == "action":
                current = "action"
            else:
                current = None  # cat/sev/etc — ignore
            if current:
                found[current] = True
                rest = m.group("rest").strip()
                if rest:
                    buckets[current].append(rest)
            continue
        # Continuation line — append to current bucket. Strip bullet markers.
        if current in ("situation", "action"):
            cleaned = re.sub(r"^[\-\*•]\s*", "", line)
            buckets[current].append(cleaned)

    situation = " ".join(buckets["situation"]).strip()
    action = " ".join(buckets["action"]).strip()

    if situation:
        situation = _strip_trailing_boilerplate(situation)
        situation = _strip_cause_clauses(situation)
        situation = extract_sentences(situation, max_sentences=1)
    if action:
        action = _strip_trailing_boilerplate(action)
        action = extract_sentences(action, max_sentences=1)

    # Fallback: if either is missing, split raw text by sentences,
    # excluding lines that already start with a label.
    if not situation or not action:
        cleaned = _strip_trailing_boilerplate(raw_text)
        parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned.strip())
        parts = [p.strip() for p in parts if p.strip() and not _LABEL_RE.match(p.strip())]
        if not situation and len(parts) >= 1:
            situation = extract_sentences(_strip_cause_clauses(parts[0]), max_sentences=1)
        if not action and len(parts) >= 2:
            action = extract_sentences(parts[1], max_sentences=1)

    # Final defensive strip of any leaked "<Word>:" prefix.
    situation = _strip_leading_label(situation)
    action = _strip_leading_label(action)

    return situation, action, (found["situation"] and found["action"])


# ─────────────────────── Cat classification (heuristic) ───────────────────────
def classify_cat(text: str) -> str:
    """Score-based fallback. Avoids other_disaster unless no signal."""
    t = text.lower()
    scores: dict[str, float] = defaultdict(float)

    # fire
    if re.search(r"\b(?:fire|wildfire|blaze)\b", t):
        scores["fire"] += 10
    if re.search(r"\b(?:flames?|burning|burned|burnt|charr\w*|soot|scorch\w*|ablaze|inferno|smoldering)\b", t):
        scores["fire"] += 6
    if re.search(r"\bsmoke\b", t) and not re.search(r"\b(?:flood|water|wave)\b", t):
        scores["fire"] += 3
    if re.search(r"\bengulfed\b", t) and not re.search(r"\b(?:flood|water|wave)\b", t):
        scores["fire"] += 4

    # flood
    if re.search(r"\b(?:flood|flooded|flooding|inundat\w*)\b", t):
        scores["flood"] += 10
    if re.search(r"\b(?:floodwaters?|submerged|waterlogged|waterline)\b", t):
        scores["flood"] += 6
    if re.search(r"\bwater\b", t) and re.search(
        r"\b(?:rising|level|deep|knee|waist|washed|standing|covering|filled)\b", t
    ):
        scores["flood"] += 4
    if re.search(r"\b(?:river|stream)\s+(?:overflow\w*|burst|breached)\b", t):
        scores["flood"] += 6

    # hurricane / storm-wind
    if re.search(r"\bhurricane\b", t):
        scores["hurricane"] += 15
    if re.search(r"\b(?:cyclone|typhoon|tornado)\b", t):
        scores["hurricane"] += 12
    if re.search(r"\bstorm\b", t) and re.search(
        r"\b(?:roof|tree|boat|pole|wind|torn|uprooted|ripped|surge)\b", t
    ):
        scores["hurricane"] += 6
    if re.search(r"\b(?:uprooted|torn\s+(?:off|apart|away)|ripped\s+(?:off|apart|away))\b", t):
        scores["hurricane"] += 4
    # Broader downed tree/pole/wire pattern (any location)
    if re.search(
        r"\b(?:snapped|fallen|toppled|downed|knocked\s+down)\s+"
        r"(?:trees?|branches|poles?|power\s*lines?|wires?|fences?)\b",
        t,
    ):
        scores["hurricane"] += 5
    if re.search(r"\b(?:wind|storm)\s+(?:damage|debris)\b", t):
        scores["hurricane"] += 5
    if re.search(r"\b(?:roof|shingles?|siding)\b", t) and re.search(
        r"\b(?:torn|ripped|blown|peeled|twisted)\b", t
    ):
        scores["hurricane"] += 4
    if re.search(r"\b(?:storm\s*surge|coastal\s+wave|tidal\s+wave)\b", t):
        scores["hurricane"] += 6
    # Coastal/shoreline + damage — strong hurricane indicator
    if re.search(r"\b(?:coast|coastal|shore|shoreline|beach|seafront|waterfront)\b", t) and re.search(
        r"\b(?:damaged|destroyed|wreck|debris|scattered|battered|ravaged|"
        r"flooded|submerged|broken|smashed|wreckage)\b",
        t,
    ):
        scores["hurricane"] += 6
    # Twisted / blown structure
    if re.search(r"\b(?:twisted|blown\s+(?:off|away|down))\b", t) and re.search(
        r"\b(?:roof|wall|tree|pole|fence|sign|structure)\b", t
    ):
        scores["hurricane"] += 3

    # landslide
    if re.search(r"\b(?:landslide|mudslide|rockslide)\b", t):
        scores["landslide"] += 10
    if re.search(r"\b(?:rockfall|debris\s*slide|slope\s*failure|slope\s*collapse)\b", t):
        scores["landslide"] += 8
    if re.search(r"\b(?:boulder|hillside|embankment|slope)\b", t) and re.search(
        r"\b(?:block|road|path|collaps|down|onto)\b", t
    ):
        scores["landslide"] += 5
    if (
        re.search(r"\b(?:road|highway|path|route|railway|tracks?)\b", t)
        and re.search(r"\b(?:block\w*|obstruct\w*|cover\w*)\b", t)
        and re.search(r"\b(?:debris|rocks?|mud|soil|earth|dirt|boulder)\b", t)
    ):
        scores["landslide"] += 4

    # earthquake
    if re.search(r"\b(?:earthquake|seismic|quake|tremor|aftershock)\b", t):
        scores["earthquake"] += 10
    if re.search(r"\bcollaps\w*\b", t) and re.search(
        r"\b(?:building|structure|wall|floor|story|storey|house|apartment|concrete|tower)\b", t
    ):
        scores["earthquake"] += 5
    if re.search(r"\brubble\b", t) and re.search(r"\b(?:building|structure|concrete|bricks?)\b", t):
        scores["earthquake"] += 4
    elif re.search(r"\brubble\b", t):
        scores["earthquake"] += 2
    if re.search(r"\b(?:cracked|tilted|leaning)\b", t) and re.search(
        r"\b(?:building|wall|tower|structure|column)\b", t
    ):
        scores["earthquake"] += 3
    if re.search(r"\bground\s+(?:cracks?|fissures?)\b", t):
        scores["earthquake"] += 3

    # secondary clues
    if re.search(r"\b(?:fallen|toppled|uprooted)\s+tree\b", t) and re.search(
        r"\b(?:road|street|path)\b", t
    ):
        scores["hurricane"] += 4
    if re.search(r"\b(?:boat|ship|vessel|yacht)\b", t) and re.search(
        r"\b(?:beach|shore|coast|damaged|destroyed|wreck)\b", t
    ):
        scores["hurricane"] += 3

    if scores and max(scores.values()) > 0:
        return max(scores, key=scores.get)

    # Soft last-resort: avoid premature "other_disaster"
    if re.search(
        r"\b(?:coast|coastal|shore|shoreline|beach|seafront|waterfront)\b", t
    ) and re.search(r"\b(?:damaged|destroyed|broken|wrecked|debris|wreckage)\b", t):
        return "hurricane"
    if re.search(r"\b(?:damaged|destroyed|broken|wrecked)\b", t) and re.search(
        r"\b(?:building|house|wall|structure|apartment)\b", t
    ):
        return "earthquake"
    if re.search(r"\b(?:debris|rocks?|mud)\b", t) and re.search(
        r"\b(?:road|street|path|highway|tracks?)\b", t
    ):
        return "landslide"
    return "other_disaster"


# ─────────────────────── Sev classification (heuristic) ───────────────────────
def classify_sev(text: str) -> int:
    """sev fallback using new rubric:
      0 = little_or_none (minor / superficial / no major damage)
      1 = mild (one section seriously damaged, localized hazard)
      2 = severe (widespread or total destruction, multi-structure, total loss)
    """
    t = text.lower()

    # 2 = severe: widespread / total destruction signals
    severe_widespread = re.search(
        r"\b(?:widespread|extensive|massive|total\s+destruction|total\s+collapse|"
        r"devastat\w*|engulf\w*|demolished|ravaged|reduced\s+to\s+rubble|"
        r"levelled|leveled|flattened|catastrophic|inferno|ablaze|deep\s+water|"
        r"submerged|washed\s+out|blocked|blocking|cut\s+off|impassable|"
        r"completely\s+(?:destroyed|collapsed|demolished|buried|covered|submerged)|"
        r"entirely\s+(?:destroyed|collapsed|demolished|buried)|"
        r"totally\s+(?:destroyed|collapsed|demolished))\b",
        t,
    )
    # Multiple structures damaged (plural)
    multi_structure = re.search(
        r"\b(?:buildings|structures|houses|homes)\b", t
    ) and re.search(r"\b(?:collaps\w*|destroyed|damaged|in\s+ruins)\b", t)
    # Single structure with clear total-loss verb — VLM often describes one building
    single_collapse = re.search(
        r"\b(?:building|structure|house|home|apartment|bridge|tower)\b", t
    ) and re.search(
        r"\b(?:collaps\w*|in\s+ruins|crumbled|caved\s+in|"
        r"completely\s+destroyed|entirely\s+destroyed|reduced\s+to\s+rubble)\b",
        t,
    )

    # 0 = little_or_none: minor / superficial markers
    little = re.search(
        r"\b(?:minor|small|slight|superficial|limited|little|no\s+visible|no\s+major|"
        r"intact|undamaged|mostly\s+intact|barely|minimal|negligible)\b",
        t,
    )

    if severe_widespread or multi_structure or single_collapse:
        return 2
    if little:
        return 0
    return 1  # mild — one section seriously damaged, localized hazard


# ─────────────────────── Quality gate ───────────────────────
# Descriptive action: present-continuous or "must/swiftly/arrive" patterns
# that mean someone is describing responders rather than issuing a command.
DESCRIPTIVE_ACTION_RE = re.compile(
    r"\b(?:rescue\s+workers?|emergency\s+responders?|emergency\s+workers?|"
    r"firefighters?|responders?|relief\s+workers?|response\s+teams?)\b"
    r".{0,80}\b(?:must|need|needs|needed|arrive|arrived|mobilize|mobilizes|"
    r"swiftly|quickly|urgently|are|is|rush|rushes|working|seen|trying|"
    r"begin|begins|started|clearing|searching|responding|inspecting|"
    r"conducting|assessing|coordinating)\b",
    re.IGNORECASE,
)

# News-style Situation opener: "X has struck / leaving destruction in its wake"
# penalty +1 for scoring; triggers retry when combined with no visual content.
NEWS_STYLE_RE = re.compile(
    r"\b(?:has\s+struck|has\s+left|has\s+devastated|has\s+caused|has\s+hit|"
    r"in\s+its\s+wake|its\s+wake|leaving\s+destruction|leaving\s+behind|"
    r"leaving\s+\w+\s+in\s+ruins|swept\s+through|ravaged\s+the)\b",
    re.IGNORECASE,
)

LOCATION_TERMS = [
    "road", "highway", "railway", "tracks", "beach", "shore", "coast",
    "street", "bridge", "building", "house", "forest", "slope", "cliff",
    "hillside", "canyon", "route", "river", "village", "town", "city", "area",
]

VISUAL_TERMS = [
    "rocks", "boulders", "debris", "slope", "cliff", "beach", "road", "water",
    "smoke", "flames", "trees", "mud", "soil", "building", "bridge", "tracks",
    "highway", "hillside", "route", "shore", "roof", "wall", "car", "boat",
    "rubble", "structure", "house",
]


def analyze_quality(situation: str, action: str) -> dict:
    reasons = []
    penalty = 0
    sit_l = situation.lower()
    act_l = action.lower()
    text_all = f"{situation} {action}".lower()

    if not situation or len(situation.split()) < 5:
        reasons.append("weak_situation")
        penalty += 3
    if not action or len(action.split()) < 3:
        reasons.append("weak_action")
        penalty += 3

    if len(situation.split()) > 25:
        reasons.append("long_situation")
        penalty += 1
    if len(action.split()) > 25:
        reasons.append("long_action")
        penalty += 1

    # Cause-guessing
    if re.search(r"\bdue to\b|\bcaused by\b|\bbecause of\b", text_all):
        reasons.append("cause_guess")
        penalty += 1

    # Descriptive action ("Rescue workers must quickly...", "responders arrive...")
    if action and DESCRIPTIVE_ACTION_RE.search(action):
        reasons.append("descriptive_action")
        penalty += 2

    # News-style Situation opener ("has struck", "in its wake", etc.)
    if situation and NEWS_STYLE_RE.search(situation):
        reasons.append("news_style_situation")
        penalty += 1

    # Location anchor
    if not any(t in sit_l for t in LOCATION_TERMS):
        reasons.append("missing_location_anchor")
        penalty += 1

    # Visual specificity
    visual_hits = sum(1 for t in VISUAL_TERMS if t in sit_l)
    if visual_hits < 1:
        reasons.append("low_visual_specificity")
        penalty += 1

    passed = penalty <= 5
    return {
        "passed": passed,
        "penalty": penalty,
        "reasons": reasons,
        "visual_hits": visual_hits,
    }


# ─────────────────────── Confidence (logprob-based) ───────────────────────
def get_mean_logprob(logprobs: list | None) -> float | None:
    if not logprobs:
        return None
    lp_vals = [entry["logprob"] for entry in logprobs if isinstance(entry, dict) and "logprob" in entry]
    return sum(lp_vals) / len(lp_vals) if lp_vals else None


def get_min_logprob(logprobs: list | None) -> float | None:
    if not logprobs:
        return None
    lp_vals = [entry["logprob"] for entry in logprobs if isinstance(entry, dict) and "logprob" in entry]
    return min(lp_vals) if lp_vals else None


def compute_confidence(
    logprobs: list | None,
    had_labels: bool,
    semantic_penalty: int = 0,
) -> tuple[int, dict]:
    mean_lp = get_mean_logprob(logprobs)
    min_lp = get_min_logprob(logprobs)

    if mean_lp is None:
        base = 3
    elif mean_lp > -0.62:
        base = 5
    elif mean_lp > -0.67:
        base = 4
    elif mean_lp > -0.72:
        base = 3
    elif mean_lp > -0.76:
        base = 2
    else:
        base = 1

    conf = base
    if min_lp is not None and min_lp < -3.0:
        conf -= 1
    conf -= max(0, semantic_penalty)
    conf = max(1, min(5, conf))

    return conf, {
        "mean_logprob": round(mean_lp, 4) if mean_lp is not None else None,
        "min_logprob": round(min_lp, 4) if min_lp is not None else None,
        "base": base,
        "had_labels": had_labels,
        "semantic_penalty": semantic_penalty,
    }


# ─────────────────────── VLM ───────────────────────
def reload_model():
    base = OLLAMA_URL.rsplit("/", 1)[0]
    try:
        requests.post(base + "/generate", json={"model": MODEL, "keep_alive": "0s"}, timeout=10)
        time.sleep(3)
        requests.post(
            base + "/generate",
            json={
                "model": MODEL,
                "prompt": "hi",
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": 1,
                    **({"seed": int(SEED)} if SEED else {}),
                },
            },
            timeout=120,
        )
        print("  [RELOAD] model reloaded")
    except Exception as e:
        print(f"  [RELOAD] warning: {e}")


def call_ollama(prompt: str, image_b64: str, num_predict: int) -> tuple[str, int, list | None]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "logprobs": True,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": num_predict,
            **({"seed": int(SEED)} if SEED else {}),
        },
    }
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            r = requests.post(OLLAMA_URL, json=payload, timeout=VLM_TIMEOUT)
            dt_ms = int((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            data = r.json()
            return data.get("response", ""), dt_ms, data.get("logprobs")
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                print(f"  [VLM_RETRY] attempt {attempt + 1}/{MAX_RETRIES}: {type(e).__name__}")
                time.sleep(5)
                reload_model()
            else:
                raise last_err


# ─────────────────────── Pi inference ───────────────────────
def run_pi_inference(img_b64: str) -> tuple[dict, list | None, int, dict]:
    """One VLM call by default. Retry only when both lines are missing or too short."""
    total_ms = 0
    debug: dict = {}

    raw1, ms1, lp1 = call_ollama(PI_A_PROMPT, img_b64, NUM_PREDICT)
    total_ms += ms1
    sit1, act1, had_labels1 = parse_situation_action(raw1)
    ana1 = analyze_quality(sit1, act1)

    debug["primary_raw"] = raw1
    debug["primary_reasons"] = ana1["reasons"]
    debug["primary_penalty"] = ana1["penalty"]
    debug["primary_had_labels"] = had_labels1

    # Descriptive action always forces retry.
    # News-style Situation forces retry only when visual content is absent
    # (visual_hits < 1) — avoids unnecessary retries for partially OK outputs.
    primary_ok = (
        bool(sit1) and bool(act1) and ana1["passed"]
        and not DESCRIPTIVE_ACTION_RE.search(act1)
        and not (NEWS_STYLE_RE.search(sit1) and ana1["visual_hits"] < 1)
    )
    if primary_ok:
        text = f"{sit1} {act1}".strip()
        cat = classify_cat(text)
        sev = classify_sev(text)
        debug["chosen"] = "primary"
        return {
            "situation": sit1,
            "action": act1,
            "text": text,
            "cat": cat,
            "sev": sev,
            "had_labels": had_labels1,
        }, lp1, total_ms, debug

    # Retry
    retry_prompt = PI_A_RETRY_PROMPT_TEMPLATE.format(prev_raw=(raw1 or "").strip())
    raw2, ms2, lp2 = call_ollama(retry_prompt, img_b64, NUM_PREDICT)
    total_ms += ms2
    sit2, act2, had_labels2 = parse_situation_action(raw2)
    ana2 = analyze_quality(sit2, act2)

    debug["backup_raw"] = raw2
    debug["backup_reasons"] = ana2["reasons"]
    debug["backup_penalty"] = ana2["penalty"]
    debug["backup_had_labels"] = had_labels2

    def _score(sit, act, ana, had):
        s = 0
        if had: s += 5
        if sit and len(sit.split()) >= 5: s += 3
        if act and len(act.split()) >= 3: s += 3
        s -= ana["penalty"]
        s += ana["visual_hits"]
        return s

    s1 = _score(sit1, act1, ana1, had_labels1)
    s2 = _score(sit2, act2, ana2, had_labels2)

    if s2 > s1:
        sit, act, had_labels, logprobs = sit2, act2, had_labels2, lp2
        debug["chosen"] = "backup"
    else:
        sit, act, had_labels, logprobs = sit1, act1, had_labels1, lp1
        debug["chosen"] = "primary"

    text = f"{sit} {act}".strip()
    cat = classify_cat(text)
    sev = classify_sev(text)
    return {
        "situation": sit,
        "action": act,
        "text": text,
        "cat": cat,
        "sev": sev,
        "had_labels": had_labels,
    }, logprobs, total_ms, debug


# ─────────────────────── M5 cost extraction ───────────────────────
def extract_m5_costs(rec: dict, patches: dict, image_id: str) -> dict:
    end_m5_cand = rec.get("end_m5") or rec.get("candidates", {}).get("end_m5", {})
    end_m5_lat = end_m5_cand.get("cost", {}).get("latency_ms", {})
    m5_pi_comm = rec.get("go_pi", rec.get("m5_pi_comm", {}))

    m5_total_ms = end_m5_lat.get("m5_total_ms", 0)
    if not m5_total_ms:
        m5_lat = rec.get("latency_ms", rec.get("latency", {}))
        m5_total_ms = m5_lat.get("m5_total_ms", 0)

    m5_pi_http_e2e_ms = m5_pi_comm.get("m5_pi_http_e2e_ms", 0)
    if not m5_pi_http_e2e_ms and image_id in patches:
        m5_pi_http_e2e_ms = patches[image_id].get("m5_pi_http_e2e_ms", 0)

    m5_pi_tx = m5_pi_comm.get("m5_pi_tx_bytes", 0)
    if not m5_pi_tx:
        m5_sig = rec.get("signal_overhead_bytes", {})
        m5_pi_tx = m5_sig.get("m5_pi_tx_bytes", 0)

    m5_pi_rx = m5_pi_comm.get("m5_pi_rx_bytes", 0)
    if not m5_pi_rx and image_id in patches:
        m5_pi_rx = patches[image_id].get("m5_pi_rx_bytes", 0)

    return {
        "m5_total_ms":       m5_total_ms,
        "m5_pi_http_e2e_ms": m5_pi_http_e2e_ms,
        "m5_pi_tx_bytes":    m5_pi_tx,
        "m5_pi_rx_bytes":    m5_pi_rx,
    }


def load_patches(merged_dir: Path, run_id: str) -> dict[str, dict]:
    patch_file = merged_dir / "oracle_patch__pilot_300_run1.jsonl"
    patches: dict[str, dict] = {}
    if not patch_file.exists():
        return patches
    with patch_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("role", obj.get("device", "").lower().replace("m5_", "")) != "a":
                continue
            image_id = obj.get("image_id", "")
            if image_id:
                patches[image_id] = {
                    "m5_pi_http_e2e_ms": obj.get("m5_pi_http_e2e_ms", 0),
                    "m5_pi_tx_bytes":    obj.get("m5_pi_tx_bytes", 0),
                    "m5_pi_rx_bytes":    obj.get("m5_pi_rx_bytes", 0),
                }
    return patches


# ─────────────────────── Jetson forward (multipart with image) ───────────────────────
def send_to_jetson(jetson_url: str, meta: dict, image_path: Path) -> dict:
    meta_json = json.dumps(meta, ensure_ascii=False)
    session = requests.Session()
    with open(image_path, "rb") as img_f:
        req = requests.Request(
            "POST", jetson_url,
            data={"meta": meta_json},
            files={"image": (image_path.name, img_f, "image/jpeg")},
        )
        prepared = session.prepare_request(req)
    tx_bytes = int(prepared.headers.get("Content-Length", 0))
    t0 = time.perf_counter()
    resp = session.send(prepared, timeout=JETSON_TIMEOUT)
    http_e2e_ms = int((time.perf_counter() - t0) * 1000)
    rx_bytes = len(resp.content)
    resp.raise_for_status()
    try:
        jetson_total_ms = resp.json().get("jetson_total_ms", 0)
    except Exception:
        jetson_total_ms = 0
    return {
        "pi_jet_http_e2e_ms": http_e2e_ms,
        "pi_jet_tx_bytes": tx_bytes,
        "pi_jet_rx_bytes": rx_bytes,
        "jetson_total_ms": jetson_total_ms,
        "status": "ok",
    }


# ─────────────────────── Image & resume handling ───────────────────────
def resolve_image(rec: dict, merged_dir: Path, image_root: str) -> Path | None:
    image_id = rec.get("image_id", "")
    saved = rec.get("saved_image", "")
    candidates = []
    if image_root:
        candidates.extend([Path(image_root) / saved, Path(image_root) / image_id])
    candidates.extend([Path(saved), merged_dir / saved, merged_dir / "images" / image_id])
    for c in candidates:
        if c.exists():
            return c
    return None


def load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("status") == "ok":
                    done.add(obj["image_id"])
            except Exception:
                pass
    return done


# ─────────────────────── Main ───────────────────────
def main():
    merged_p = Path(MERGED_PATH)
    out_p = Path(OUT_PATH)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    merged_dir = merged_p.parent

    print(f"[CONFIG] merged : {merged_p}")
    print(f"[CONFIG] out    : {out_p}")
    print(f"[CONFIG] jetson : {JETSON_URL or '(skipped)'}")
    print(f"[CONFIG] model  : {MODEL}")
    print(f"[CONFIG] decode : temperature={TEMPERATURE} seed={SEED or '(backend default)'} (no top_k/top_p)")
    print("[CONFIG] prompt : role=disaster specialist, Situation+Action (v18)")
    print("[CONFIG] cat/sev: heuristic from text (sev: 0=little_or_none, 1=mild, 2=severe)")

    patches = load_patches(merged_dir, RUN_ID)
    done_ids = load_done_ids(out_p) if RESUME else set()
    if done_ids:
        print(f"[RESUME] found {len(done_ids)} already-done records → skipping them")
    elif RESUME:
        print(f"[RESUME] no previous output found at {out_p} → starting from scratch")

    jetson_ok = False
    if JETSON_URL:
        try:
            requests.get(JETSON_URL.rsplit("/", 1)[0] + "/health", timeout=3)
            jetson_ok = True
            print("[CHECK] Jetson reachable")
        except Exception:
            print("[CHECK] Jetson NOT reachable - will skip Jetson forward")
    print()

    n_ok, n_skip, n_fail = 0, 0, 0

    with merged_p.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "complete":
                continue

            image_id = rec["image_id"]
            run_id = RUN_ID or rec.get("run_id", "unknown")

            if RESUME and image_id in done_ids:
                n_skip += 1
                continue

            img_path = resolve_image(rec, merged_dir, IMAGE_ROOT)
            if img_path is None:
                append_jsonl(str(out_p),
                    {"type": "pi_a_oracle", "image_id": image_id, "status": "fail", "error": "image_not_found"})
                n_fail += 1
                continue

            seq = n_ok + n_fail + 1
            print(f"[{seq}] {image_id}")

            costs = extract_m5_costs(rec, patches, image_id)
            t_pi_start = time.perf_counter()

            try:
                img_b64 = b64_image(img_path)
                result, logprobs, vlm_ms, debug_info = run_pi_inference(img_b64)
            except Exception as e:
                print(f"  [FAIL] vlm error: {e}")
                append_jsonl(str(out_p),
                    {"type": "pi_a_oracle", "image_id": image_id, "ts": now_kst(),
                     "status": "fail", "error": f"vlm_fail: {e}"})
                n_fail += 1
                if LIMIT and (n_ok + n_fail) >= LIMIT:
                    break
                continue

            situation = result["situation"]
            action = result["action"]
            text = result["text"]
            cat = result["cat"]
            sev = result["sev"]
            had_labels = result["had_labels"]

            if not text or len(text.split()) < 4:
                append_jsonl(str(out_p),
                    {"type": "pi_a_oracle", "image_id": image_id, "ts": now_kst(),
                     "status": "fail", "error": f"no_vlm_output (text='{text}')"})
                n_fail += 1
                if LIMIT and (n_ok + n_fail) >= LIMIT:
                    break
                continue

            final_quality = analyze_quality(situation, action)
            _quality_issues = int(not final_quality["passed"])
            _retry_used = int(debug_info.get("chosen") == "backup")
            semantic_penalty = min(2, _quality_issues + _retry_used)
            conf_final, conf_detail = compute_confidence(logprobs, had_labels, semantic_penalty=semantic_penalty)
            pi_total_ms = int((time.perf_counter() - t_pi_start) * 1000)

            jetson_meas = {
                "pi_jet_http_e2e_ms": 0,
                "pi_jet_tx_bytes": 0,
                "pi_jet_rx_bytes": 0,
                "jetson_total_ms": 0,
                "status": "skipped",
            }
            if jetson_ok:
                jetson_meta = {
                    "type": "pi_a_result",
                    "run_id": run_id,
                    "image_id": image_id,
                    "pi_a_text": text,
                    "pi_a_cat": cat,
                    "pi_a_sev": sev,
                    "pi_a_conf": conf_final,
                    "ts": now_kst(),
                    "m5_total_ms": costs["m5_total_ms"],
                    "m5_pi_http_e2e_ms": costs["m5_pi_http_e2e_ms"],
                    "m5_pi_tx_bytes": costs["m5_pi_tx_bytes"],
                    "m5_pi_rx_bytes": costs["m5_pi_rx_bytes"],
                    "pi_total_ms": pi_total_ms,
                }
                try:
                    jetson_meas = send_to_jetson(JETSON_URL, jetson_meta, img_path)
                except Exception as e:
                    jetson_meas["status"] = f"fail: {e}"
                    jetson_ok = False
                    print(f"  [JETSON] failed, disabling: {e}")

            end_pi_latency = costs["m5_total_ms"] + costs["m5_pi_http_e2e_ms"] + pi_total_ms
            end_pi_overhead = costs["m5_pi_tx_bytes"] + costs["m5_pi_rx_bytes"]

            oracle_rec = {
                "type": "pi_a_oracle",
                "run_id": run_id,
                "image_id": image_id,
                "ts": now_kst(),
                "status": "ok",
                "end_pi": {
                    "rep_output": "Pi_A",
                    "output": {"text": text, "cat": cat, "sev": sev, "conf": conf_final},
                    "conf_detail": conf_detail,
                    "conf_final": conf_final,
                    "quality_gate": {
                        "passed": final_quality["passed"],
                        "penalty": final_quality["penalty"],
                        "reasons": final_quality["reasons"],
                        "visual_hits": final_quality["visual_hits"],
                    },
                    "cost": {
                        "latency_ms": {
                            "total": end_pi_latency,
                            "m5_total_ms": costs["m5_total_ms"],
                            "m5_pi_http_e2e_ms": costs["m5_pi_http_e2e_ms"],
                            "pi_total_ms": pi_total_ms,
                            "pi_vlm_ms": vlm_ms,
                        },
                        "signal_overhead_bytes": {
                            "total": end_pi_overhead,
                            "m5_pi_tx_bytes": costs["m5_pi_tx_bytes"],
                            "m5_pi_rx_bytes": costs["m5_pi_rx_bytes"],
                        },
                    },
                },
                "go_jetson": jetson_meas,
            }

            append_jsonl(str(out_p), oracle_rec)
            n_ok += 1

            if RELOAD_EVERY and n_ok % RELOAD_EVERY == 0:
                print(f"\n[MAINTENANCE] {n_ok} records done, reloading model...")
                reload_model()

            lp_str = f"{conf_detail['mean_logprob']:.4f}" if conf_detail["mean_logprob"] is not None else "None"
            min_lp_str = f"{conf_detail['min_logprob']:.4f}" if conf_detail["min_logprob"] is not None else "None"
            print(f"  sit: {situation}")
            print(f"  act: {action}")
            print(f"  cat={cat} sev={sev} conf={conf_final} mean_lp={lp_str} min_lp={min_lp_str}")
            print(f"  vlm={vlm_ms}ms pi={pi_total_ms}ms chosen={debug_info.get('chosen')}")
            print(f"  quality: passed={final_quality['passed']} penalty={final_quality['penalty']} "
                  f"reasons={final_quality['reasons']}")
            print(f"  END@Pi: L={end_pi_latency}ms O={end_pi_overhead}B")

            if LIMIT and (n_ok + n_fail) >= LIMIT:
                break

    print(f"\n[DONE] ok={n_ok}  skip={n_skip}  fail={n_fail}  -> {out_p}")


if __name__ == "__main__":
    main()
