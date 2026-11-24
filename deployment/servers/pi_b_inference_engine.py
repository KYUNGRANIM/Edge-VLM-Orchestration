#!/usr/bin/env python3
"""
Pi_B Inference Pipeline v16 - Oracle Dataset Collection
=======================================================
Based on pi_b_pipeline_v15 with three targeted fixes:

(A) Non-ASCII detection (Arabic/foreign script from image signs):
    * parse_description returns empty if non-ASCII chars detected → forces retry
    * analyze_quality adds non_ascii penalty +4 as safety net

(B) Cause-guessing strip extended:
    * _strip_action_clauses now removes ", indicating that/suggesting X" trailing clauses

(C) Generic opener penalty:
    * analyze_quality adds generic_opener +1 for "A scene of destruction/devastation"
"""

import base64
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ─────────────────────── Config ───────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
MERGED_PATH = str(_SCRIPT_DIR / "tier2_online_a_oracle/online_result_m5.jsonl")
OUT_PATH    = str(_SCRIPT_DIR / "tier2_online_a_oracle/pi_b_online_routing_v3.jsonl")
IMAGE_ROOT  = ""
JETSON_URL  = "http://192.168.0.92:8000/ingest_pi_b"
RUN_ID      = "medic_rerun600"
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

NUM_PREDICT = 40

# Keep the same temperature as the 5526-image collection and only add an
# optional seed for reproducibility. Do not set top_k/top_p here; leaving them
# to the backend defaults best matches the original v16 collection.
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
PI_B_PROMPT = (
    "Disaster photo. Output exactly one line:\n"
    "Description: <one sentence describing what is damaged, blocked, or collapsed>\n"
    "\n"
    "12-18 words. Name specific structures, materials, and location. "
    "Mention what is blocked, fallen, submerged, or at risk. "
    "No action plan, no quotes, no bullets, no news headlines."
)

PI_B_RETRY_PROMPT_TEMPLATE = (
    "Rewrite for the same image as one line (12-18 words):\n"
    "Description: <what is damaged, blocked, or collapsed — name structures and location>\n\n"
    "Previous output:\n{prev_raw}"
)

PI_B_SHORT_OUTPUT_RETRY_PROMPT_TEMPLATE = (
    "The previous answer was too short: {prev_raw}\n"
    "Ignore landmark names, country names, and organization names. "
    "Describe only the visible physical hazard in the image.\n"
    "If water is visible, mention standing water or flooding. "
    "Return exactly one line:\n"
    "Description: <8-18 words about what is flooded, blocked, damaged, or at risk>"
)

PI_B_WATER_LANDMARK_FALLBACK = (
    "Standing water covers the paved area near vehicles and buildings, creating a possible flooding hazard."
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
            r"|the (?:damage|situation|scene) visible in this image (?:includes?|is|shows?)"
            r"|the (?:damage|situation|scene) visible in this image"
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


def _strip_action_clauses(text: str) -> str:
    # Trailing cause-guessing clauses
    text = re.sub(
        r",?\s*(?:due to|caused by|because of|as a result of|likely due to|possibly due to)\s+[^.]+[.]?",
        ".",
        text,
        flags=re.IGNORECASE,
    )
    # "indicating/suggesting that X" trailing clauses
    text = re.sub(
        r",?\s*(?:indicating|suggesting)\s+(?:that\s+)?[^.]+[.]?",
        ".",
        text,
        flags=re.IGNORECASE,
    )
    # Action-recommendation sub-clauses
    text = re.sub(
        r",?\s*(?:will need to|needs to|should|must|recommend(?:ed)?|advise[d]?)\s+[^.]+[.]?",
        ".",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\.", ".", text)
    return text


def parse_description(raw_text: str) -> tuple[str, bool]:
    description = ""
    found_label = False

    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m_desc = re.match(r"(?:\*{0,2})Description\s*:\s*(?:\*{0,2})\s*(.+)", line, re.IGNORECASE)
        m_skip = re.match(
            r"(?:\*{0,2})(?:Confidence|cat|category|sev|severity|Action)\s*:\s*(?:\*{0,2})\s*(.+)",
            line,
            re.IGNORECASE,
        )
        if m_desc and not description:
            description = m_desc.group(1).strip()
            found_label = True
            continue
        if m_skip:
            continue

    if not description:
        cleaned = []
        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if re.match(r"(?:\*{0,2})(?:Confidence|cat|category|sev|severity|Action)\s*:", line, re.IGNORECASE):
                continue
            line = re.sub(r"^(?:\*{0,2})Description\s*:\s*(?:\*{0,2})\s*", "", line, flags=re.IGNORECASE)
            cleaned.append(line)
        description = " ".join(cleaned)

    description = _strip_trailing_boilerplate(description)
    description = _strip_action_clauses(description)
    description = extract_sentences(description, max_sentences=1)
    # Non-ASCII (e.g. Arabic from image signs) → treat as empty to force retry
    if description and any(ord(ch) > 0x02FF for ch in description):
        description = ""
    return description, found_label


def is_short_landmark_fragment(text: str, raw_text: str = "") -> bool:
    candidate = normalize_ws(text or raw_text).strip(" .,:;\"'").lower()
    if len(candidate.split()) >= 4:
        return False
    return bool(re.search(r"\b(?:the\s+)?u\.?\s*s\.?\b", candidate)) or candidate in {"us", "u s"}


# ─────────────────────── Cat classification (heuristic) ───────────────────────
def classify_cat(text: str) -> str:
    """v16-equivalent scoring. Avoids other_disaster unless no signal."""
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
    # Broader downed-tree/pole pattern (any location)
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
    """0 = little_or_none, 1 = mild, 2 = severe."""
    t = text.lower()

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
    multi_structure = re.search(
        r"\b(?:buildings|structures|houses|homes)\b", t
    ) and re.search(r"\b(?:collaps\w*|destroyed|damaged|in\s+ruins)\b", t)
    single_collapse = re.search(
        r"\b(?:building|structure|house|home|apartment|bridge|tower)\b", t
    ) and re.search(
        r"\b(?:collaps\w*|in\s+ruins|crumbled|caved\s+in|"
        r"completely\s+destroyed|entirely\s+destroyed|reduced\s+to\s+rubble)\b",
        t,
    )

    little = re.search(
        r"\b(?:minor|small|slight|superficial|limited|little|no\s+visible|no\s+major|"
        r"intact|undamaged|mostly\s+intact|barely|minimal|negligible)\b",
        t,
    )

    if severe_widespread or multi_structure or single_collapse:
        return 2
    if little:
        return 0
    return 1


# ─────────────────────── Quality gate ───────────────────────
FORBIDDEN_ACTIVITY_PATTERNS = [
    r"\bresponse action\b",
    r"\bfirst-response\b",
    r"\brecommend(?:ed)?\b",
    r"\b(?:should|must|will need to|needs to)\b",
]

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


def analyze_quality(text: str) -> dict:
    reasons = []
    penalty = 0
    text_l = text.lower()

    if not text or len(text.split()) < 5:
        reasons.append("weak_text")
        penalty += 3

    if len(text.split()) > 40:
        reasons.append("long_text")
        penalty += 1

    for pat in FORBIDDEN_ACTIVITY_PATTERNS:
        if re.search(pat, text_l):
            reasons.append(f"forbidden_activity:{pat}")
            penalty += 2

    if re.search(r"\bdue to\b|\bcaused by\b|\bbecause of\b|\bindicating\b|\bsuggesting\b", text_l):
        reasons.append("cause_guess")
        penalty += 1

    if any(ord(ch) > 0x02FF for ch in text):
        reasons.append("non_ascii")
        penalty += 4

    if re.match(r"^a scene of (?:destruction|devastation|extensive|severe)", text_l):
        reasons.append("generic_opener")
        penalty += 1

    if not any(t in text_l for t in LOCATION_TERMS):
        reasons.append("missing_location_anchor")
        penalty += 1

    visual_hits = sum(1 for t in VISUAL_TERMS if t in text_l)
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
    total_ms = 0
    debug: dict = {}

    raw1, ms1, lp1 = call_ollama(PI_B_PROMPT, img_b64, NUM_PREDICT)
    total_ms += ms1
    text1, had_labels1 = parse_description(raw1)
    ana1 = analyze_quality(text1)

    debug["primary_raw"] = raw1
    debug["primary_reasons"] = ana1["reasons"]
    debug["primary_penalty"] = ana1["penalty"]
    debug["primary_had_labels"] = had_labels1

    primary_ok = bool(text1) and len(text1.split()) >= 5 and ana1["passed"]
    if primary_ok:
        cat = classify_cat(text1)
        sev = classify_sev(text1)
        debug["chosen"] = "primary"
        return {
            "text": text1,
            "cat": cat,
            "sev": sev,
            "had_labels": had_labels1,
        }, lp1, total_ms, debug

    retry_prompt = PI_B_RETRY_PROMPT_TEMPLATE.format(prev_raw=(raw1 or "").strip())
    raw2, ms2, lp2 = call_ollama(retry_prompt, img_b64, NUM_PREDICT)
    total_ms += ms2
    text2, had_labels2 = parse_description(raw2)
    ana2 = analyze_quality(text2)

    debug["backup_raw"] = raw2
    debug["backup_reasons"] = ana2["reasons"]
    debug["backup_penalty"] = ana2["penalty"]
    debug["backup_had_labels"] = had_labels2

    def _score(text: str, ana: dict, had_labels: bool) -> int:
        s = 0
        if had_labels: s += 5
        if text and len(text.split()) >= 5: s += 3
        s -= ana["penalty"]
        s += ana["visual_hits"]
        return s

    s1 = _score(text1, ana1, had_labels1)
    s2 = _score(text2, ana2, had_labels2)

    candidates = [
        ("primary", text1, had_labels1, lp1, s1),
        ("backup", text2, had_labels2, lp2, s2),
    ]

    short_landmark_failure = (
        is_short_landmark_fragment(text1, raw1)
        or is_short_landmark_fragment(text2, raw2)
    )
    if short_landmark_failure:
        short_retry_prompt = PI_B_SHORT_OUTPUT_RETRY_PROMPT_TEMPLATE.format(
            prev_raw=normalize_ws(" ".join([raw1 or "", raw2 or ""]))
        )
        raw3, ms3, lp3 = call_ollama(short_retry_prompt, img_b64, NUM_PREDICT)
        total_ms += ms3
        text3, had_labels3 = parse_description(raw3)
        ana3 = analyze_quality(text3)

        debug["short_retry_raw"] = raw3
        debug["short_retry_reasons"] = ana3["reasons"]
        debug["short_retry_penalty"] = ana3["penalty"]
        debug["short_retry_had_labels"] = had_labels3

        s3 = _score(text3, ana3, had_labels3)
        candidates.append(("short_retry", text3, had_labels3, lp3, s3))

    chosen, text, had_labels, logprobs, _ = max(candidates, key=lambda item: item[4])
    debug["chosen"] = chosen

    if is_short_landmark_fragment(text) and short_landmark_failure:
        text = PI_B_WATER_LANDMARK_FALLBACK
        had_labels = False
        debug["chosen"] = "water_landmark_fallback"
        debug["fallback_reason"] = "short_landmark_fragment"

    cat = classify_cat(text)
    sev = classify_sev(text)
    return {
        "text": text,
        "cat": cat,
        "sev": sev,
        "had_labels": had_labels,
    }, logprobs, total_ms, debug


# ─────────────────────── M5 cost extraction ───────────────────────
def extract_m5_costs(rec: dict, image_id: str) -> dict:
    end_m5_cand = rec.get("end_m5") or rec.get("candidates", {}).get("end_m5", {})
    end_m5_lat = end_m5_cand.get("cost", {}).get("latency_ms", {})
    m5_pi_comm = rec.get("go_pi_b", rec.get("go_pi", rec.get("m5_pi_comm", {})))

    m5_total_ms = end_m5_lat.get("m5_total_ms", 0)
    if not m5_total_ms:
        m5_lat = rec.get("latency_ms", rec.get("latency", {}))
        m5_total_ms = m5_lat.get("m5_total_ms", 0)

    m5_pi_http_e2e_ms = m5_pi_comm.get("m5_pi_b_http_e2e_ms", m5_pi_comm.get("m5_pi_http_e2e_ms", 0))
    m5_pi_tx = m5_pi_comm.get("m5_pi_b_tx_bytes", m5_pi_comm.get("m5_pi_tx_bytes", 0))
    if not m5_pi_tx:
        m5_sig = rec.get("signal_overhead_bytes", {})
        m5_pi_tx = m5_sig.get("m5_pi_b_tx_bytes", m5_sig.get("m5_pi_tx_bytes", 0))

    m5_pi_rx = m5_pi_comm.get("m5_pi_b_rx_bytes", m5_pi_comm.get("m5_pi_rx_bytes", 0))

    return {
        "image_id": image_id,
        "m5_total_ms": m5_total_ms,
        "m5_pi_b_http_e2e_ms": m5_pi_http_e2e_ms,
        "m5_pi_b_tx_bytes": m5_pi_tx,
        "m5_pi_b_rx_bytes": m5_pi_rx,
    }


# ─────────────────────── Jetson forward (JSON only) ───────────────────────
def send_to_jetson(jetson_url: str, payload: dict) -> dict:
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    tx_bytes = len(payload_bytes)
    t0 = time.perf_counter()
    resp = requests.post(jetson_url, json=payload, timeout=JETSON_TIMEOUT)
    http_e2e_ms = int((time.perf_counter() - t0) * 1000)
    rx_bytes = len(resp.content)
    resp.raise_for_status()
    return {
        "pi_b_jet_http_e2e_ms": http_e2e_ms,
        "pi_b_jet_tx_bytes": tx_bytes,
        "pi_b_jet_rx_bytes": rx_bytes,
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
    print("[CONFIG] prompt : damage-scene description (v13 — structures/blocked/hazards)")
    print("[CONFIG] cat/sev: heuristic v17-equivalent (single_collapse, completely/entirely/totally)")
    print("[CONFIG] fixes  : non-ASCII retry, indicating/suggesting strip, generic_opener penalty")
    print("[CONFIG] fallback: short landmark fragment retry + conservative water/flood fallback")

    done_ids = load_done_ids(out_p) if RESUME else set()
    if done_ids:
        print(f"[RESUME] found {len(done_ids)} already-done records → skipping them")
    elif RESUME:
        print(f"[RESUME] no previous output found at {out_p} → starting from scratch")

    jetson_ok = bool(JETSON_URL)
    if JETSON_URL:
        try:
            base = JETSON_URL.rsplit("/", 1)[0]
            requests.get(base + "/health", timeout=3)
            print("[CHECK] Jetson reachable")
        except Exception as e:
            print(f"[CHECK] Jetson health check failed, will try POST anyway: {e}")
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
                    {"type": "pi_b_oracle", "image_id": image_id, "status": "fail", "error": "image_not_found"})
                n_fail += 1
                continue

            seq = n_ok + n_fail + 1
            print(f"[{seq}] {image_id}")

            costs = extract_m5_costs(rec, image_id)
            t_pi_start = time.perf_counter()

            try:
                img_b64 = b64_image(img_path)
                result, logprobs, vlm_ms, debug_info = run_pi_inference(img_b64)
            except Exception as e:
                print(f"  [FAIL] vlm error: {e}")
                append_jsonl(str(out_p),
                    {"type": "pi_b_oracle", "image_id": image_id, "ts": now_kst(),
                     "status": "fail", "error": f"vlm_fail: {e}"})
                n_fail += 1
                if LIMIT and (n_ok + n_fail) >= LIMIT:
                    break
                continue

            text = result["text"]
            cat = result["cat"]
            sev = result["sev"]
            had_labels = result["had_labels"]

            if not text or len(text.split()) < 4:
                append_jsonl(str(out_p),
                    {"type": "pi_b_oracle", "image_id": image_id, "ts": now_kst(),
                     "status": "fail", "error": f"no_vlm_output (text='{text}')"})
                n_fail += 1
                if LIMIT and (n_ok + n_fail) >= LIMIT:
                    break
                continue

            final_quality = analyze_quality(text)
            _quality_issues = int(not final_quality["passed"])
            _retry_used = int(debug_info.get("chosen") == "backup")
            semantic_penalty = min(2, _quality_issues + _retry_used)
            conf_final, conf_detail = compute_confidence(logprobs, had_labels, semantic_penalty=semantic_penalty)
            pi_b_total_ms = int((time.perf_counter() - t_pi_start) * 1000)

            jetson_meas = {
                "pi_b_jet_http_e2e_ms": 0,
                "pi_b_jet_tx_bytes": 0,
                "pi_b_jet_rx_bytes": 0,
                "status": "skipped" if not JETSON_URL else "pending",
            }

            if jetson_ok:
                jetson_payload = {
                    "type": "pi_b_result",
                    "run_id": run_id,
                    "image_id": image_id,
                    "ts": now_kst(),
                    "pi_b_text": text,
                    "pi_b_cat": cat,
                    "pi_b_sev": sev,
                    "pi_b_conf": conf_final,
                    "pi_b_conf_detail": conf_detail,
                    "m5_total_ms": costs["m5_total_ms"],
                    "m5_pi_b_http_e2e_ms": costs["m5_pi_b_http_e2e_ms"],
                    "pi_b_total_ms": pi_b_total_ms,
                }
                try:
                    jetson_meas = send_to_jetson(JETSON_URL, jetson_payload)
                except Exception as e:
                    jetson_meas["status"] = f"fail: {e}"
                    jetson_ok = False
                    print(f"  [JETSON] failed, disabling: {e}")

            end_pi_latency = costs["m5_total_ms"] + costs["m5_pi_b_http_e2e_ms"] + pi_b_total_ms
            end_pi_overhead = costs["m5_pi_b_tx_bytes"] + costs["m5_pi_b_rx_bytes"]

            oracle_rec = {
                "type": "pi_b_oracle",
                "run_id": run_id,
                "image_id": image_id,
                "ts": now_kst(),
                "status": "ok",
                "end_pi": {
                    "rep_output": "Pi_B",
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
                            "m5_pi_b_http_e2e_ms": costs["m5_pi_b_http_e2e_ms"],
                            "pi_b_total_ms": pi_b_total_ms,
                            "pi_vlm_ms": vlm_ms,
                        },
                        "signal_overhead_bytes": {
                            "total": end_pi_overhead,
                            "m5_pi_b_tx_bytes": costs["m5_pi_b_tx_bytes"],
                            "m5_pi_b_rx_bytes": costs["m5_pi_b_rx_bytes"],
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
            print(f"  text: {text}")
            print(f"  cat={cat} sev={sev} conf={conf_final} mean_lp={lp_str} min_lp={min_lp_str}")
            print(f"  vlm={vlm_ms}ms pi={pi_b_total_ms}ms chosen={debug_info.get('chosen')}")
            print(f"  quality: passed={final_quality['passed']} penalty={final_quality['penalty']} "
                  f"reasons={final_quality['reasons']}")
            print(f"  END@Pi: L={end_pi_latency}ms O={end_pi_overhead}B jetson={jetson_meas['status']}")

            if LIMIT and (n_ok + n_fail) >= LIMIT:
                break

    print(f"\n[DONE] ok={n_ok}  skip={n_skip}  fail={n_fail}  -> {out_p}")


if __name__ == "__main__":
    main()
