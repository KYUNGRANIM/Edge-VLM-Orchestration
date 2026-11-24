#!/usr/bin/env python3
"""
GPT image-grounded judge for tier text quality.

Purpose
-------
This script evaluates whether each representative tier answer
(M5_A, Pi_A, Jetson) is visually grounded in the input disaster image.

It is intentionally separate from cat/sev accuracy:
  - cat/sev remain GT-based structured quality components.
  - GPT judge replaces or complements BERTScore for the text component.

Default behavior
----------------
One API call per image. The call includes:
  - the image
  - M5_A, Pi_A, Jetson text answers from the collection dataset

Output
------
final_pipeline_codex/outputs/00_gpt_judge_scores.jsonl

Each line contains per-tier scores and a normalized text_judge_score in [0, 1].

Safety
------
No API key is embedded. Set OPENAI_API_KEY in the environment.
Use --dry-run or --estimate-only before running paid evaluation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


# 00_gpt_judge.py lives at profiling/code/pipeline/ -> parents[3] is the m5_vlm root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "profiling" / "results" / "pipeline_outputs"

DEFAULT_COLLECTION = PROJECT_ROOT / "profiling" / "results" / "collection.jsonl"
DEFAULT_IMAGE_ROOT = Path("/Volumes/CORSAIR/MEDIC_v2_train/images_sampled")
FALLBACK_IMAGE_ROOTS = (
    Path("/Volumes/CORSAIR/MEDIC_v2_train/images_sampled"),
)
DEFAULT_OUT = OUTPUT_DIR / "00_gpt_judge_scores.jsonl"

KST = timezone(timedelta(hours=9))
TIERS = ("m5_a", "pi_a", "jetson")
VALID_TIER = set(TIERS)
CANDIDATES = ("candidate_1", "candidate_2", "candidate_3")

DEFAULT_MODEL = os.getenv("GPT_JUDGE_MODEL", "gpt-4o")
DEFAULT_DETAIL = os.getenv("GPT_JUDGE_DETAIL", "high")

PRICES_PER_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

IMAGE_TOKEN_TABLE = {
    "gpt-4o": {"low_base": 85, "high_base": 85, "high_tile": 170},
    # Current docs list higher image token accounting for 4o-mini.
    "gpt-4o-mini": {"low_base": 2833, "high_base": 2833, "high_tile": 5667},
}


SYSTEM_PROMPT = """You are a strict but practical evaluator for disaster-response vision-language outputs.
Judge whether each candidate report correctly describes the visible damage/hazard
and recommends a reasonable field response for that visible situation.

Important distinction:
- The damage/situation description must be grounded in visible image evidence.
- The response/action may mention reasonable tools, crews, equipment, pumps,
  cranes, barricades, firefighters, inspection teams, or evacuation actions
  even if they are not visible, as long as they are appropriate for the visible
  damage/hazard.

Do not penalize a response just because it is somewhat generic when the image
itself contains limited or generic disaster evidence. Penalize it only when it
is empty, mismatched to the visible situation, or purely long-term reconstruction.

Candidate reports may use different valid response styles:
- assessment-oriented: inspect/check/evaluate a specific visible risk;
- repair/equipment-oriented: use pumps, cranes, tools, crews, or repairs that
  are reasonable for the visible damage;
- first-response-oriented: clear, cordon, evacuate, extinguish, stabilize, drain;
- generic safety-oriented: secure or organize response when the image itself has
  limited visible damage or shows logistics/people rather than a concrete hazard.

Do not prefer one style over another. Score by correctness of the described
damage/hazard and appropriateness of the recommended action for that damage.
Do not reward imperative wording by itself, and do not penalize descriptive
wording when it identifies an appropriate ongoing or needed response.
Return JSON only."""


USER_PROMPT_TEMPLATE = """Evaluate three candidate disaster reports for the same image.

Candidate reports:

[candidate_1]
text: {candidate_1_text}

[candidate_2]
text: {candidate_2_text}

[candidate_3]
text: {candidate_3_text}

Score each candidate independently using integer values from 0 to 5:
1. damage_grounding:
   Is the damage/hazard/situation description consistent with visible image evidence?
   0 = wrong or unrelated situation, 5 = strongly grounded.
2. damage_overclaim:
   Does it invent or exaggerate visible damage, hazard, cause, people, vehicles, fire,
   flooding, collapse, injuries, deaths, or trapped people?
   0 = no overclaim, 5 = severe overclaim.
3. response_appropriateness:
   Is the recommended response appropriate for the visible damage/hazard?
   Tools, crews, pumps, cranes, barricades, firefighters, or inspection teams do not
   need to be visible if they are reasonable for the situation.
   0 = inappropriate, 5 = highly appropriate.
4. response_operationality:
   Is the response practically useful for handling the visible situation, regardless
   of whether it is written as a command, description, or observation?
   5 = clearly useful field response or observed response aligned with the damage;
   4 = relevant assessment/inspection/safety action tied to a visible risk;
   3 = generally relevant but somewhat generic;
   2 = weak, long-term, or only indirectly useful;
   0-1 = irrelevant, unsafe, empty, non-operational, or prayer/comfort only.
5. response_target_alignment:
   Is the response tied to the actual visible target or risk?
   Examples: flooded street -> drain/restrict traffic; fallen tree -> cut/lift/close road;
   cracked wall -> cordon/shore/inspect that wall.
   0 = wrong target, 5 = clearly aligned.

Assessment/inspection actions are acceptable when tied to a specific visible risk.
Do not penalize reasonable emergency equipment or crews merely because they are not
shown in the image. Penalize unsupported damage claims more strongly than reasonable
response extrapolation.

Response-style fairness rule:
Candidate responses may be assessment-oriented, repair/equipment-oriented,
first-response-oriented, or generic safety-oriented. Do not prefer one style.
Do not reward imperative sentences merely because they sound like commands.
Do not penalize plain descriptive sentences if they correctly identify an
appropriate response already happening or needed for the visible damage.
For example, "use a crane to lift the fallen tree" can be appropriate even if
the crane is not visible; "inspect the cracked wall" can be appropriate when
a structural risk is visible; "organize volunteers" can be appropriate for a
logistics/donation image with little visible physical damage.

Also provide a short reason for each candidate, maximum 20 words.

Return exactly this JSON shape:
{{
  "candidate_1": {{
    "damage_grounding": 0,
    "damage_overclaim": 0,
    "response_appropriateness": 0,
    "response_operationality": 0,
    "response_target_alignment": 0,
    "reason": ""
  }},
  "candidate_2": {{
    "damage_grounding": 0,
    "damage_overclaim": 0,
    "response_appropriateness": 0,
    "response_operationality": 0,
    "response_target_alignment": 0,
    "reason": ""
  }},
  "candidate_3": {{
    "damage_grounding": 0,
    "damage_overclaim": 0,
    "response_appropriateness": 0,
    "response_operationality": 0,
    "response_target_alignment": 0,
    "reason": ""
  }},
  "best_text_candidate": "candidate_1",
  "global_notes": ""
}}"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_collection(path: Path) -> dict[str, dict[str, Any]]:
    return {r["image_id"]: r for r in read_jsonl(path)}


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("image_id"):
                done.add(row["image_id"])
    return done


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _candidate_image_paths(image_id: str, root: Path) -> list[Path]:
    cat = image_id.split("__", 1)[0]
    return [
        root / image_id,
        root / cat / image_id,
    ]


def build_image_index(image_root: Path) -> dict[str, Path]:
    """Index image files once so folder-per-category layouts are fast."""
    index: dict[str, Path] = {}
    roots = []
    for root in (image_root, *FALLBACK_IMAGE_ROOTS):
        if root not in roots:
            roots.append(root)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts and path.name not in index:
                index[path.name] = path
    return index


def resolve_image_path(image_id: str, image_root: Path, image_index: dict[str, Path] | None = None) -> Path | None:
    if image_index and image_id in image_index:
        return image_index[image_id]

    roots = []
    for root in (image_root, *FALLBACK_IMAGE_ROOTS):
        if root not in roots:
            roots.append(root)

    for root in roots:
        for candidate in _candidate_image_paths(image_id, root):
            if candidate.is_file():
                return candidate

    for root in roots:
        matches = list(root.rglob(image_id)) if root.is_dir() else []
        if matches:
            return matches[0]
    return None


def clamp_int_score(value: Any) -> int:
    try:
        x = int(round(float(value)))
    except (TypeError, ValueError):
        x = 0
    return max(0, min(5, x))


def compute_text_judge_score(tier_score: dict[str, Any]) -> float:
    """Normalize GPT judge sub-scores to [0, 1].

    V4 separates visible-damage description from response usefulness:
      - damage must be grounded in the image
      - response may reasonably extrapolate tools/crews/actions if appropriate
      - operational usefulness is style-neutral, not command-wording dependent
    """
    damage = clamp_int_score(tier_score.get("damage_grounding"))
    overclaim = clamp_int_score(tier_score.get("damage_overclaim"))
    response = clamp_int_score(tier_score.get("response_appropriateness"))
    operational = clamp_int_score(tier_score.get("response_operationality"))
    target = clamp_int_score(tier_score.get("response_target_alignment"))

    raw = (
        0.30 * damage
        + 0.25 * response
        + 0.20 * operational
        + 0.15 * target
        + 0.10 * (5 - overclaim)
    )
    return round(raw / 5.0, 4)


def validate_and_enrich(
    parsed: dict[str, Any],
    candidate_to_tier: dict[str, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for candidate in CANDIDATES:
        tier = candidate_to_tier[candidate]
        item = parsed.get(candidate)
        if not isinstance(item, dict):
            item = {}
        enriched = {
            "damage_grounding": clamp_int_score(item.get("damage_grounding")),
            "damage_overclaim": clamp_int_score(item.get("damage_overclaim")),
            "response_appropriateness": clamp_int_score(item.get("response_appropriateness")),
            "response_operationality": clamp_int_score(item.get("response_operationality")),
            "response_target_alignment": clamp_int_score(item.get("response_target_alignment")),
            "reason": str(item.get("reason", ""))[:240],
        }
        enriched["text_judge_score"] = compute_text_judge_score(enriched)
        out[tier] = enriched

    best_candidate = parsed.get("best_text_candidate")
    if best_candidate in candidate_to_tier:
        best = candidate_to_tier[best_candidate]
    else:
        best = max(TIERS, key=lambda t: out[t]["text_judge_score"])
    out["best_text_tier"] = best
    out["best_text_candidate"] = best_candidate if best_candidate in candidate_to_tier else ""
    out["global_notes"] = str(parsed.get("global_notes", ""))[:300]
    return out


def extract_json_object(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def candidate_order(image_id: str) -> tuple[str, ...]:
    """Use deterministic per-image shuffling to reduce order and tier-name bias."""
    seed = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    tiers = list(TIERS)
    rng.shuffle(tiers)
    return tuple(tiers)


def build_user_prompt(image_id: str, record: dict[str, Any]) -> tuple[str, dict[str, str]]:
    ordered_tiers = candidate_order(image_id)
    candidate_to_tier = {
        candidate: tier for candidate, tier in zip(CANDIDATES, ordered_tiers)
    }
    prompt = USER_PROMPT_TEMPLATE.format(
        candidate_1_text=record[candidate_to_tier["candidate_1"]].get("text", ""),
        candidate_2_text=record[candidate_to_tier["candidate_2"]].get("text", ""),
        candidate_3_text=record[candidate_to_tier["candidate_3"]].get("text", ""),
    )
    return prompt, candidate_to_tier


def call_judge(
    client: Any,
    model: str,
    detail: str,
    image_b64: str,
    prompt: str,
    candidate_to_tier: dict[str, str],
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any], str, dict[str, Any], int]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": detail,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    raw = (resp.choices[0].message.content or "").strip()
    parsed = validate_and_enrich(extract_json_object(raw), candidate_to_tier)
    # "gpt-4o" is an alias; keep the dated snapshot the API actually served (reviewers ask for the version).
    usage = {"resolved_model": getattr(resp, "model", None),
             "system_fingerprint": getattr(resp, "system_fingerprint", None)}
    if getattr(resp, "usage", None):
        usage |= {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }
    return parsed, raw, usage, elapsed_ms


def estimate_image_tokens(path: Path | None, model: str, detail: str) -> int:
    table = IMAGE_TOKEN_TABLE.get(model, IMAGE_TOKEN_TABLE["gpt-4o"])
    if detail == "low":
        return table["low_base"]
    if detail not in {"high", "auto"}:
        return table["low_base"]

    if path is None:
        return table["high_base"] + 4 * table["high_tile"]

    try:
        from PIL import Image

        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        return table["high_base"] + 4 * table["high_tile"]

    scale = min(2048 / max(w, h), 1.0)
    w2 = w * scale
    h2 = h * scale
    shortest = min(w2, h2)
    if shortest > 0:
        scale2 = 768 / shortest
        w2 *= scale2
        h2 *= scale2
    tiles = math.ceil(w2 / 512) * math.ceil(h2 / 512)
    return table["high_base"] + tiles * table["high_tile"]


def estimate_cost(
    records: list[tuple[str, dict[str, Any], Path | None]],
    model: str,
    detail: str,
    max_output_tokens: int,
) -> dict[str, float]:
    price = PRICES_PER_1M.get(model, PRICES_PER_1M["gpt-4o"])
    image_tokens = 0
    text_tokens = 0
    for image_id, record, image_path in records:
        image_tokens += estimate_image_tokens(image_path, model, detail)
        prompt, _ = build_user_prompt(image_id, record)
        text_tokens += max(1, len(prompt) // 4)

    output_tokens = len(records) * max_output_tokens
    input_tokens = image_tokens + text_tokens
    input_cost = input_tokens / 1_000_000 * price["input"]
    output_cost = output_tokens / 1_000_000 * price["output"]
    return {
        "records": len(records),
        "input_tokens_est": float(input_tokens),
        "output_tokens_est_max": float(output_tokens),
        "input_cost_est": round(input_cost, 4),
        "output_cost_est_max": round(output_cost, 4),
        "total_cost_est_max": round(input_cost + output_cost, 4),
    }


def build_work_items(
    collection: dict[str, dict[str, Any]],
    image_root: Path,
    done_ids: set[str],
    resume: bool,
    require_images: bool,
) -> tuple[list[tuple[str, dict[str, Any], Path | None]], dict[str, int]]:
    items = []
    stats = {"missing_image": 0, "already_done": 0}
    image_index = build_image_index(image_root)
    for image_id in sorted(collection):
        if resume and image_id in done_ids:
            stats["already_done"] += 1
            continue
        image_path = resolve_image_path(image_id, image_root, image_index=image_index)
        if image_path is None:
            stats["missing_image"] += 1
            if require_images:
                continue
        items.append((image_id, collection[image_id], image_path))
    return items, stats


def make_output_record(
    image_id: str,
    record: dict[str, Any],
    image_path: Path,
    model: str,
    detail: str,
    judged: dict[str, Any],
    candidate_to_tier: dict[str, str],
    raw: str,
    usage: dict[str, Any],
    latency_ms: int,
) -> dict[str, Any]:
    return {
        "type": "gpt_text_judge",
        "status": "ok",
        "run_ts": datetime.now(KST).isoformat(timespec="milliseconds"),
        "image_id": image_id,
        "image_path": str(image_path),
        "model": model,
        "detail": detail,
        "candidate_to_tier": candidate_to_tier,
        "tier_outputs": {
            tier: {
                "cat": record[tier].get("cat", ""),
                "sev": record[tier].get("sev", ""),
                "text": record[tier].get("text", ""),
            }
            for tier in TIERS
        },
        "scores": {tier: judged[tier] for tier in TIERS},
        "best_text_tier": judged["best_text_tier"],
        "best_text_candidate": judged.get("best_text_candidate", ""),
        "global_notes": judged.get("global_notes", ""),
        "usage": usage,
        "latency_ms": latency_ms,
        "raw": raw,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GPT-4o image-grounded text judge.")
    p.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    p.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--detail", choices=["low", "high", "auto"], default=DEFAULT_DETAIL)
    p.add_argument("--limit", type=int, default=0, help="0 means all records")
    p.add_argument("--max-tokens", type=int, default=900)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--estimate-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.collection.is_file():
        print(f"[ERROR] collection not found: {args.collection}", file=sys.stderr)
        return 1

    collection = load_collection(args.collection)
    done_ids = load_done_ids(args.out) if args.resume else set()
    require_images = not (args.dry_run or args.estimate_only)
    items, stats = build_work_items(
        collection,
        args.image_root,
        done_ids,
        args.resume,
        require_images=require_images,
    )
    if args.limit > 0:
        items = items[: args.limit]

    print("[CONFIG]")
    print(f"  collection : {args.collection}")
    print(f"  image_root : {args.image_root}")
    print(f"  output     : {args.out}")
    print(f"  model      : {args.model}")
    print(f"  detail     : {args.detail}")
    print(f"  records    : {len(items)} selected")
    print(f"  skipped    : {stats}")

    estimate = estimate_cost(items, args.model, args.detail, args.max_tokens)
    print("[ESTIMATE]")
    print(json.dumps(estimate, indent=2))

    if args.estimate_only:
        return 0

    if args.dry_run:
        print("[DRY-RUN] No API calls will be made.")
        for image_id, record, image_path in items[:3]:
            path_label = str(image_path) if image_path is not None else "(image not found in this session)"
            _, candidate_to_tier = build_user_prompt(image_id, record)
            print(f"  {image_id} -> {path_label}")
            print(f"    candidate_to_tier: {candidate_to_tier}")
            print(f"    m5 : {record['m5_a'].get('text', '')[:90]}")
        return 0

    if OpenAI is None:
        print("[ERROR] openai package is not installed.", file=sys.stderr)
        return 1

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERROR] Set OPENAI_API_KEY before running paid GPT judge.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key)

    n_ok = 0
    n_fail = 0
    with args.out.open("a", encoding="utf-8") as f:
        for idx, (image_id, record, image_path) in enumerate(items, 1):
            print(f"[{idx}/{len(items)}] {image_id}")
            prompt, candidate_to_tier = build_user_prompt(image_id, record)
            image_b64 = encode_image(image_path)

            last_error = ""
            for attempt in range(args.max_retries + 1):
                try:
                    judged, raw, usage, latency_ms = call_judge(
                        client=client,
                        model=args.model,
                        detail=args.detail,
                        image_b64=image_b64,
                        prompt=prompt,
                        candidate_to_tier=candidate_to_tier,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    out = make_output_record(
                        image_id=image_id,
                        record=record,
                        image_path=image_path,
                        model=args.model,
                        detail=args.detail,
                        judged=judged,
                        candidate_to_tier=candidate_to_tier,
                        raw=raw,
                        usage=usage,
                        latency_ms=latency_ms,
                    )
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                    best = out["best_text_tier"]
                    scores = {t: out["scores"][t]["text_judge_score"] for t in TIERS}
                    print(f"  ok best={best} scores={scores} latency={latency_ms}ms")
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    print(f"  retry {attempt + 1}/{args.max_retries + 1}: {last_error[:220]}")
                    if attempt < args.max_retries:
                        time.sleep(2.0 + attempt)
                    else:
                        fail = {
                            "type": "gpt_text_judge",
                            "status": "error",
                            "run_ts": datetime.now(KST).isoformat(timespec="milliseconds"),
                            "image_id": image_id,
                            "image_path": str(image_path),
                            "model": args.model,
                            "detail": args.detail,
                            "error": last_error,
                        }
                        f.write(json.dumps(fail, ensure_ascii=False) + "\n")
                        f.flush()
                        n_fail += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

    print("[DONE]")
    print(f"  ok   : {n_ok}")
    print(f"  fail : {n_fail}")
    print(f"  out  : {args.out}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
