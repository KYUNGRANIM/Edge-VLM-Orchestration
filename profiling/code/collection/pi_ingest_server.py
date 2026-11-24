#!/usr/bin/env python3
"""
Pi Ingest Server  –  M5→Pi Oracle Dataset (v3-paper-aligned)
=============================================================
Each M5 (A/B/C) POSTs:
  - multipart: meta(JSON) + image(JPEG)  →  /ingest_m5
  - json: http_e2e latency/overhead      →  /patch_e2e

This server:
  1. Saves every incoming record   → raw_ingest__{run_id}.jsonl
  2. Saves uploaded images         → tier2_ingest/images/{image_id}
  3. Caches partial results in memory (+ disk backup)
  4. When merge-ready (A + A patch, with optional delayed partial for missing B/C)
     → merged_oracle__{run_id}.jsonl (Paper-aligned schema)
  5. Saves patches                 → oracle_patch__{run_id}.jsonl
  6. /status for live monitoring

Merged record schema (paper §2-2):
  - m5_a_text, m5_b_text, m5_c_text (Main + Aux)
    - cat, sev (from M5_A payload, fallback to keyword extraction)
  - latency_ms.m5_total_ms, m5_sd_read_ms, m5_decode_resize_ms,
    m5_uart_ms, m5_vlm_infer_ms, m5_postprocess_ms
  - latency_ms.m5_pi_http_e2e_ms (from PATCH, 0 if not yet received)
  - signal_overhead_bytes.m5_pi_tx_bytes, m5_pi_rx_bytes

Usage:
    pip install fastapi uvicorn python-multipart
    python3 pi_server.py
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# ─────────────────────── Config ───────────────────────
# Oracle collection/merge server.
# Online implementation uses pi_a_online_server.py / pi_b_online_server.py instead.
# Override when needed, e.g. PI_SAVE_DIR=./tier2_fire python pi_server.py
SAVE_DIR = os.environ.get("PI_SAVE_DIR", "./tier2_ingest")
IMG_DIR  = os.path.join(SAVE_DIR, "images")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

KST = timezone(timedelta(hours=9))
REQUIRED_ROLES = {"a", "b", "c"}
PARTIAL_MERGE_WAIT_SEC = 300.0  # 5분 — B/C가 같은 이미지 처리할 시간 충분히 대기
# image_id가 같은데 run_id까지 같은 케이스를 기준으로 late patch 캐시를 구분한다.
# key: (run_id, image_id)
late_patch_cache: dict[tuple[str, str], dict[str, Any]] = {}

PENDING_BACKUP_PATH    = os.path.join(SAVE_DIR, "_pending_cache.json")
PENDING_BACKUP_INTERVAL = 5   # save to disk every N new arrivals
REQUIRE_NONEMPTY_M5_TEXT = True

STREAM_CHUNK = 1024 * 1024  # 1MB for file copy

def _safe_int(val) -> Optional[int]:
    """Convert to int safely, return None if not possible."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def _safe_float(val) -> Optional[float]:
    """Convert to float safely, return None if not possible."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_sev_0_2(val) -> Optional[int]:
    iv = _safe_int(val)
    if iv is None:
        return None
    if iv < 0:
        iv = 0
    if iv > 2:
        iv = 2
    return iv


def _safe_conf_1_5(val) -> Optional[int]:
    """Normalize confidence to integer 1..5."""
    fv = _safe_float(val)
    if fv is None:
        return None

    # Native 1..5 confidence
    if 1.0 <= fv <= 5.0:
        iv = int(round(fv))
        return 1 if iv < 1 else (5 if iv > 5 else iv)

    # Backward compatibility: normalized confidence 0..1
    if 0.0 <= fv <= 1.0:
        iv = int(round(1.0 + 4.0 * fv))
        return 1 if iv < 1 else (5 if iv > 5 else iv)

    # Percent-like values (e.g., 87)
    if 5.0 < fv <= 100.0:
        iv = int(round(1.0 + 4.0 * (fv / 100.0)))
        return 1 if iv < 1 else (5 if iv > 5 else iv)

    return None


def parse_a_output(raw_text: Optional[str]) -> dict:
    """Extract cat/sev from A's plain text using keywords."""
    text = str(raw_text).strip() if raw_text is not None else ""
    low = text.lower()

    # category: keyword matching
    if "landslide" in low or "mudslide" in low or "rockslide" in low:
        cat = "landslide"
    elif "earthquake" in low or "quake" in low:
        cat = "earthquake"
    elif "flood" in low or "floodwater" in low:
        cat = "flood"
    elif "hurricane" in low or "cyclone" in low or "typhoon" in low:
        cat = "hurricane"
    elif "fire" in low or "wildfire" in low or "blaze" in low:
        cat = "fire"
    else:
        cat = "landslide"  # dataset default

    # severity: word intensity
    score = 0
    for w in ["severe", "massive", "devastating", "catastroph", "destroy"]:
        if w in low:
            score += 2
    for w in ["significant", "major", "extensive", "large", "serious"]:
        if w in low:
            score += 1
    for w in ["damage", "collaps", "block", "threat", "risk", "hazard", "urgent"]:
        if w in low:
            score += 1
    sev = 2 if score >= 3 else (1 if score >= 1 else 0)

    return {"cat": cat, "sev": sev, "text": (text if text else None)}


def _role_from_device(device: str) -> str:
    if not isinstance(device, str):
        return ""
    d = device.strip().lower()
    if d.startswith("m5_"):
        d = d[3:]
    if d in REQUIRED_ROLES:
        return d
    return d[-1:] if d else ""


def _extract_a_output(a: dict[str, Any]) -> dict[str, Any]:
    """Use A payload cat/sev/conf when available, fallback to keyword extraction."""
    parsed = parse_a_output(a.get("text"))
    out = {
        "text": parsed["text"],
        "cat": parsed["cat"],
        "sev": parsed["sev"],
        "conf": None,
    }

    # cat: payload 우선, 없으면 keyword 폴백
    cat = a.get("cat")
    if cat is None:
        pass  # keep keyword-based fallback
    elif isinstance(cat, str) and cat.strip():
        out["cat"] = cat.strip()

    # sev: payload 우선
    sev = a.get("sev")
    try:
        if sev is not None:
            sev_int = _safe_sev_0_2(sev)
            if sev_int is not None:
                out["sev"] = sev_int
    except (TypeError, ValueError):
        pass  # keep keyword-based fallback

    # conf: new field
    out["conf"] = _safe_conf_1_5(a.get("conf"))

    return out


def _meta_has_valid_text(meta_obj: dict[str, Any]) -> bool:
    text = meta_obj.get("text")
    if text is None:
        return False
    if not isinstance(text, str):
        return False
    return len(text.strip()) > 0


def _is_m5a_coordinator_meta(meta_obj: dict[str, Any]) -> bool:
    """True for the online M5_A Stage-1 coordinator payload.

    In the implementation run, M5_B/C send lightweight JSON to M5_A. M5_A then
    uploads one multipart request containing the image plus all M5_A/B/C outputs.
    This payload intentionally has no legacy role field, so it needs a separate
    ingest path from the original three-role oracle collector.
    """
    return (
        str(meta_obj.get("type", "")) == "m5a_stage1_coordinator"
        and isinstance(meta_obj.get("m5_a"), dict)
        and isinstance(meta_obj.get("m5_b"), dict)
        and isinstance(meta_obj.get("m5_c"), dict)
    )


def _cache_late_patch(run_id: str, image_id: str, role: str, patch: dict[str, Any]):
    key = (str(run_id), os.path.basename(str(image_id)))
    if key not in late_patch_cache:
        late_patch_cache[key] = {}
    late_patch_cache[key][role] = {
        "m5_pi_http_e2e_ms": int(patch.get("m5_pi_http_e2e_ms", 0) or 0),
        "m5_pi_tx_bytes": int(patch.get("m5_pi_tx_bytes", 0) or 0),
        "m5_pi_rx_bytes": int(patch.get("m5_pi_rx_bytes", 0) or 0),
    }


def _load_late_patch_cache():
    late_patch_cache.clear()
    for fname in os.listdir(SAVE_DIR):
        if not fname.startswith("oracle_patch__") or not fname.endswith(".jsonl"):
            continue
        path = os.path.join(SAVE_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    run_id = str(rec.get("run_id", "unknown"))
                    image_id = os.path.basename(str(rec.get("image_id", "")))
                    role = _role_from_device(str(rec.get("device", "")))
                    if role not in REQUIRED_ROLES or not image_id:
                        continue
                    _cache_late_patch(
                        run_id,
                        image_id,
                        role,
                        rec,
                    )
        except Exception as e:
            print(f"[WARN] failed to load patch cache from {path}: {e}")
            continue

    cache_cnt = sum(len(v) for v in late_patch_cache.values())
    print(f"[RESTORE] loaded late patch cache: {len(late_patch_cache)} images, {cache_cnt} roles")


# ─────────────────────── App ───────────────────────
app = FastAPI(title="M5→Pi Oracle Merge Ingest (v3-paper-aligned)")

# in-memory merge cache:  image_id -> { "a": {...}, "b": {...}, "c": {...}, "_patches": {...} }
pending: dict[str, dict[str, Any]] = {}
pending_lock = threading.Lock()

stats = {
    "raw_received": 0,
    "images_saved": 0,
    "merged": 0,
    "patches": 0,
    "patches_into_pending": 0,
    "patches_late": 0,
    "last_recv_ts": None,
    "last_merge_ts": None,
}


# ─────────────────────── Helpers ───────────────────────
def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="milliseconds")


def raw_path(run_id: str) -> str:
    return os.path.join(SAVE_DIR, f"raw_ingest__{run_id}.jsonl")


def merged_path(run_id: str) -> str:
    return os.path.join(SAVE_DIR, f"merged_oracle__{run_id}.jsonl")


def patch_path(run_id: str) -> str:
    return os.path.join(SAVE_DIR, f"oracle_patch__{run_id}.jsonl")


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def save_upload_file(upload: UploadFile, dst_path: str) -> int:
    """Save uploaded file to disk, return file size in bytes."""
    await run_in_threadpool(upload.file.seek, 0)
    with open(dst_path, "wb") as f:
        await run_in_threadpool(shutil.copyfileobj, upload.file, f, STREAM_CHUNK)
    return os.path.getsize(dst_path)


# ─────────── Pending cache disk backup ───────────
def save_pending_to_disk():
    tmp = PENDING_BACKUP_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pending, f, ensure_ascii=False)
        os.replace(tmp, PENDING_BACKUP_PATH)
    except Exception as e:
        print(f"[WARN] pending backup failed: {e}")


def load_pending_from_disk():
    global pending
    if not os.path.exists(PENDING_BACKUP_PATH):
        return
    try:
        with open(PENDING_BACKUP_PATH, "r", encoding="utf-8") as f:
            pending = json.load(f)
        print(f"[RESTORE] loaded {len(pending)} pending entries from disk")
    except Exception as e:
        print(f"[WARN] pending restore failed: {e}")
        pending = {}


def _apply_cached_patches(entry: dict[str, Any], image_id: str) -> bool:
    """Merge late-cache patch values into pending entry before final merge."""
    run_id = str(entry.get("_run_id", "unknown"))
    key = (run_id, image_id)
    cached = late_patch_cache.get(key, {})
    if not cached:
        return False

    if "_patches" not in entry:
        entry["_patches"] = {}

    changed = False
    for role in REQUIRED_ROLES:
        if role not in entry["_patches"] and role in cached:
            entry["_patches"][role] = cached[role]
            changed = True

    if "_coordinator" in entry:
        if "a" in entry.get("_patches", {}):
            late_patch_cache.pop(key, None)
    elif all(r in entry.get("_patches", {}) for r in REQUIRED_ROLES):
        late_patch_cache.pop(key, None)

    return changed


def _is_merge_ready(entry: dict[str, Any]) -> bool:
    """Merge only when all 3 roles (a,b,c) arrived for same image_id."""
    if "_coordinator" in entry:
        return "a" in entry.get("_patches", {})
    have_roles = {r for r in REQUIRED_ROLES if r in entry}
    return have_roles == REQUIRED_ROLES


def _entry_age_sec(entry: dict[str, Any]) -> float:
    ts = entry.get("_first_wall_ts")
    if isinstance(ts, (float, int)):
        return max(0.0, time.time() - float(ts))
    iso = entry.get("_ts_first")
    if isinstance(iso, str) and iso:
        try:
            dt = datetime.fromisoformat(iso)
            return max(0.0, (datetime.now(KST) - dt).total_seconds())
        except Exception:
            return 0.0
    return 0.0


def _flush_ready_partials_locked():
    ready_ids: list[str] = []
    for image_id, entry in pending.items():
        if _is_merge_ready(entry):
            ready_ids.append(image_id)
        if len(ready_ids) >= 16:
            break
    for image_id in ready_ids:
        entry = pending.get(image_id)
        if entry is None:
            continue
        _build_and_save_merged(image_id, entry)


def _entry_image_path(entry: dict[str, Any]) -> str:
    for role in ("a", "b", "c"):
        rec = entry.get(role)
        if isinstance(rec, dict):
            path = rec.get("_pi", {}).get("saved_image")
            if path:
                return path
    return ""


# ─────────────────────── Startup / Shutdown ───────────────────────
@app.on_event("startup")
def on_startup():
    load_pending_from_disk()
    _load_late_patch_cache()
    print(f"[SERVER] ready | save_dir={SAVE_DIR} | pending={len(pending)}")


@app.on_event("shutdown")
def on_shutdown():
    with pending_lock:
        save_pending_to_disk()
    print(f"[SHUTDOWN] pending cache saved ({len(pending)} entries)")


# ─────────────────────── Endpoints ───────────────────────

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "pi_merge_ingest_v3",
        "ts": now_kst(),
        "stats": stats,
        "pending": len(pending),
    }


@app.get("/health")
def health():
    return {"ok": True, "ts": now_kst()}


@app.get("/status")
def status():
    """Live dashboard for monitoring merge progress."""
    with pending_lock:
        pending_summary = {}
        for img_id, entry in list(pending.items())[:100]:
            roles_have = sorted([r for r in REQUIRED_ROLES if r in entry])
            patches_have = sorted(entry.get("_patches", {}).keys())
            pending_summary[img_id] = {
                "roles": roles_have,
                "patches": patches_have,
            }

    return {
        "ok": True,
        "ts": now_kst(),
        "stats": stats,
        "late_patch_cache": len(late_patch_cache),
        "pending_count": len(pending),
        "pending_detail": pending_summary,
    }


@app.post("/ingest_m5")
async def ingest_m5(
    request: Request,
    meta: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Receive one device's result for one image (multipart: meta JSON + image JPEG).
    """
    t0 = time.perf_counter()

    # ── parse meta JSON ──
    try:
        meta_obj = json.loads(meta)
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": f"meta_json_parse_fail: {e}"}, status_code=400
        )

    if not isinstance(meta_obj, dict):
        return JSONResponse(
            {"ok": False, "error": "meta_must_be_object"}, status_code=400
        )

    run_id   = str(meta_obj.get("run_id", "unknown"))
    device   = str(meta_obj.get("device", "unknown"))
    image_id = os.path.basename(str(meta_obj.get("image_id", f"unknown_{int(time.time())}.jpg")))

    if _is_m5a_coordinator_meta(meta_obj):
        img_dst = os.path.join(IMG_DIR, image_id)
        saved_bytes = await save_upload_file(image, img_dst)
        stats["images_saved"] += 1

        content_length = int(request.headers.get("content-length", "0"))
        proc_ms = max(1, int((time.perf_counter() - t0) * 1000))
        meta_obj["_pi"] = {
            "recv_ts": now_kst(),
            "content_length": content_length,
            "saved_image": img_dst,
            "image_bytes": saved_bytes,
            "proc_ms": proc_ms,
        }

        append_jsonl(raw_path(run_id), meta_obj)
        stats["raw_received"] += 1
        stats["last_recv_ts"] = now_kst()

        merged = False
        with pending_lock:
            pending[image_id] = {
                "_run_id": run_id,
                "_ts_first": now_kst(),
                "_first_wall_ts": time.time(),
                "_patches": {},
                "_coordinator": meta_obj,
                "_coord_img_dst": img_dst,
            }
            entry = pending[image_id]
            _apply_cached_patches(entry, image_id)
            if _is_merge_ready(entry):
                merged = _build_and_save_merged(image_id, entry, img_dst)
            else:
                print(
                    f"[RECV-COORD] {image_id} from={device} waiting_for_patch_a "
                    f"pending={len(pending)}"
                )
            _flush_ready_partials_locked()
            if stats["raw_received"] % PENDING_BACKUP_INTERVAL == 0:
                save_pending_to_disk()

        return JSONResponse(
            {
                "ok": True,
                "ts": now_kst(),
                "run_id": run_id,
                "image_id": image_id,
                "device": device,
                "coordinator": True,
                "merged": merged,
                "proc_ms": proc_ms,
            },
            status_code=200,
        )

    role     = str(meta_obj.get("role", "unknown")).lower()
    role = _role_from_device(role)

    if role not in REQUIRED_ROLES:
        return JSONResponse(
            {"ok": False, "error": f"invalid role: {role}"}, status_code=400
        )

    if REQUIRE_NONEMPTY_M5_TEXT:
        status = str(meta_obj.get("status", "")).strip().lower()
        if status != "ok":
            return JSONResponse(
                {
                    "ok": False,
                    "error": "m5_status_not_ok",
                    "image_id": image_id,
                    "device": device,
                    "role": role,
                    "status": status,
                    "fail_reason": meta_obj.get("fail_reason"),
                },
                status_code=422,
            )
        # A(representative)는 text 필수, B/C(auxiliary)는 cat/sev/conf 필수 (text는 CoT용, optional)
        if role == "a" and not _meta_has_valid_text(meta_obj):
            cat = str(meta_obj.get("cat", "other_disaster") or "other_disaster")
            sev = _safe_int(meta_obj.get("sev"))
            if sev is None:
                sev = 1
            meta_obj["text"] = (
                f"This image indicates {cat} damage around severity level {sev}. "
                "Responders should apply immediate safety controls and inspect risks."
            )
            meta_obj["_pi_warn"] = "a_text_missing_filled"
        if role in ("b", "c") and meta_obj.get("cat") is None:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "m5_aux_cat_sev_conf_required",
                    "image_id": image_id,
                    "device": device,
                    "role": role,
                },
                status_code=422,
            )

    # ── save image ──
    img_dst = os.path.join(IMG_DIR, image_id)
    saved_bytes = await save_upload_file(image, img_dst)
    stats["images_saved"] += 1

    # ── enrich with Pi-side measurements ──
    content_length = int(request.headers.get("content-length", "0"))

    meta_obj["_pi"] = {
        "recv_ts":        now_kst(),
        "content_length": content_length,
        "saved_image":    img_dst,
        "image_bytes":    saved_bytes,
    }

    # Keep paper schema stable even when device firmware omits these fields.
    sig = meta_obj.get("signal_overhead_bytes")
    if not isinstance(sig, dict):
        sig = {}
    if sig.get("m5_pi_tx_bytes") is None:
        sig["m5_pi_tx_bytes"] = content_length
    if "m5_pi_rx_bytes" not in sig:
        sig["m5_pi_rx_bytes"] = None
    if "m5_pi_http_e2e_ms" not in sig:
        sig["m5_pi_http_e2e_ms"] = None
    meta_obj["signal_overhead_bytes"] = sig

    pwr = meta_obj.get("power")
    if not isinstance(pwr, dict):
        pwr = {}
    if "energy_j" not in pwr:
        pwr["energy_j"] = None
    if "power_w" not in pwr:
        pwr["power_w"] = None
    meta_obj["power"] = pwr

    proc_ms = max(1, int((time.perf_counter() - t0) * 1000))
    meta_obj["_pi"]["proc_ms"] = proc_ms

    # ── raw save (append-only, never lost) ──
    append_jsonl(raw_path(run_id), meta_obj)

    stats["raw_received"] += 1
    stats["last_recv_ts"] = now_kst()

    # ── merge logic ──
    merged = False
    with pending_lock:
        if image_id not in pending:
            pending[image_id] = {
                "_run_id":   run_id,
                "_ts_first": now_kst(),
                "_first_wall_ts": time.time(),
                "_patches":  {},
            }

        pending[image_id][role] = meta_obj

        entry = pending[image_id]
        _apply_cached_patches(entry, image_id)
        have  = {r for r in REQUIRED_ROLES if r in entry}

        if _is_merge_ready(entry):
            merged = _build_and_save_merged(image_id, entry, img_dst)
        else:
            missing_roles = sorted(REQUIRED_ROLES - have)
            patch_a = "a" in entry.get("_patches", {})
            print(
                f"[RECV] {device} {image_id} role={role}  "
                f"have={sorted(have)} miss={missing_roles} patch_a={patch_a} "
                f"pending={len(pending)}"
            )

        _flush_ready_partials_locked()

        # periodic disk backup
        if stats["raw_received"] % PENDING_BACKUP_INTERVAL == 0:
            save_pending_to_disk()

    # ── response ──
    resp = {
        "ok": True,
        "ts": now_kst(),
        "run_id": run_id,
        "image_id": image_id,
        "device": device,
        "merged": merged,
        "proc_ms": proc_ms,
    }
    return JSONResponse(resp, status_code=200)


def _build_and_save_coordinator_merged(image_id: str, entry: dict, img_dst: Optional[str] = None) -> bool:
    """Save one complete M5_A-coordinated Stage-1 record.

    This is the online implementation schema consumed by Pi_A/Pi_B workers.
    It is produced after M5_A's multipart upload and its PATCH measurement arrive.
    """
    coord = entry.get("_coordinator") if isinstance(entry.get("_coordinator"), dict) else None
    if coord is None:
        return False

    patches = entry.get("_patches", {})
    a_patch = patches.get("a", {})
    run_id = entry.get("_run_id", coord.get("run_id", "unknown"))
    pi_meta = coord.get("_pi") if isinstance(coord.get("_pi"), dict) else {}

    tx_bytes = int(a_patch.get("m5_pi_tx_bytes", 0) or pi_meta.get("content_length", 0) or 0)
    rx_bytes = int(a_patch.get("m5_pi_rx_bytes", 0) or 0)
    http_ms = int(a_patch.get("m5_pi_http_e2e_ms", 0) or 0)

    merged_record = dict(coord)
    merged_record["type"] = "m5a_stage1_coordinator"
    merged_record["status"] = "complete"
    merged_record["ts_merged"] = now_kst()
    merged_record["ts_first"] = entry.get("_ts_first")
    merged_record["saved_image"] = img_dst or entry.get("_coord_img_dst") or pi_meta.get("saved_image") or ""
    merged_record["image_bytes"] = pi_meta.get("image_bytes", 0)
    merged_record["_pi"] = pi_meta
    merged_record["_patch_status"] = {"a_patched": "a" in patches}

    # Keep both names so the same merged file can be consumed on Pi_A and Pi_B.
    merged_record["go_pi"] = {
        "m5_pi_http_e2e_ms": http_ms,
        "m5_pi_tx_bytes": tx_bytes,
        "m5_pi_rx_bytes": rx_bytes,
    }
    merged_record["go_pi_b"] = {
        "m5_pi_b_http_e2e_ms": http_ms,
        "m5_pi_b_tx_bytes": tx_bytes,
        "m5_pi_b_rx_bytes": rx_bytes,
    }

    append_jsonl(merged_path(run_id), merged_record)
    del pending[image_id]

    stats["merged"] += 1
    stats["last_merge_ts"] = now_kst()
    print(f"[MERGED-COORD] {image_id} -> {merged_path(run_id)}")
    return True


def _build_and_save_merged(image_id: str, entry: dict, img_dst: Optional[str] = None) -> bool:
    """Build paper-aligned merged record and save. Called under pending_lock."""
    if "_coordinator" in entry:
        return _build_and_save_coordinator_merged(image_id, entry, img_dst)

    _apply_cached_patches(entry, image_id)
    a = entry.get("a") if isinstance(entry.get("a"), dict) else None
    b = entry.get("b") if isinstance(entry.get("b"), dict) else None
    c = entry.get("c") if isinstance(entry.get("c"), dict) else None
    patches = entry.get("_patches", {})
    run_id = entry.get("_run_id", "unknown")
    have_roles = {r for r in REQUIRED_ROLES if isinstance(entry.get(r), dict)}
    if not have_roles:
        return False
    is_complete = (have_roles == REQUIRED_ROLES)

    # Extract cat/sev from A payload first, fallback to keyword matching
    a_parsed = _extract_a_output(a) if a else {"text": None, "cat": None, "sev": None, "conf": None}

    # M5_A latency fields (paper §2-1)
    a_lat = a.get("latency", {}) if a else {}
    a_patch = patches.get("a", {})

    # M5_B / M5_C latency (aux, for reference)
    b_lat = b.get("latency", {}) if b else {}
    c_lat = c.get("latency", {}) if c else {}

    # Signal overhead: M5_A POST Content-Length = m5_pi_tx_bytes
    # m5_pi_rx_bytes comes from PATCH (M5-measured response size)
    a_tx = a.get("_pi", {}).get("content_length", 0) if a else 0
    a_rx = a_patch.get("m5_pi_rx_bytes", 0)

    m5_total_ms = _safe_int(a_lat.get("m5_total_ms")) or 0
    m5_sd_read_ms = _safe_int(a_lat.get("m5_sd_read_ms", a_lat.get("sd_read_ms"))) or 0
    m5_decode_resize_ms = _safe_int(a_lat.get("m5_decode_resize_ms", a_lat.get("decode_resize_ms"))) or 0
    m5_uart_ms = _safe_int(a_lat.get("m5_uart_ms", a_lat.get("uart_ms", a.get("uart_push_ms") if a else None))) or 0
    m5_vlm_infer_ms = _safe_int(a_lat.get("m5_vlm_infer_ms", a_lat.get("vlm_infer_ms"))) or 0
    m5_postprocess_ms = _safe_int(a_lat.get("m5_postprocess_ms", a_lat.get("postprocess_ms"))) or 0
    m5_recover_ms = _safe_int(a_lat.get("m5_recover_ms", a_lat.get("recover_ms"))) or 0
    m5_vlm_roundtrip_ms = _safe_int(a_lat.get("m5_vlm_roundtrip_ms", a_lat.get("vlm_roundtrip_ms"))) or 0
    m5_cls_infer_ms = _safe_int(a_lat.get("m5_cls_infer_ms", a_lat.get("cls_infer_ms"))) or 0
    m5_pi_http_e2e_ms = _safe_int(a_patch.get("m5_pi_http_e2e_ms")) or 0

    merged_record = {
        "type":      "m5_oracle",
        "run_id":    run_id,
        "image_id":  image_id,
        "ts_merged": now_kst(),
        "ts_first":  entry.get("_ts_first"),
        "status":    "complete" if is_complete else f"partial_have_{''.join(sorted(have_roles))}",

        # ── END@M5: stop at M5 (paper §D) ──
        "end_m5": {
            "rep_output": "M5_A",
            "output": {
                "text": a_parsed["text"],
                "cat":  a_parsed["cat"],
                "sev":  a_parsed["sev"],
                "conf": a_parsed["conf"],
            },
            "cost": {
                "latency_ms": {
                    "total":          m5_total_ms,
                    "m5_total_ms":    m5_total_ms,
                    "m5_sd_read_ms":  m5_sd_read_ms,
                    "m5_decode_resize_ms": m5_decode_resize_ms,
                    "m5_uart_ms": m5_uart_ms,
                    "m5_vlm_infer_ms": m5_vlm_infer_ms,
                    "m5_postprocess_ms": m5_postprocess_ms,
                    "m5_recover_ms": m5_recover_ms,
                    "m5_vlm_roundtrip_ms": m5_vlm_roundtrip_ms,
                    "m5_cls_infer_ms": m5_cls_infer_ms,
                },
                "signal_overhead_bytes": {
                    "total": 0,
                },
                "energy_j": None,
            },
        },

        # ── go_pi: M5→Pi forwarding cost (for Pi pipeline to use) ──
        "go_pi": {
            "m5_pi_http_e2e_ms": m5_pi_http_e2e_ms,
            "m5_pi_tx_bytes":    a_tx,
            "m5_pi_rx_bytes":    a_rx,
        },

        # ── Cross-tier aggregate latency placeholders ──
        "final_aggregate_ms": {
            "m5_total_ms": m5_total_ms,
            "pi_total_ms": None,
            "jetson_total_ms": None,
            "m5_pi_http_e2e_ms": m5_pi_http_e2e_ms,
            "pi_jetson_http_e2e_ms": None,
        },

        # ── All agent outputs (for agreement logic & analysis) ──
        "artifacts": {
            "m5_a": {
                "text": a_parsed["text"],
                "cat":  a_parsed["cat"],
                "sev":  a_parsed["sev"],
                "conf": _safe_conf_1_5(a.get("conf")) if a else None,
            },
            "m5_b": {
                "text": b.get("text") if b else None,
                "cat":  b.get("cat") if b else None,
                "sev":  _safe_sev_0_2(b.get("sev")) if b else None,
                "conf": _safe_conf_1_5(b.get("conf")) if b else None,
            },
            "m5_c": {
                "text": c.get("text") if c else None,
                "cat":  c.get("cat") if c else None,
                "sev":  _safe_sev_0_2(c.get("sev")) if c else None,
                "conf": _safe_conf_1_5(c.get("conf")) if c else None,
            },
        },

        # ── Aux device latency (for reference) ──
        "aux_latency_ms": {
            "m5_b_total_ms": b_lat.get("m5_total_ms") if b else None,
            "m5_c_total_ms": c_lat.get("m5_total_ms") if c else None,
        },

        # ── Device timestamps ──
        "device_ts": {
            "a": a.get("ts", "") if a else "",
            "b": b.get("ts", "") if b else "",
            "c": c.get("ts", "") if c else "",
        },

        # ── Image ──
        "saved_image":  img_dst if img_dst else _entry_image_path(entry),
        "image_bytes":  a.get("_pi", {}).get("image_bytes", 0) if a else 0,

        # ── PATCH status ──
        "_patch_status": {
            "a_patched": "a" in patches,
            "b_patched": "b" in patches,
            "c_patched": "c" in patches,
        },
    }

    append_jsonl(merged_path(run_id), merged_record)
    del pending[image_id]

    stats["merged"] += 1
    stats["last_merge_ts"] = now_kst()

    patch_flag = "✓" if "a" in patches else "✗"
    arts = merged_record.get("artifacts", {})
    a_cat = arts.get("m5_a", {}).get("cat", "?")
    b_cat = arts.get("m5_b", {}).get("cat", "?")
    c_cat = arts.get("m5_c", {}).get("cat", "?")
    print(
        f"[MERGED] {image_id} status={merged_record['status']} patch_a={patch_flag}  "
        f"cat=A:{a_cat}/B:{b_cat}/C:{c_cat}  "
        f"m5_total={m5_total_ms}ms  "
        f"(total={stats['merged']}  pending={len(pending)})"
    )
    return True


# ─────────── Patch endpoint (M5 sends http_e2e after main POST) ───────────

class PatchBody(BaseModel):
    run_id: str
    device: str
    image_id: str
    m5_pi_http_e2e_ms: int
    m5_pi_tx_bytes: Optional[int] = None
    m5_pi_rx_bytes: Optional[int] = None
    ts: Optional[str] = None


@app.post("/patch_e2e")
async def patch_e2e(body: PatchBody):
    """Record M5-measured HTTP round-trip time and signal overhead.

    If the image is still in pending cache, store patch data there
    so it gets merged into the final record. Otherwise save to patch file.
    """
    image_id = os.path.basename(str(body.image_id))
    run_id = str(body.run_id)
    role = _role_from_device(body.device)
    patch_obj = body.model_dump()
    patch_obj["_pi_recv_ts"] = now_kst()

    if role not in REQUIRED_ROLES:
        role = str(body.device)[-1:].lower()

    integrated = False
    merged = False
    with pending_lock:
        if image_id in pending:
            if "_patches" not in pending[image_id]:
                pending[image_id]["_patches"] = {}
            pending[image_id]["_patches"][role] = {
                "m5_pi_http_e2e_ms": body.m5_pi_http_e2e_ms,
                "m5_pi_tx_bytes":    body.m5_pi_tx_bytes or 0,
                "m5_pi_rx_bytes":    body.m5_pi_rx_bytes or 0,
            }
            integrated = True
            stats["patches_into_pending"] += 1
            if _is_merge_ready(pending[image_id]):
                merged = _build_and_save_merged(
                    image_id,
                    pending[image_id],
                )
            _flush_ready_partials_locked()
        else:
            stats["patches_late"] += 1
            _cache_late_patch(
                run_id,
                image_id,
                role,
                {
                    "m5_pi_http_e2e_ms": body.m5_pi_http_e2e_ms,
                    "m5_pi_tx_bytes":    body.m5_pi_tx_bytes or 0,
                    "m5_pi_rx_bytes":    body.m5_pi_rx_bytes or 0,
                },
            )

    # Always save to patch file (for late patches or backup)
    patch_obj["status"] = "integrated" if integrated else "late"
    patch_obj["role"] = role
    append_jsonl(patch_path(run_id), patch_obj)

    stats["patches"] += 1

    return JSONResponse(
        {
            "ok": True,
            "ts": now_kst(),
            "run_id": run_id,
            "image_id": image_id,
            "device": body.device,
            "integrated": integrated,
            "merged_now": merged,
        },
        status_code=200,
    )


# ─────────────────────── Main ───────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
