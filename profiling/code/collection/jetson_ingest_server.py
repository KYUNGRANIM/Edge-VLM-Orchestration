#!/usr/bin/env python3
"""
Jetson Collector - Pi_A + Pi_B 수집 서버 (Oracle Dataset)
=========================================================
Pi_A → /ingest_pi_a : multipart (image + Pi_A result + cost chain)
Pi_B → /ingest_pi_b : JSON only (Pi_B result, no image)

Design (pi_server.py 패턴):
  - 즉시 응답 (VLM 없음, latency 오염 없음)
  - Pi_A + Pi_B 모두 도착 → merged record 저장 (시간 무제한 대기)
  - 저장: tier3_collected/jetson_collected__{run_id}.jsonl

Merged record schema:
  {
    "type": "jetson_collected",
    "run_id": ..., "image_id": ..., "ts": ...,
    "status": "complete" | "pi_a_only",
    "pi_a": {text, cat, sev, conf, costs: {m5_total_ms, ...}},
    "pi_b": {text, cat, sev, conf} | null,
    "saved_image": ...,
    "pi_a_rx_bytes": ...,   # Jetson 수신 bytes (image size)
    "_pi_b_status": "merged" | "timeout_no_pi_b"
  }

Usage:
    source jetson_vlm/bin/activate
    python3 jetson_collector.py [--port 8000] [--run_id pilot_300_run1]
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# ─────────────────────── Config ───────────────────────
HOST         = "0.0.0.0"
PORT         = 8000
RUN_ID       = "pilot_300_run1"
SAVE_DIR     = "./tier3_fire" #./tier3_collected_pi_a_v17"
IMG_DIR      = os.path.join(SAVE_DIR, "images")
KST          = timezone(timedelta(hours=9))
STREAM_CHUNK = 1024 * 1024   # 1 MB

PENDING_TTL  = 86400.0       # stale pending 정리 (24시간). Pi_A/B 둘 다 올 때까지 대기

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ─────────────────────── App + state ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_monitor_pending, daemon=True)
    t.start()
    print(f"[COLLECTOR] ready | save_dir={SAVE_DIR} | merge on Pi_A+Pi_B both received")
    yield


app = FastAPI(title="Jetson Collector (Pi_A + Pi_B ingest)", lifespan=lifespan)

# pending[image_id] = {
#   "run_id": str, "img_path": str,
#   "pi_a": dict|None, "pi_b": dict|None,
#   "pi_a_rx_bytes": int,
#   "pi_a_arrival_ts": float|None,
#   "first_ts": float,
#   "saved": bool
# }
pending: dict[str, dict] = {}
pending_lock = threading.Lock()
RUN_ID_DEFAULT = RUN_ID

stats = {
    "pi_a_received": 0,
    "pi_b_received": 0,
    "saved":         0,
    "last_ts":       None,
}


# ─────────────────────── Helpers ───────────────────────
def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="milliseconds")


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collected_path(run_id: str) -> str:
    return os.path.join(SAVE_DIR, f"jetson_collected__{run_id}.jsonl")


async def save_upload(upload: UploadFile, dst: str) -> int:
    await run_in_threadpool(upload.file.seek, 0)
    with open(dst, "wb") as f:
        await run_in_threadpool(shutil.copyfileobj, upload.file, f, STREAM_CHUNK)
    return os.path.getsize(dst)


# ─────────────────────── Merge + Save ───────────────────────
def _do_save(image_id: str, entry: dict, pi_b_status: str) -> None:
    """Write merged collected record. Called with entry snapshot (no lock needed)."""
    pi_b = entry["pi_b"]
    record = {
        "type":          "jetson_collected",
        "run_id":        entry["run_id"],
        "image_id":      image_id,
        "ts":            now_kst(),
        "status":        "complete" if pi_b else "pi_a_only",
        "pi_a":          entry["pi_a"],
        "pi_b":          pi_b,
        "saved_image":   entry["img_path"],
        "pi_a_rx_bytes": entry["pi_a_rx_bytes"],
        "_pi_b_status":  pi_b_status,
    }
    append_jsonl(collected_path(entry["run_id"]), record)
    stats["saved"] += 1
    stats["last_ts"] = now_kst()
    print(f"[SAVED] {image_id}  pi_b={pi_b_status}  total={stats['saved']}")


def _maybe_save(image_id: str, pi_b_status: str = "merged") -> None:
    """Mark saved and write record. Idempotent (saved flag prevents double-write)."""
    with pending_lock:
        entry = pending.get(image_id)
        if entry is None or entry["saved"] or entry["pi_a"] is None:
            return
        entry["saved"] = True
        snap = {
            "run_id":        entry["run_id"],
            "img_path":      entry["img_path"],
            "pi_a":          entry["pi_a"],
            "pi_b":          entry["pi_b"],
            "pi_a_rx_bytes": entry["pi_a_rx_bytes"],
        }

    _do_save(image_id, snap, pi_b_status)

    with pending_lock:
        pending.pop(image_id, None)


# ─────────────────────── Endpoints ───────────────────────
@app.get("/")
def root():
    return {"ok": True, "service": "jetson_collector", "ts": now_kst(), "stats": stats}


@app.get("/health")
def health():
    return {"ok": True, "ts": now_kst()}


@app.get("/status")
def status():
    with pending_lock:
        n_pending    = len(pending)
        pending_list = list(pending.keys())[:50]
    return {
        "ok":             True,
        "ts":             now_kst(),
        "stats":          stats,
        "pending_count":  n_pending,
        "pending_sample": pending_list,
    }


@app.post("/ingest_pi_a")
async def ingest_pi_a(
    meta:  str        = Form(...),
    image: UploadFile = File(...),
):
    """Receive Pi_A result + image. Save image, store data in pending. Return immediately.
    jetson_total_ms=0 so Pi_A's measured e2e is not inflated by any Jetson processing.
    """
    t0 = time.perf_counter()

    try:
        meta_obj = json.loads(meta)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"meta_parse_fail: {e}"}, status_code=400)

    run_id   = str(meta_obj.get("run_id",   RUN_ID_DEFAULT))
    image_id = os.path.basename(str(meta_obj.get("image_id", f"unk_{int(time.time())}.jpg")))

    img_dst     = os.path.join(IMG_DIR, image_id)
    saved_bytes = await save_upload(image, img_dst)
    io_ms       = int((time.perf_counter() - t0) * 1000)

    pi_a_data = {
        "text":        meta_obj.get("pi_a_text"),
        "cat":         meta_obj.get("pi_a_cat"),
        "sev":         meta_obj.get("pi_a_sev"),
        "conf":        meta_obj.get("pi_a_conf"),
        "conf_detail": meta_obj.get("pi_a_conf_detail"),
        # Cost chain: M5 + Pi costs passed through for Jetson END@Jetson computation
        # pi_jet_* 는 Pi_A가 측정 (go_jetson에 저장), 여기선 제외
        "costs": {
            "m5_total_ms":        meta_obj.get("m5_total_ms",        0),
            "m5_pi_http_e2e_ms":  meta_obj.get("m5_pi_http_e2e_ms",  0),
            "m5_pi_tx_bytes":     meta_obj.get("m5_pi_tx_bytes",     0),
            "m5_pi_rx_bytes":     meta_obj.get("m5_pi_rx_bytes",     0),
            "pi_total_ms":        meta_obj.get("pi_total_ms",        0),
        },
    }

    with pending_lock:
        if image_id not in pending:
            pending[image_id] = {
                "run_id":          run_id,
                "img_path":        img_dst,
                "pi_a":            None,
                "pi_b":            None,
                "pi_a_rx_bytes":   0,
                "pi_a_arrival_ts": None,
                "first_ts":        time.time(),
                "saved":           False,
            }
        entry = pending[image_id]
        entry["pi_a"]            = pi_a_data
        entry["img_path"]        = img_dst
        entry["pi_a_rx_bytes"]   = saved_bytes
        entry["pi_a_arrival_ts"] = time.time()
        pi_b_ready = entry["pi_b"] is not None

    stats["pi_a_received"] += 1
    stats["last_ts"] = now_kst()

    print(f"[PI_A] {image_id}  rx={saved_bytes}B  io={io_ms}ms  "
          f"pi_b={'ready' if pi_b_ready else 'waiting'}  "
          f"total={stats['pi_a_received']}")

    if pi_b_ready:
        _maybe_save(image_id, "merged")

    return JSONResponse({
        "ok":              True,
        "ts":              now_kst(),
        "image_id":        image_id,
        "jetson_total_ms": 0,  # no VLM here; Pi_A measures round-trip to this endpoint only
    })


@app.post("/ingest_pi_b")
async def ingest_pi_b(body: dict):
    """Receive Pi_B result JSON (no image). Trigger save if Pi_A already arrived."""
    image_id = os.path.basename(str(body.get("image_id", "")))
    run_id   = str(body.get("run_id", RUN_ID_DEFAULT))

    if not image_id:
        return JSONResponse({"ok": False, "error": "image_id_missing"}, status_code=400)

    pi_b_data = {
        "text":        body.get("pi_b_text"),
        "cat":         body.get("pi_b_cat"),
        "sev":         body.get("pi_b_sev"),
        "conf":        body.get("pi_b_conf"),
        "conf_detail": body.get("pi_b_conf_detail"),
    }

    with pending_lock:
        if image_id not in pending:
            pending[image_id] = {
                "run_id":          run_id,
                "img_path":        "",
                "pi_a":            None,
                "pi_b":            None,
                "pi_a_rx_bytes":   0,
                "pi_a_arrival_ts": None,
                "first_ts":        time.time(),
                "saved":           False,
            }
        pending[image_id]["pi_b"] = pi_b_data
        pi_a_ready = pending[image_id]["pi_a"] is not None

    stats["pi_b_received"] += 1
    stats["last_ts"] = now_kst()

    print(f"[PI_B] {image_id}  cat={pi_b_data['cat']} sev={pi_b_data['sev']} "
          f"conf={pi_b_data['conf']}  pi_a={'ready' if pi_a_ready else 'waiting'}  "
          f"total={stats['pi_b_received']}")

    if pi_a_ready:
        _maybe_save(image_id, "merged")

    return JSONResponse({"ok": True, "ts": now_kst(), "image_id": image_id})


# ─────────────────────── Stale pending cleanup ───────────────────────
def _monitor_pending():
    """Background: stale 정리만. Pi_A+Pi_B 둘 다 올 때까지 무제한 대기."""
    while True:
        time.sleep(60)
        now_t = time.time()
        to_delete: list[str] = []

        with pending_lock:
            n_waiting = 0
            for iid, entry in list(pending.items()):
                age = now_t - entry["first_ts"]
                if entry["saved"] and age > PENDING_TTL:
                    to_delete.append(iid)
                elif not entry["saved"]:
                    n_waiting += 1

        with pending_lock:
            for iid in to_delete:
                pending.pop(iid, None)

        if to_delete:
            print(f"[MONITOR] stale cleaned={len(to_delete)}")
        if n_waiting > 0:
            print(f"[MONITOR] waiting for both Pi_A+Pi_B: {n_waiting} images")


# ─────────────────────── Main ───────────────────────
if __name__ == "__main__":
    print(f"[COLLECTOR] starting on {HOST}:{PORT}  run_id={RUN_ID_DEFAULT}")
    uvicorn.run(app, host=HOST, port=PORT)
