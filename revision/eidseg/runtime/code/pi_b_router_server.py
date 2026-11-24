#!/usr/bin/env python3
"""Pi_B all-in-one online server for hierarchical routing.

This is the final implementation-facing Pi_B process.

Responsibilities:
  1. Receive the same M5_A coordinator uploads at /ingest_m5.
  2. Receive M5_A HTTP/cost patches at /patch_e2e.
  3. Run Pi_B inference for records escalated by M5_A.
  4. Send Pi_B output to Pi_A at /ingest_pi_b_result.

Run on Pi_B:
    python3 pi_b_router_server.py

Environment overrides, if needed:
    PIB_ONLINE_DIR=./tier2_eidseg_runtime_b
    PI_A_URL=http://<PI_A_IP>:8000/ingest_pi_b_result
    ONLINE_PORT=8000
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

DEFAULT_SAVE_DIR = Path(os.environ.get("PIB_ONLINE_DIR", "./tier2_eidseg_runtime_b"))
IMG_DIR = DEFAULT_SAVE_DIR / "images"

import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

import pi_b_inference_engine as pib

app = FastAPI(title="Pi_B Online Routing Server")

MERGED_PATH = DEFAULT_SAVE_DIR / f"merged_oracle__{pib.RUN_ID}.jsonl"
OUT_PATH = DEFAULT_SAVE_DIR / "online_pi_b_results.jsonl"
SENT_CACHE = DEFAULT_SAVE_DIR / "online_pi_b_sent_cache.jsonl"
PI_A_URL = os.environ.get("PI_A_URL", "http://192.168.0.60:8000/ingest_pi_b_result")
HOST = os.environ.get("ONLINE_HOST", "0.0.0.0")
PORT = int(os.environ.get("ONLINE_PORT", "8000"))
POLL_S = float(os.environ.get("ONLINE_POLL_S", "2.0"))

_state = {
    "lock": threading.Lock(),
    "worker_started": False,
    "pending_m5": {},
    "late_patches": {},
}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("status") == "ok" and obj.get("image_id"):
                done.add(obj["image_id"])
    return done


def now_kst() -> str:
    return pib.now_kst()


def raw_path(run_id: str) -> Path:
    return DEFAULT_SAVE_DIR / f"raw_ingest__{run_id}.jsonl"


def patch_path(run_id: str) -> Path:
    return DEFAULT_SAVE_DIR / f"oracle_patch__{run_id}.jsonl"


def role_from_device(device: str) -> str:
    d = str(device or "").strip().lower()
    if d.startswith("m5_"):
        d = d[3:]
    if d in {"a", "b", "c"}:
        return d
    return d[-1:] if d else "a"


async def save_upload_file(upload: UploadFile, dst_path: Path) -> int:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    await run_in_threadpool(upload.file.seek, 0)
    with dst_path.open("wb") as f:
        await run_in_threadpool(shutil.copyfileobj, upload.file, f)
    return dst_path.stat().st_size


def merge_m5_record(image_id: str, entry: dict) -> bool:
    coord = entry.get("coord")
    patch = entry.get("patch")
    if not isinstance(coord, dict) or not isinstance(patch, dict):
        return False

    run_id = str(coord.get("run_id", pib.RUN_ID))
    saved_image = str(entry.get("saved_image", ""))
    pi_meta = coord.get("_pi") if isinstance(coord.get("_pi"), dict) else {}
    tx_bytes = int(patch.get("m5_pi_tx_bytes", 0) or pi_meta.get("content_length", 0) or 0)
    rx_bytes = int(patch.get("m5_pi_rx_bytes", 0) or 0)
    http_ms = int(patch.get("m5_pi_http_e2e_ms", 0) or 0)

    merged = dict(coord)
    merged.update({
        "type": "m5a_stage1_coordinator",
        "status": "complete",
        "ts_merged": now_kst(),
        "saved_image": saved_image,
        "image_bytes": int(entry.get("image_bytes", 0) or 0),
        "go_pi": {
            "m5_pi_http_e2e_ms": http_ms,
            "m5_pi_tx_bytes": tx_bytes,
            "m5_pi_rx_bytes": rx_bytes,
        },
        "go_pi_b": {
            "m5_pi_b_http_e2e_ms": http_ms,
            "m5_pi_b_tx_bytes": tx_bytes,
            "m5_pi_b_rx_bytes": rx_bytes,
        },
        "_patch_status": {"a_patched": True},
    })
    append_jsonl(MERGED_PATH, merged)
    print(f"[M5-MERGED] {image_id} -> {MERGED_PATH}")
    return True


@app.post("/ingest_m5")
async def ingest_m5(request: Request, meta: str = Form(...), image: UploadFile = File(...)) -> dict:
    try:
        meta_obj = json.loads(meta)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"meta_json_parse_fail: {e}")
    if not isinstance(meta_obj, dict):
        raise HTTPException(status_code=400, detail="meta_must_be_object")
    if meta_obj.get("type") != "m5a_stage1_coordinator":
        raise HTTPException(status_code=400, detail="Pi_B online server expects m5a_stage1_coordinator payload")

    run_id = str(meta_obj.get("run_id", pib.RUN_ID))
    image_id = os.path.basename(str(meta_obj.get("image_id", f"unknown_{int(time.time())}.jpg")))
    img_dst = IMG_DIR / image_id
    saved_bytes = await save_upload_file(image, img_dst)
    content_length = int(request.headers.get("content-length", "0") or 0)
    meta_obj["_pi"] = {
        "recv_ts": now_kst(),
        "content_length": content_length,
        "saved_image": str(img_dst),
        "image_bytes": saved_bytes,
    }
    append_jsonl(raw_path(run_id), meta_obj)

    merged = False
    with _state["lock"]:
        entry = _state["pending_m5"].setdefault(image_id, {})
        entry["coord"] = meta_obj
        entry["saved_image"] = str(img_dst)
        entry["image_bytes"] = saved_bytes
        if image_id in _state["late_patches"]:
            entry["patch"] = _state["late_patches"].pop(image_id)
        if "patch" in entry:
            merged = merge_m5_record(image_id, entry)
            if merged:
                _state["pending_m5"].pop(image_id, None)

    print(f"[M5-IN] {image_id} merged={merged}")
    return {"ok": True, "image_id": image_id, "merged": merged}


@app.post("/patch_e2e")
async def patch_e2e(request: Request) -> dict:
    body = await request.json()
    run_id = str(body.get("run_id", pib.RUN_ID))
    image_id = os.path.basename(str(body.get("image_id", "")))
    role = role_from_device(str(body.get("device", "M5_A")))
    patch = {
        "m5_pi_http_e2e_ms": int(body.get("m5_pi_http_e2e_ms", 0) or 0),
        "m5_pi_tx_bytes": int(body.get("m5_pi_tx_bytes", 0) or 0),
        "m5_pi_rx_bytes": int(body.get("m5_pi_rx_bytes", 0) or 0),
        "role": role,
        "_pi_recv_ts": now_kst(),
    }

    merged = False
    with _state["lock"]:
        entry = _state["pending_m5"].get(image_id)
        if entry is not None:
            entry["patch"] = patch
            merged = merge_m5_record(image_id, entry)
            if merged:
                _state["pending_m5"].pop(image_id, None)
        else:
            _state["late_patches"][image_id] = patch

    patch_obj = dict(body)
    patch_obj.update(patch)
    patch_obj["status"] = "integrated" if merged else "pending_or_late"
    append_jsonl(patch_path(run_id), patch_obj)
    return {"ok": True, "image_id": image_id, "merged_now": merged}


@app.get("/online_status")
def online_status() -> dict:
    return {
        "status": "ok",
        "role": "pi_b_online_server",
        "save_dir": str(DEFAULT_SAVE_DIR),
        "merged": str(MERGED_PATH),
        "out": str(OUT_PATH),
        "pi_a_url": PI_A_URL,
        "pending_m5": len(_state["pending_m5"]),
    }


def post_to_pi_a(payload: dict, timeout_s: int = 30) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    t0 = time.perf_counter()
    resp = requests.post(PI_A_URL, json=payload, timeout=timeout_s)
    http_ms = int((time.perf_counter() - t0) * 1000)
    rx_bytes = len(resp.content)
    resp.raise_for_status()
    return {
        "status": "ok",
        "pi_b_pi_a_http_e2e_ms": http_ms,
        "pi_b_pi_a_tx_bytes": len(body),
        "pi_b_pi_a_rx_bytes": rx_bytes,
    }


def build_pi_b_result(rec: dict, image_path: Path) -> dict:
    image_id = rec["image_id"]
    costs = pib.extract_m5_costs(rec, image_id)

    t0 = time.perf_counter()
    img_b64 = pib.b64_image(image_path)
    result, logprobs, vlm_ms, debug_info = pib.run_pi_inference(img_b64)
    pi_b_total_ms = int((time.perf_counter() - t0) * 1000)

    text = result["text"]
    quality = pib.analyze_quality(text)
    semantic_penalty = min(2, int(not quality["passed"]) + int(debug_info.get("chosen") == "backup"))
    conf_final, conf_detail = pib.compute_confidence(
        logprobs,
        result.get("had_labels", False),
        semantic_penalty=semantic_penalty,
    )
    pi_b_output = {
        "text": text,
        "cat": result.get("cat", ""),
        "sev": result.get("sev", 0),
        "conf": conf_final,
    }

    return {
        "type": "pi_b_stage2_result",
        "run_id": rec.get("run_id", pib.RUN_ID),
        "image_id": image_id,
        "ts": pib.now_kst(),
        "status": "ok",
        "pi_b": pi_b_output,
        "pi_b_conf_detail": conf_detail,
        "pi_b_conf_final": conf_final,
        "pi_b_quality_gate": quality,
        "pi_b_debug": debug_info,
        "pi_b_cost": {
            "latency_ms": {
                "pi_b_total_ms": pi_b_total_ms,
                "pi_vlm_ms": vlm_ms,
            },
            "m5_pi_b": costs,
        },
    }


def worker_loop() -> None:
    done_ids = load_done_ids(OUT_PATH)
    sent_ids = load_done_ids(SENT_CACHE)
    print(f"[PI_B-ONLINE] save_dir={DEFAULT_SAVE_DIR}")
    print(f"[PI_B-ONLINE] merged={MERGED_PATH}")
    print(f"[PI_B-ONLINE] out={OUT_PATH}")
    print(f"[PI_B-ONLINE] pi_a_url={PI_A_URL}")
    print(f"[PI_B-ONLINE] decode temperature={pib.TEMPERATURE} seed={pib.SEED or '(backend default)'} no_top_k_top_p=1")

    n_ok = n_fail = n_skip = 0
    while True:
        if not MERGED_PATH.exists():
            time.sleep(POLL_S)
            continue

        progressed = False
        with MERGED_PATH.open(encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("status") != "complete":
                    continue
                image_id = rec.get("image_id", "")
                if not image_id:
                    continue
                if image_id in done_ids or image_id in sent_ids:
                    continue

                stage1 = rec.get("stage1_router") or rec.get("stage1") or {}
                if stage1.get("decision") == "END@M5":
                    sent_ids.add(image_id)
                    append_jsonl(SENT_CACHE, {"status": "ok", "image_id": image_id, "ts": pib.now_kst(), "skipped": "END@M5"})
                    n_skip += 1
                    continue

                seq = n_ok + n_fail + 1
                print(f"[PI_B {seq}] {image_id}")
                img_path = pib.resolve_image(rec, MERGED_PATH.parent, pib.IMAGE_ROOT)
                if img_path is None:
                    err = {
                        "type": "pi_b_stage2_result",
                        "image_id": image_id,
                        "ts": pib.now_kst(),
                        "status": "fail",
                        "error": "image_not_found",
                    }
                    append_jsonl(OUT_PATH, err)
                    n_fail += 1
                    progressed = True
                    continue

                try:
                    payload = build_pi_b_result(rec, img_path)
                    payload["pi_b_to_pi_a"] = {
                        "status": "sending",
                        "pi_b_pi_a_http_e2e_ms": 0,
                        "pi_b_pi_a_tx_bytes": len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
                        "pi_b_pi_a_rx_bytes": 0,
                    }
                    post_info = post_to_pi_a(payload)
                    payload["pi_b_to_pi_a"] = post_info
                    append_jsonl(OUT_PATH, payload)
                    append_jsonl(SENT_CACHE, {"status": "ok", "image_id": image_id, "ts": pib.now_kst()})
                    done_ids.add(image_id)
                    sent_ids.add(image_id)
                    n_ok += 1
                    progressed = True
                    print(f"         sent -> Pi_A | cat={payload['pi_b']['cat']} sev={payload['pi_b']['sev']} conf={payload['pi_b']['conf']}")
                except Exception as e:
                    append_jsonl(
                        OUT_PATH,
                        {
                            "type": "pi_b_stage2_result",
                            "image_id": image_id,
                            "ts": pib.now_kst(),
                            "status": "fail",
                            "error": str(e),
                        },
                    )
                    n_fail += 1
                    progressed = True
                    print(f"         FAIL: {e}")

                if pib.RELOAD_EVERY and n_ok > 0 and n_ok % pib.RELOAD_EVERY == 0:
                    pib.reload_model()

        if not progressed:
            time.sleep(POLL_S)


@app.on_event("startup")
def start_online_worker() -> None:
    with _state["lock"]:
        if _state["worker_started"]:
            return
        _state["worker_started"] = True
    threading.Thread(target=worker_loop, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
