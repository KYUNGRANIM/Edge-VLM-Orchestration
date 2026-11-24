#!/usr/bin/env python3
"""Pi_A all-in-one online server for hierarchical routing.

This is the final implementation-facing Pi_A process.

Responsibilities:
  1. Receive M5_A coordinator uploads at /ingest_m5.
  2. Receive M5_A HTTP/cost patches at /patch_e2e.
  3. Receive END@M5 final rows at /ingest_final_route.
  4. Run Pi_A inference for records escalated by M5_A.
  5. Receive Pi_B results at /ingest_pi_b_result.
  6. Apply the exported Stage-2 router.
  7. Either write END@Pi final rows or forward to Jetson.

Run on Pi_A:
    python pi_a_online_server.py

Environment overrides, if needed:
    PIA_ONLINE_DIR=./tier2_online_a
    JETSON_URL=http://<JETSON_IP>:8000/ingest_pi_a
    ONLINE_PORT=8000
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_SAVE_DIR = Path(os.environ.get("PIA_ONLINE_DIR", "./tier2_online_a"))
IMG_DIR = DEFAULT_SAVE_DIR / "images"

import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling" / "code" / "collection"))
import pi_a_inference_pipeline as pia

app = FastAPI(title="Pi_A Online Routing Server")

MERGED_PATH = DEFAULT_SAVE_DIR / f"merged_oracle__{pia.RUN_ID}.jsonl"
FINAL_OUT = DEFAULT_SAVE_DIR / "online_routing_final.jsonl"
PI_B_CACHE = DEFAULT_SAVE_DIR / "online_pi_b_received_cache.jsonl"
JETSON_URL = os.environ.get("JETSON_URL", pia.JETSON_URL)
HOST = os.environ.get("ONLINE_HOST", "0.0.0.0")
PORT = int(os.environ.get("ONLINE_PORT", "8000"))
POLL_S = float(os.environ.get("ONLINE_POLL_S", "2.0"))
PI_B_TIMEOUT_S = int(os.environ.get("PI_B_TIMEOUT_S", "900"))

_state: dict[str, Any] = {
    "lock": threading.Lock(),
    "pi_b_results": {},
    "final_ids": set(),
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


def load_pi_b_cache(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            image_id = obj.get("image_id")
            if image_id and obj.get("status") == "ok":
                rows[image_id] = obj
    return rows


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def majority_fraction(vals: list[Any]) -> float:
    if not vals:
        return 0.0
    return Counter(vals).most_common(1)[0][1] / len(vals)


def stage1_features(rec: dict[str, Any]) -> dict[str, Any]:
    a, b, c = rec["m5_a"], rec["m5_b"], rec["m5_c"]
    sevs = [safe_int(a.get("sev")), safe_int(b.get("sev")), safe_int(c.get("sev"))]
    confs = [safe_int(a.get("conf")), safe_int(b.get("conf")), safe_int(c.get("conf"))]
    cats = [a.get("cat", ""), b.get("cat", ""), c.get("cat", "")]
    return {
        "m5a_cat": a.get("cat", ""),
        "m5a_sev": sevs[0],
        "m5_conf_mean": sum(confs) / len(confs),
        "m5_conf_min": min(confs),
        "m5_sev_range": max(sevs) - min(sevs),
        "m5_cat_agree": majority_fraction(cats),
        "m5_sev_agree": majority_fraction(sevs),
    }


def stage2_features(
    rec: dict[str, Any],
    pi_a_output: dict[str, Any],
    pi_a_conf_detail: dict[str, Any] | None,
    pi_b_output: dict[str, Any],
    pi_b_conf_detail: dict[str, Any] | None,
) -> dict[str, Any]:
    out = stage1_features(rec)
    pa_sev = safe_int(pi_a_output.get("sev"))
    pb_sev = safe_int(pi_b_output.get("sev"))
    pa_conf = safe_int(pi_a_output.get("conf"))
    pb_conf = safe_int(pi_b_output.get("conf"))
    out.update({
        "pia_cat": pi_a_output.get("cat", ""),
        "pia_sev": pa_sev,
        "pi_conf_mean": (pa_conf + pb_conf) / 2.0,
        "pi_conf_min": min(pa_conf, pb_conf),
        "pi_cat_match": int(pi_a_output.get("cat") == pi_b_output.get("cat")),
        "pi_sev_match": int(pa_sev == pb_sev),
        "pia_min_logprob": safe_float((pi_a_conf_detail or {}).get("min_logprob"), -5.0),
        "m5a_pia_cat_match": int(rec["m5_a"].get("cat") == pi_a_output.get("cat")),
        "m5a_pia_sev_diff": abs(safe_int(rec["m5_a"].get("sev")) - pa_sev),
    })
    return out


class EmbeddedStage2Router:
    """Hard-coded exported Stage-2 decision tree.

    This removes the runtime dependency on sklearn/joblib/router_policy_ab.py.
    The thresholds below are the raw-value equivalents of
    revision/first_submission/routers/stage2_pi_router.txt.
    Return True for END@Pi and False for ESCALATE@Jetson.
    """

    def end_at_pi(
        self,
        rec: dict[str, Any],
        pi_a_output: dict[str, Any],
        pi_a_conf_detail: dict[str, Any] | None,
        pi_b_output: dict[str, Any],
        pi_b_conf_detail: dict[str, Any] | None,
    ) -> tuple[bool, dict[str, Any]]:
        f = stage2_features(rec, pi_a_output, pi_a_conf_detail, pi_b_output, pi_b_conf_detail)

        if f["pia_sev"] <= 1.5:
            if f["pia_cat"] != "landslide":
                pred = 0
            else:
                if f["pi_cat_match"] <= 0.5:
                    pred = 1 if f["m5_cat_agree"] > 0.8333409031498956 else 0
                else:
                    pred = 1 if f["pi_conf_mean"] <= 2.7500429009853637 else 0
        else:
            if f["pi_cat_match"] <= 0.5:
                if f["pia_cat"] != "hurricane":
                    pred = 0
                else:
                    pred = 1 if f["pi_sev_match"] > 0.49997807817021 else 0
            else:
                if f["pia_cat"] != "other_disaster":
                    pred = 1
                else:
                    pred = 1 if f["m5_conf_min"] > 1.5000452468986498 else 0

        return pred == 1, f


def now_kst() -> str:
    return pia.now_kst()


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

    run_id = str(coord.get("run_id", pia.RUN_ID))
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
        raise HTTPException(status_code=400, detail="Pi_A online server expects m5a_stage1_coordinator payload")

    run_id = str(meta_obj.get("run_id", pia.RUN_ID))
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
    run_id = str(body.get("run_id", pia.RUN_ID))
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
    with _state["lock"]:
        return {
            "status": "ok",
            "role": "pi_a_online_server",
            "save_dir": str(DEFAULT_SAVE_DIR),
            "merged": str(MERGED_PATH),
            "final_out": str(FINAL_OUT),
            "pi_b_cached": len(_state["pi_b_results"]),
            "final_records": len(_state["final_ids"]),
            "pending_m5": len(_state["pending_m5"]),
        }


@app.post("/ingest_pi_b_result")
async def ingest_pi_b_result(request: Request) -> dict:
    payload = await request.json()
    image_id = payload.get("image_id")
    if not image_id:
        raise HTTPException(status_code=400, detail="missing image_id")
    if payload.get("status") != "ok":
        raise HTTPException(status_code=400, detail="pi_b payload status is not ok")
    with _state["lock"]:
        _state["pi_b_results"][image_id] = payload
        append_jsonl(PI_B_CACHE, payload)
    print(f"[PI_B-IN] {image_id}")
    return {"status": "ok", "image_id": image_id}


@app.post("/ingest_final_route")
async def ingest_final_route(request: Request) -> dict:
    payload = await request.json()
    image_id = payload.get("image_id")
    if not image_id:
        raise HTTPException(status_code=400, detail="missing image_id")
    if payload.get("status") != "ok":
        raise HTTPException(status_code=400, detail="final route status is not ok")

    payload["type"] = "online_routing_final"
    payload.setdefault("received_ts", pia.now_kst())
    with _state["lock"]:
        if image_id in _state["final_ids"]:
            return {"status": "ok", "image_id": image_id, "duplicate": True}
        append_jsonl(FINAL_OUT, payload)
        _state["final_ids"].add(image_id)
    print(f"[FINAL-M5] {image_id} final_tier={payload.get('final_tier') or payload.get('route_label')}")
    return {"status": "ok", "image_id": image_id, "duplicate": False}


def get_pi_b_result(image_id: str) -> dict | None:
    with _state["lock"]:
        return _state["pi_b_results"].get(image_id)


def wait_pi_b_result(image_id: str) -> dict | None:
    deadline = time.time() + PI_B_TIMEOUT_S
    while time.time() < deadline:
        got = get_pi_b_result(image_id)
        if got is not None:
            return got
        time.sleep(POLL_S)
    return None


def m5_costs_for_pi_a(rec: dict, image_id: str) -> dict:
    try:
        return pia.extract_m5_costs(rec, {}, image_id)
    except TypeError:
        return pia.extract_m5_costs(rec, image_id)


def pi_b_m5_costs(pi_b_result: dict) -> dict:
    return ((pi_b_result.get("pi_b_cost") or {}).get("m5_pi_b") or {})


def build_pi_a_result(image_path: Path) -> tuple[dict, dict, dict, int, int, dict]:
    t0 = time.perf_counter()
    img_b64 = pia.b64_image(image_path)
    result, logprobs, vlm_ms, debug_info = pia.run_pi_inference(img_b64)
    pi_a_total_ms = int((time.perf_counter() - t0) * 1000)

    situation = result.get("situation", "")
    action = result.get("action", "")
    quality = pia.analyze_quality(situation, action)
    semantic_penalty = min(2, int(not quality["passed"]) + int(debug_info.get("chosen") == "backup"))
    conf_final, conf_detail = pia.compute_confidence(
        logprobs,
        result.get("had_labels", False),
        semantic_penalty=semantic_penalty,
    )
    pi_a_output = {
        "text": result.get("text", ""),
        "cat": result.get("cat", ""),
        "sev": result.get("sev", 0),
        "conf": conf_final,
    }
    return pi_a_output, conf_detail, quality, pi_a_total_ms, vlm_ms, debug_info


def send_jetson(meta: dict, image_path: Path) -> dict:
    if not JETSON_URL:
        return {
            "status": "skipped_no_url",
            "pi_jet_http_e2e_ms": 0,
            "pi_jet_tx_bytes": 0,
            "pi_jet_rx_bytes": 0,
            "jetson_total_ms": 0,
        }
    try:
        meta_json = json.dumps(meta, ensure_ascii=False)
        session = requests.Session()
        with image_path.open("rb") as img_f:
            req = requests.Request(
                "POST",
                JETSON_URL,
                data={"meta": meta_json},
                files={"image": (image_path.name, img_f, "image/jpeg")},
            )
            prepared = session.prepare_request(req)
            tx_bytes = int(prepared.headers.get("Content-Length", 0))
            t0 = time.perf_counter()
            resp = session.send(prepared, timeout=pia.JETSON_TIMEOUT)
        http_ms = int((time.perf_counter() - t0) * 1000)
        rx_bytes = len(resp.content)
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            data = {}
        return {
            "status": "ok",
            "pi_jet_http_e2e_ms": http_ms,
            "pi_jet_tx_bytes": tx_bytes,
            "pi_jet_rx_bytes": rx_bytes,
            "jetson_total_ms": data.get("jetson_total_ms", 0),
            "jetson_response": data,
        }
    except Exception as e:
        return {
            "status": f"fail: {e}",
            "pi_jet_http_e2e_ms": 0,
            "pi_jet_tx_bytes": 0,
            "pi_jet_rx_bytes": 0,
            "jetson_total_ms": 0,
        }


def process_record(rec: dict, router: EmbeddedStage2Router) -> dict:
    image_id = rec["image_id"]
    stage1 = rec.get("stage1_router") or rec.get("stage1") or {}

    if stage1.get("decision") == "END@M5":
        return {
            "type": "online_routing_final",
            "run_id": rec.get("run_id", pia.RUN_ID),
            "image_id": image_id,
            "ts": pia.now_kst(),
            "status": "ok",
            "final_tier": "m5_a",
            "route_label": "m5_a",
            "stage1": stage1,
            "stage2": {"decision": "skipped_by_stage1"},
            "m5": {"m5_a": rec.get("m5_a"), "m5_b": rec.get("m5_b"), "m5_c": rec.get("m5_c")},
            "pi": None,
            "jetson": None,
            "final_output": rec.get("m5_a"),
        }

    image_path = pia.resolve_image(rec, MERGED_PATH.parent, pia.IMAGE_ROOT)
    if image_path is None:
        return {
            "type": "online_routing_final",
            "image_id": image_id,
            "ts": pia.now_kst(),
            "status": "fail",
            "error": "image_not_found",
        }

    stage2_start = time.perf_counter()
    pi_a_output, pi_a_conf_detail, pi_a_quality, pi_a_total_ms, pi_a_vlm_ms, pi_a_debug = build_pi_a_result(image_path)

    pi_b_result = wait_pi_b_result(image_id)
    if pi_b_result is None:
        return {
            "type": "online_routing_final",
            "image_id": image_id,
            "ts": pia.now_kst(),
            "status": "fail",
            "error": "pi_b_timeout",
            "pi_a": pi_a_output,
            "pi_a_conf_detail": pi_a_conf_detail,
        }

    pi_b_output = pi_b_result["pi_b"]
    pi_b_conf_detail = pi_b_result.get("pi_b_conf_detail") or {}
    end_pi, stage2_features = router.end_at_pi(
        rec,
        pi_a_output,
        pi_a_conf_detail,
        pi_b_output,
        pi_b_conf_detail,
    )
    stage2_wall_ms = int((time.perf_counter() - stage2_start) * 1000)

    m5a_costs = m5_costs_for_pi_a(rec, image_id)
    m5b_costs = pi_b_m5_costs(pi_b_result)
    pi_b_to_pi_a = pi_b_result.get("pi_b_to_pi_a") or {}

    m5_total_ms = int(m5a_costs.get("m5_total_ms", 0))
    m5_pi_http_ms = int(m5a_costs.get("m5_pi_http_e2e_ms", 0))
    end_pi_latency = m5_total_ms + m5_pi_http_ms + stage2_wall_ms

    overhead_total = (
        int(m5a_costs.get("m5_pi_tx_bytes", 0)) +
        int(m5a_costs.get("m5_pi_rx_bytes", 0)) +
        int(m5b_costs.get("m5_pi_b_tx_bytes", 0)) +
        int(m5b_costs.get("m5_pi_b_rx_bytes", 0)) +
        int(pi_b_to_pi_a.get("pi_b_pi_a_tx_bytes", 0)) +
        int(pi_b_to_pi_a.get("pi_b_pi_a_rx_bytes", 0))
    )

    route_label = "pi_a" if end_pi else "jetson"
    jetson_meta = {
        "type": "pi_a_stage2_coordinator",
        "run_id": rec.get("run_id", pia.RUN_ID),
        "image_id": image_id,
        "ts": pia.now_kst(),
        "m5_a": rec.get("m5_a"),
        "m5_b": rec.get("m5_b"),
        "m5_c": rec.get("m5_c"),
        "pi_a": pi_a_output,
        "pi_b": pi_b_output,
        "pi_a_conf_detail": pi_a_conf_detail,
        "pi_b_conf_detail": pi_b_conf_detail,
        "m5_total_ms": m5_total_ms,
        "m5_pi_http_e2e_ms": m5_pi_http_ms,
        "m5_pi_tx_bytes": m5a_costs.get("m5_pi_tx_bytes", 0),
        "m5_pi_rx_bytes": m5a_costs.get("m5_pi_rx_bytes", 0),
        "pi_total_ms": stage2_wall_ms,
        "stage1": stage1,
        "stage2": {"decision": "ESCALATE@Jetson", "features": stage2_features},
    }
    jetson_meas = {
        "status": "skipped_by_stage2",
        "pi_jet_http_e2e_ms": 0,
        "pi_jet_tx_bytes": 0,
        "pi_jet_rx_bytes": 0,
        "jetson_total_ms": 0,
    }
    if not end_pi:
        jetson_meas = send_jetson(jetson_meta, image_path)

    jetson_response = jetson_meas.get("jetson_response") if isinstance(jetson_meas, dict) else None
    jetson_output = (jetson_response or {}).get("jetson") if isinstance(jetson_response, dict) else None
    final_output = pi_a_output if end_pi else jetson_output

    return {
        "type": "online_routing_final",
        "run_id": rec.get("run_id", pia.RUN_ID),
        "image_id": image_id,
        "ts": pia.now_kst(),
        "status": "ok",
        "final_tier": route_label,
        "route_label": route_label,
        "route_rep_output": "Pi_A" if end_pi else "Jetson",
        "final_output": final_output,
        "stage1": {"decision": stage1.get("decision", "ESCALATE@Pi"), "source": "m5_a_coordinator", "raw": stage1},
        "stage2": {"decision": "END@Pi" if end_pi else "ESCALATE@Jetson", "features": stage2_features},
        "m5": {"m5_a": rec.get("m5_a"), "m5_b": rec.get("m5_b"), "m5_c": rec.get("m5_c")},
        "pi": {"pi_a": pi_a_output, "pi_b": pi_b_output},
        "jetson": jetson_response if not end_pi else None,
        "m5_a": rec.get("m5_a"),
        "m5_b": rec.get("m5_b"),
        "m5_c": rec.get("m5_c"),
        "pi_a": pi_a_output,
        "pi_b": pi_b_output,
        "pi_a_conf_detail": pi_a_conf_detail,
        "pi_b_conf_detail": pi_b_conf_detail,
        "pi_a_quality_gate": pi_a_quality,
        "pi_b_quality_gate": pi_b_result.get("pi_b_quality_gate"),
        "pi_a_debug": pi_a_debug,
        "pi_b_debug": pi_b_result.get("pi_b_debug"),
        "end_pi": {
            "rep_output": "Pi_A",
            "output": pi_a_output,
            "cost": {
                "latency_ms": {
                    "total": end_pi_latency,
                    "m5_total_ms": m5_total_ms,
                    "m5_pi_http_e2e_ms": m5_pi_http_ms,
                    "stage2_wall_ms": stage2_wall_ms,
                    "pi_a_total_ms": pi_a_total_ms,
                    "pi_a_vlm_ms": pi_a_vlm_ms,
                    "pi_b_total_ms": ((pi_b_result.get("pi_b_cost") or {}).get("latency_ms") or {}).get("pi_b_total_ms", 0),
                },
                "signal_overhead_bytes": {
                    "total": overhead_total,
                    "m5_pi_a_tx_bytes": m5a_costs.get("m5_pi_tx_bytes", 0),
                    "m5_pi_a_rx_bytes": m5a_costs.get("m5_pi_rx_bytes", 0),
                    "m5_pi_b_tx_bytes": m5b_costs.get("m5_pi_b_tx_bytes", 0),
                    "m5_pi_b_rx_bytes": m5b_costs.get("m5_pi_b_rx_bytes", 0),
                    "pi_b_pi_a_tx_bytes": pi_b_to_pi_a.get("pi_b_pi_a_tx_bytes", 0),
                    "pi_b_pi_a_rx_bytes": pi_b_to_pi_a.get("pi_b_pi_a_rx_bytes", 0),
                },
            },
        },
        "go_jetson": jetson_meas,
    }


def worker_loop() -> None:
    router = EmbeddedStage2Router()
    with _state["lock"]:
        _state["pi_b_results"] = load_pi_b_cache(PI_B_CACHE)
        _state["final_ids"] = load_done_ids(FINAL_OUT)

    print(f"[PI_A-ONLINE] save_dir={DEFAULT_SAVE_DIR}")
    print(f"[PI_A-ONLINE] merged={MERGED_PATH}")
    print(f"[PI_A-ONLINE] final_out={FINAL_OUT}")
    print(f"[PI_A-ONLINE] jetson={JETSON_URL or '(disabled)'}")
    print(f"[PI_A-ONLINE] cached Pi_B results={len(_state['pi_b_results'])}")

    n_ok = n_fail = 0
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
                with _state["lock"]:
                    if image_id in _state["final_ids"]:
                        continue

                seq = n_ok + n_fail + 1
                print(f"[PI_A {seq}] {image_id}")
                try:
                    out = process_record(rec, router)
                except Exception as e:
                    out = {
                        "type": "online_routing_final",
                        "image_id": image_id,
                        "ts": pia.now_kst(),
                        "status": "fail",
                        "error": str(e),
                    }

                append_jsonl(FINAL_OUT, out)
                progressed = True
                if out.get("status") == "ok":
                    with _state["lock"]:
                        _state["final_ids"].add(image_id)
                    n_ok += 1
                    print(f"         route={out.get('route_label')} stage2={out.get('stage2', {}).get('decision')}")
                else:
                    n_fail += 1
                    print(f"         FAIL: {out.get('error')}")

                if pia.RELOAD_EVERY and n_ok > 0 and n_ok % pia.RELOAD_EVERY == 0:
                    pia.reload_model()

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
