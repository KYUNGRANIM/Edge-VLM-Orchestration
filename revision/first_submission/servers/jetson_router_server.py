#!/usr/bin/env python3
"""Online Jetson inference server for the Stage-2 escalation path.

Pi_A posts image + metadata to /ingest_pi_a only when Stage 2 escalates. This
server runs the Jetson VLM immediately, writes one JSONL row, and returns the
Jetson output to Pi_A. This routing copy imports jetson_infer_routing.py,
which keeps the original Jetson temperature and adds an optional seed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (
    SCRIPT_DIR,
    SCRIPT_DIR / "tier3_jetson",
    SCRIPT_DIR.parent / "tier3_jetson",
):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import jetson_inference_engine as jet
except ModuleNotFoundError as e:
    raise SystemExit(
        "Cannot import jetson_inference_engine.py. Put jetson_router_server.py in the same "
        "directory as jetson_infer_routing.py, or copy the tier3_jetson/ directory to this "
        "machine and run from there."
    ) from e

SAVE_DIR = Path(os.environ.get("JETSON_SAVE_DIR", "./tier3_online"))
IMG_DIR = SAVE_DIR / "images"
OUT_PATH = Path(os.environ.get("JETSON_ONLINE_OUT", str(SAVE_DIR / "online_jetson_results.jsonl")))
MAX_IMAGE_DIM = int(os.environ.get("JETSON_MAX_IMAGE_DIM", str(getattr(jet, "DEFAULT_MAX_IMAGE_DIM", 768))))

SAVE_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Online Jetson VLM Server")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def save_upload(upload: UploadFile, dst: Path) -> int:
    data = await upload.read()
    dst.write_bytes(data)
    return len(data)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "save_dir": str(SAVE_DIR),
        "out": str(OUT_PATH),
        "vllm_url": jet.VLLM_URL,
        "model": jet.VLLM_MODEL,
        "temperature": jet.VLM_TEMPERATURE,
        "seed": jet.VLM_SEED or None,
        "top_k": None,
        "top_p": None,
    }


@app.post("/ingest_pi_a")
async def ingest_pi_a(meta: str = Form(...), image: UploadFile = File(...)):
    t0 = time.perf_counter()
    try:
        meta_obj = json.loads(meta)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"meta_parse_fail: {e}"}, status_code=400)

    run_id = str(meta_obj.get("run_id", jet.DEFAULT_RUN_ID))
    image_id = os.path.basename(str(meta_obj.get("image_id", f"unk_{int(time.time())}.jpg")))
    img_dst = IMG_DIR / image_id
    rx_bytes = await save_upload(image, img_dst)

    result, err = jet.run_one_image(img_dst, MAX_IMAGE_DIM)
    if result is None:
        rec = {
            "type": "online_jetson_result",
            "run_id": run_id,
            "image_id": image_id,
            "ts": jet.now_kst(),
            "status": "fail",
            "saved_image": str(img_dst),
            "error": f"vlm_fail:{err}",
        }
        append_jsonl(OUT_PATH, rec)
        return JSONResponse({"ok": False, "error": rec["error"], "image_id": image_id}, status_code=500)

    damage = result.get("damage", "")
    counter = result.get("countermeasure", "")
    text = (damage + " " + counter).strip() if damage and counter else (damage or counter)
    conf_final, conf_detail = jet.compute_confidence(result.get("combined_lp"), result.get("had_labels", False))

    jet_ms = int(result.get("total_ms", 0) or 0)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    out = {
        "type": "online_jetson_result",
        "run_id": run_id,
        "image_id": image_id,
        "ts": jet.now_kst(),
        "status": "ok",
        "saved_image": str(img_dst),
        "pi_a_meta": meta_obj,
        "end_jetson": {
            "rep_output": "jetson",
            "output": {
                "text": text,
                "cat": result.get("cat"),
                "sev": result.get("sev"),
                "conf": conf_final,
            },
            "conf_detail": conf_detail,
            "conf_final": conf_final,
            "vlm_ms": jet_ms,
            "repair": {
                "triggered": bool(result.get("repair_reason")),
                "reason": result.get("repair_reason", ""),
            },
            "cost": {
                "latency_ms": {
                    "jetson_total_ms": jet_ms,
                    "jetson_wall_ms": wall_ms,
                },
                "signal_overhead_bytes": {
                    "pi_jet_rx_bytes": rx_bytes,
                },
                "power_mw": result.get("avg_mw"),
                "energy_j": result.get("energy_j"),
            },
        },
    }
    append_jsonl(OUT_PATH, out)

    return JSONResponse({
        "ok": True,
        "status": "ok",
        "run_id": run_id,
        "image_id": image_id,
        "jetson_total_ms": jet_ms,
        "jetson_wall_ms": wall_ms,
        "jetson": out["end_jetson"]["output"],
        "conf_detail": conf_detail,
        "energy_j": result.get("energy_j"),
        "saved_image": str(img_dst),
    })


if __name__ == "__main__":
    port = int(os.environ.get("JETSON_PORT", "8000"))
    print(f"[JETSON-ONLINE] save_dir={SAVE_DIR}")
    print(f"[JETSON-ONLINE] out={OUT_PATH}")
    print(f"[JETSON-ONLINE] vllm={jet.VLLM_URL} model={jet.VLLM_MODEL}")
    print(f"[JETSON-ONLINE] decode temperature={jet.VLM_TEMPERATURE} seed={jet.VLM_SEED or '(backend default)'} no_top_k_top_p=1")
    uvicorn.run(app, host="0.0.0.0", port=port)
