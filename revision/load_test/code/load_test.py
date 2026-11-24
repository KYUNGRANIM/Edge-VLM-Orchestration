#!/usr/bin/env python3
"""Concurrent-request test: replay the recorded M5 responses into the shared Pi-Jetson path.

    python3 revision/load_test/code/load_test.py --rep 1 [--methods ours,direct_pi,direct_jetson] [--ks 1,2,4] [--dry-run]

Scope: the M5 tier is not run. Each request is the upload M5_A made in the MEDIC rerun (image, M5_A/B/C answers,
Stage-1 decision), re-sent under a new request id; Pi_A, Pi_B and Jetson run unchanged except for the load-test
copies of the servers in this folder (Direct@Pi switch, start timestamps). The measured quantity is the extra
response time after the M5 response is ready.

Methods (fixed before running):
  ours           recorded Stage-1 decision; END@M5 images send nothing; escalated images go to Pi_A and Pi_B,
                 then the deployed Stage-2 router decides on Jetson
  direct_pi      every image runs on Pi_A only and ends there (no Pi_B, no Stage-2, no Jetson)
  direct_jetson  every image goes straight to the Jetson server

Concurrency is closed-loop: at most k requests are unfinished; when one finishes the next is sent. Pi requests
finish when Pi_A's ok-final count rises (Pi_A has no per-request status); Jetson requests finish at the reply.
Per-request timing for Pi requests comes from Pi_A's final file (recv_ts, work_start_ts, ts on Pi_A's clock).
"""
import argparse
import json
import random
import subprocess
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
M5VLM = HERE.parents[2]
RUN = M5VLM / "deployment/live_run"
# Local copy of the 60 chosen images from /Volumes/CORSAIR/MEDIC_v2_test/images_sampled, so a run does not depend on
# the external drive staying connected (the first attempt, RUN_ID r28_load, stopped at 14:59 when it dropped).
IMAGE_ROOT = HERE.parent / "images_60"
PI_A, PI_B, JETSON = "http://192.168.0.60:8000", "http://192.168.0.93:8000", "http://192.168.0.92:8000"
IMAGES = HERE.parent / "images_60.txt"
LOG = HERE.parent / "results" / "load_test_log.jsonl"
RUN_ID = "r28_load2"
STALL_S = 1200  # abort a condition if nothing finishes for 20 min


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def now():
    return datetime.now().isoformat(timespec="milliseconds")


def post(url, meta=None, image=None, body=None, timeout=300):
    """multipart (meta + image) or JSON body; returns (http code, response text)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(meta if meta is not None else body, f, ensure_ascii=False)
    cmd = ["curl", "-s", "--connect-timeout", "10", "-m", str(timeout), "-w", "\n%{http_code}"]
    cmd += ["-F", f"meta=<{f.name}", "-F", f"image=@{image}"] if meta is not None else \
           ["-H", "Content-Type: application/json", "--data-binary", f"@{f.name}"]
    out = subprocess.run(cmd + [url], capture_output=True, text=True).stdout
    Path(f.name).unlink()
    text, _, code = out.rpartition("\n")
    return code, text


def pi_a_finals():
    for _ in range(60):  # ride out a network blip of up to ~10 min
        out = subprocess.run(["curl", "-s", "-m", "8", f"{PI_A}/online_status"], capture_output=True, text=True).stdout
        try:
            return json.loads(out)["final_records"]
        except (json.JSONDecodeError, KeyError):
            time.sleep(3)
    raise RuntimeError("Pi_A /online_status not answering")


def choose_images(run):
    """60 images, stratified by the rerun's route in proportion to the 600 (30 M5 / 16 Pi / 14 Jetson)."""
    if IMAGES.exists():
        return [l.split("\t")[0] for l in IMAGES.read_text().splitlines() if l.strip()]
    rng = random.Random(20261106)
    by = {t: sorted(i for i, r in run.items() if r["route_label"] == t) for t in ("m5_a", "pi_a", "jetson")}
    pick = {t: rng.sample(by[t], round(60 * len(by[t]) / len(run))) for t in by}
    rows = [(i, t) for t in by for i in pick[t]]
    IMAGES.write_text("".join(f"{i}\t{t}\n" for i, t in rows))
    return [i for i, _ in rows]


def requests_for(method, rep, k, ids, run, raw_a, raw_b, patch_a, patch_b, paths):
    """List of (request id, original id, kind) and the payloads to send."""
    out = []
    for orig in ids:
        rid = f"{RUN_ID}_rep{rep}_{method}_k{k}__{orig}"
        if method == "direct_jetson":
            out.append({"rid": rid, "orig": orig, "kind": "jetson",
                        "meta": {"type": "pi_a_stage2_coordinator", "run_id": RUN_ID, "image_id": rid}})
            continue
        if method == "ours" and run[orig]["route_label"] == "m5_a":
            out.append({"rid": rid, "orig": orig, "kind": "m5_end"})
            continue
        r = run[orig]
        base = raw_a.get(orig) or {  # images that ended at M5 in the rerun have no recorded Pi upload
            "type": "m5a_stage1_coordinator", "device": "M5_A", "m5_a": r["m5"]["m5_a"], "m5_b": r["m5"]["m5_b"],
            "m5_c": r["m5"]["m5_c"], "stage1_router": {"decision": "ESCALATE@Pi", "policy": "r28_direct_pi"}}
        meta_a = {k_: v for k_, v in base.items() if k_ not in ("_pi", "saved_image")}
        meta_a.update(run_id=RUN_ID, image_id=rid)
        p = patch_a.get(orig) or {"m5_pi_http_e2e_ms": 0, "m5_pi_tx_bytes": 0, "m5_pi_rx_bytes": 0}
        pa = {"run_id": RUN_ID, "image_id": rid, "device": "M5_A", "m5_pi_http_e2e_ms": p["m5_pi_http_e2e_ms"],
              "m5_pi_tx_bytes": p["m5_pi_tx_bytes"], "m5_pi_rx_bytes": p["m5_pi_rx_bytes"]}
        req = {"rid": rid, "orig": orig, "kind": "pi", "meta_a": meta_a, "patch_a": pa}
        if method == "direct_pi":
            meta_a["policy_override"] = "direct_pi"
        else:
            meta_b = {k_: v for k_, v in raw_b[orig].items() if k_ not in ("_pi", "saved_image")}
            meta_b.update(run_id=RUN_ID, image_id=rid)
            pb = dict(pa, **{k_: patch_b[orig][k_] for k_ in ("m5_pi_http_e2e_ms", "m5_pi_tx_bytes", "m5_pi_rx_bytes")})
            req.update(meta_b=meta_b, patch_b=pb)
        out.append(req)
    for q in out:
        q["image"] = paths[q["orig"]]
    return out


def log(row):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def send_pi(q):
    def send(url, **kw):
        # An upload that never arrives never finishes, and the closed loop would wait on it; on 11-06 the Mac slept
        # (lid closed) and three tries 10 s apart were not enough, so keep trying for up to ~20 min.
        for _ in range(60):
            code = post(url, **kw)[0]
            if code == "200":
                break
            time.sleep(10)
        return code
    codes = [send(f"{PI_A}/ingest_m5", meta=q["meta_a"], image=q["image"])]
    if "meta_b" in q:
        codes.append(send(f"{PI_B}/ingest_m5", meta=q["meta_b"], image=q["image"]))
    codes.append(send(f"{PI_A}/patch_e2e", body=q["patch_a"]))
    if "patch_b" in q:
        codes.append(send(f"{PI_B}/patch_e2e", body=q["patch_b"]))
    return codes


def run_condition(method, rep, k, reqs):
    """Closed loop with at most k unfinished requests."""
    t_start = time.time()
    for q in (q for q in reqs if q["kind"] == "m5_end"):
        log({"rep": rep, "method": method, "k": k, "rid": q["rid"], "orig": q["orig"], "kind": "m5_end"})
    todo = [q for q in reqs if q["kind"] != "m5_end"]
    if not todo:
        return
    if todo[0]["kind"] == "jetson":
        lock, queue = threading.Lock(), list(todo)

        def worker():
            while True:
                with lock:
                    if not queue:
                        return
                    q = queue.pop(0)
                sub = now(); t0 = time.time()
                code, _ = post(f"{JETSON}/ingest_pi_a", meta=q["meta"], image=q["image"], timeout=600)
                log({"rep": rep, "method": method, "k": k, "rid": q["rid"], "orig": q["orig"], "kind": "jetson",
                     "submit": sub, "done": now(), "seconds": round(time.time() - t0, 2), "http": code})
        threads = [threading.Thread(target=worker) for _ in range(k)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    else:
        base = pi_a_finals()
        sent, last_done, last_change = 0, 0, time.time()
        while True:
            done = pi_a_finals() - base
            if done != last_done:
                last_done, last_change = done, time.time()
            if done >= len(todo):
                break
            while sent < len(todo) and sent - done < k:
                q = todo[sent]
                sub = now()
                codes = send_pi(q)
                log({"rep": rep, "method": method, "k": k, "rid": q["rid"], "orig": q["orig"], "kind": "pi",
                     "submit": sub, "http": codes})
                sent += 1
            if time.time() - last_change > STALL_S:
                log({"rep": rep, "method": method, "k": k, "abort": "stall", "sent": sent, "done": done})
                print(f"  ABORT {method} k={k}: nothing finished for {STALL_S // 60} min ({done}/{len(todo)})", flush=True)
                return
            time.sleep(2)
    print(f"  {method} k={k}: {len(todo)} requests in {(time.time() - t_start) / 60:.1f} min", flush=True)


def main():
    global RUN_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--methods", default="ours,direct_pi,direct_jetson")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--dry-run", action="store_true")
    # a fresh request-id prefix for re-running conditions: Pi_A skips ids it has already finished
    ap.add_argument("--run-id", default=RUN_ID)
    a = ap.parse_args()
    RUN_ID = a.run_id
    run = {r["image_id"]: r for r in jl(RUN / "online_routing_final.jsonl")}
    raw_a = {r["image_id"]: r for r in jl(RUN / "pi_a/raw_ingest__medic_rerun600.jsonl")}
    raw_b = {r["image_id"]: r for r in jl(RUN / "pi_b/raw_ingest__medic_rerun600.jsonl")}
    patch_a = {r["image_id"]: r for r in jl(RUN / "pi_a/oracle_patch__medic_rerun600.jsonl")}
    patch_b = {r["image_id"]: r for r in jl(RUN / "pi_b/oracle_patch__medic_rerun600.jsonl")}
    paths = {p.name: p for p in IMAGE_ROOT.rglob("*.jpg") if not p.name.startswith("._")}
    ids = choose_images(run)
    order = list(ids)
    random.Random(a.rep).shuffle(order)  # same order for every method within a repetition
    methods = a.methods.split(",")
    ks = [int(x) for x in a.ks.split(",")]
    if a.rep % 2 == 0:
        methods, ks = methods[::-1], ks[::-1]  # alternate the method and k order between repetitions
    print(f"rep {a.rep}: {len(ids)} images {dict(Counter(run[i]['route_label'] for i in ids))}, methods {methods}, k {ks}")
    for m in methods:
        for k in ks:
            reqs = requests_for(m, a.rep, k, order, run, raw_a, raw_b, patch_a, patch_b, paths)
            kinds = Counter(q["kind"] for q in reqs)
            print(f"  {m} k={k}: {dict(kinds)}", flush=True)
            if not a.dry_run:
                run_condition(m, a.rep, k, reqs)


if __name__ == "__main__":
    main()
