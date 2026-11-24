#!/usr/bin/env python3
"""M5 진행 신호 받기 (2026-11-02, EIDSeg 런타임) — Jetson :8020 에서 상주.

왜: 런타임에서는 B·C 가 M5_A 로만 보내서 Pi_A 는 M5 별 진행을 모른다. 그래서 워치독이 Pi_A 완료 수 하나로
판단했고, 느린 C 를 기다리던 A 까지 껐다. M5 가 사진을 시작할 때마다 POST /hb {"device":"M5_A","idx":151}
을 보내면, 이 서버가 워치독(m5_watchdog_matter.py)이 읽는 모양으로 GET /status 를 돌려준다:
{"roles": {"a": {"received": 151, "idle_s": 12.3}, ...}}. idle_s 는 idx 가 마지막으로 바뀐 뒤 경과.
신호가 한 번도 없는 기기도 서버 시작 시각부터 센다 (부팅 중 멈춘 기기도 잡으려고).

    nohup python3 m5_heartbeat_server.py > ~/m5_watchdog/heartbeat.out 2>&1 < /dev/null &
    nohup python3 m5_heartbeat_server.py 37 ...   # 빠진 사진만 다시 돌릴 때 (SD 의 retry 목록 길이)
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

END = int(sys.argv[1]) if len(sys.argv) > 1 else 600  # = RUN_END in platformio.ini, or the retry list size
LONG_IDLE_S = 2700  # 45 min: longer than any wait for a lagging device seen in the EIDSeg run (~40 min)
roles = {r: {"received": -1, "t": time.time()} for r in "abc"}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            role, idx = str(d["device"])[-1].lower(), int(d["idx"])
            st = roles.setdefault(role, {"received": -1, "t": time.time()})
            if idx != st["received"]:
                st.update(received=idx, t=time.time())
                print(f"{time.strftime('%m-%d %H:%M:%S')}  {role} {idx}", flush=True)
            self.send_response(200)
        except (ValueError, KeyError, TypeError):
            self.send_response(400)
        self.end_headers()

    def do_GET(self):
        # Only the device furthest behind is reported idle: every image needs all three, so the laggard is
        # the bottleneck. A stalled B makes A throttle and C hit backpressure, and cutting those waiting
        # devices only loses data. A wedged A still accepts B/C answers, so B/C run ahead and A becomes the
        # laggard. "Stopped first" failed when A throttled 9 s before B stalled (2026-11-02 21:36).
        # Devices that reached END are left out. A device that is ahead but stopped far longer than any wait
        # for the others is reported too: on 2026-11-03 M5_A hung at 139 for 62 min while C (80) was the laggard.
        now = time.time()
        raw = {r: now - s["t"] for r, s in roles.items() if s["received"] < END}
        first = min(raw, key=lambda r: roles[r]["received"]) if raw else None
        body = json.dumps({"roles": {r: {"received": s["received"],
                                         "idle_s": round(raw[r], 1) if r == first or raw.get(r, 0) > LONG_IDLE_S else 0.0,
                                         "raw_idle_s": round(now - s["t"], 1)}
                                     for r, s in roles.items()}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8020), Handler).serve_forever()
