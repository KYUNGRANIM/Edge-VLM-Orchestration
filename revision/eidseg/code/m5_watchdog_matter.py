#!/usr/bin/env python3
"""M5 워치독 (Matter 판) — 수집 서버가 조용해지면 Tapo P100M 을 Matter 로 껐다 켠다. Jetson 에서 상주 (2026-09-18).

왜 필요한가: M5 의 하드 wedge(검은 화면·리셋 버튼 무효)는 펌웨어 바깥의 문제라 **전원 제거로만** 풀린다(25/26건).
플러그 LAN 제어(python-kasa)는 9/11 펌웨어 자동 업데이트(TPAP)로 막혔고, 대신 P100M 의 Matter 를 Jetson 의
python-matter-server 로 잡았다(`matter_plug.py`). 이 스크립트는 그 위에서 "정지 → 3분 → 전원 사이클" 을 자동화한다.
Tapo 앱 예약(매시 :58 끔/:00 켬)과 공존한다 — 플러그가 꺼져 있는 동안과 켜진 직후 부팅 유예 동안은 판단하지 않는다.

판단 근거: 수집 서버 `/status` 의 `idle_s`(마지막 결과 뒤 경과) 와 플러그 OnOff 상태.

    정지     idle_s ≥ IDLE(180 s) · 플러그 ON · 부팅 유예(ON 전이·사이클 뒤 240 s) 지남 · 쿨다운(300 s) 지남
    사이클   OFF 10 s → ON. 로그 + events.jsonl 기록
    한도     시간당 6회. 사이클 뒤 결과가 안 늘어난 채 다시 정지가 3회 연속이면 30분 쉬고 NOTIFY
    플러그 OFF 가 10분 넘게 이어지면(예약 ON 실패?) NOTIFY 만 — 사람이 끈 것일 수 있어 켜지 않는다

로그의 `NOTIFY:` 줄은 맥 쪽 감시가 푸시 알림으로 올린다(14:00~23:00 규칙은 맥 쪽에서 적용).

    nohup python3 ~/m5_watchdog_matter.py --port 8020 --node 1 > ~/gate0_server/results/watchdog_matter.out 2>&1 &

이 저장소(오프라인 수집)에서는 수집 서버가 Pi_A 의 pi_ingest_server.py 이고 M5 가 세 대다.
`--status-url` 로 Pi_A 를 가리키고, `--expected` 에 이미지 수를 주면 M5 별 idle 중 아직 덜 끝난
M5 의 최댓값으로 판단한다(한 대만 멈춰도 잡고, 먼저 끝난 M5 는 멈춤으로 보지 않는다). 셋 다 끝나면 손대지 않는다.

    nohup python3 m5_watchdog_matter.py --status-url http://192.168.0.60:8000/status --expected 2931 --node 1 \
        > ~/m5_watchdog/watchdog.out 2>&1 &

런타임(held-out 600장)은 Pi_A `/online_status` 의 final_records 로 판단한다. M5 별로 구분이 안 되므로
플러그 a·b 워치독이 같은 신호로 같이 돈다. C 는 플러그가 없어 자동으로 못 살린다.

    nohup python3 m5_watchdog_matter.py --status-url http://192.168.0.60:8000/online_status --expected 600 \
        --idle-s 300 --node 2 --roles a > ~/m5_watchdog/watchdog_runtime.out 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matter_plug import Matter, ONOFF_CLUSTER, _node, _switch, onoff_of  # noqa: E402


_final = {"n": None, "t": 0.0}


def status(url: str, expected: int = 0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            st = json.load(r)
    except Exception:  # noqa: BLE001
        return None
    # 런타임 Pi_A `/online_status` 는 M5 별 수가 없고 final_records 뿐이다 (B·C 는 M5_A 로만 보낸다).
    # idle_s = final_records 가 마지막으로 는 뒤 경과. 한 대만 멈춰도 M5_A 가 기다려 final 이 멈춘다.
    if "final_records" in st:
        n = st["final_records"]
        if n != _final["n"]:
            _final.update(n=n, t=time.time())
        st.update(done=n, idle_s=time.time() - _final["t"], total=expected or "?",
                  remaining=max(0, expected - n) if expected else 1)
    return st


def effective_idle(st: dict, expected: int, only: list[str] | None = None) -> float | None:
    """덜 끝난 M5 중 가장 오래 조용한 시간(s). 모두 끝났으면 None. M5 별 정보가 없으면 서버 전체 idle_s.
    only 가 있으면 그 M5 들만 본다 (이 플러그에 꽂힌 M5)."""
    roles = st.get("roles")
    if not expected or not roles:
        return float(st.get("idle_s", 0))
    working = [r["idle_s"] for k, r in roles.items() if (not only or k in only) and r["received"] < expected]
    return max(working) if working else None


def _onoff(val, key: str) -> bool | None:
    if isinstance(val, dict):
        v = val.get(key)
        return None if v is None else bool(v)
    return None if val is None else bool(val)


async def _plug_state(nid: int) -> bool | None:
    async with Matter() as m:
        node = await _node(m, nid)
        ep, _ = onoff_of(node)
        if ep is None:
            return None
        key = f"{ep}/{ONOFF_CLUSTER}/0"
        return _onoff(await m.call("read_attribute", node_id=nid, attribute_path=key, timeout=20), key)


async def _plug_cycle(nid: int, off_s: float) -> bool | None:
    async with Matter() as m:
        node = await _node(m, nid)
        ep, _ = onoff_of(node)
        await _switch(m, nid, ep, False)
        await asyncio.sleep(off_s)
        await _switch(m, nid, ep, True)
        await asyncio.sleep(1.5)
        key = f"{ep}/{ONOFF_CLUSTER}/0"
        return _onoff(await m.call("read_attribute", node_id=nid, attribute_path=key, timeout=20), key)


async def _plug_on(nid: int) -> bool | None:
    async with Matter() as m:
        node = await _node(m, nid)
        ep, _ = onoff_of(node)
        await _switch(m, nid, ep, True)
        await asyncio.sleep(1.5)
        key = f"{ep}/{ONOFF_CLUSTER}/0"
        return _onoff(await m.call("read_attribute", node_id=nid, attribute_path=key, timeout=20), key)


def plug_on(nid: int) -> bool | None:
    try:
        return asyncio.run(asyncio.wait_for(_plug_on(nid), 60))
    except Exception:  # noqa: BLE001
        return None


def plug_state(nid: int) -> bool | None:
    try:
        return asyncio.run(asyncio.wait_for(_plug_state(nid), 40))
    except Exception:  # noqa: BLE001
        return None


def plug_cycle(nid: int, off_s: float) -> bool | None:
    try:
        return asyncio.run(asyncio.wait_for(_plug_cycle(nid, off_s), off_s + 60))
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--status-url", default=None, help="기본값 http://127.0.0.1:<port>/status")
    ap.add_argument("--expected", type=int, default=0, help="M5 한 대가 보낼 이미지 수 (0 = 모름)")
    ap.add_argument("--node", type=int, default=1)
    ap.add_argument("--roles", default="", help="이 플러그에 꽂힌 M5 (예: a 또는 b,c). 비우면 전부")
    ap.add_argument("--idle-s", type=float, default=180.0)
    ap.add_argument("--boot-grace-s", type=float, default=240.0)
    ap.add_argument("--cooldown-s", type=float, default=300.0)
    ap.add_argument("--off-s", type=float, default=10.0)
    ap.add_argument("--max-per-hour", type=int, default=6)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--log", type=Path, default=Path.home() / "m5_watchdog/watchdog_matter.log")
    ap.add_argument("--events", type=Path, default=Path.home() / "m5_watchdog/watchdog_events.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    status_url = args.status_url or f"http://127.0.0.1:{args.port}/status"
    only = [r.strip() for r in args.roles.split(",") if r.strip()] or None
    tag = f"[M5 {args.roles} · node {args.node}] " if only else ""

    args.log.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%m-%d %H:%M:%S')}  {tag}{msg}"
        print(line, flush=True)
        with args.log.open("a") as f:
            f.write(line + "\n")

    def event(kind: str, **kw) -> None:
        with args.events.open("a") as f:
            f.write(json.dumps({"t": time.time(), "ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, **kw}) + "\n")

    log(f"워치독(Matter) 시작 · 서버 {status_url} · 기대 {args.expected or '?'}장 · node {args.node} · 정지 {args.idle_s:.0f}s → OFF {args.off_s:.0f}s → ON"
        f" · 부팅 유예 {args.boot_grace_s:.0f}s · 쿨다운 {args.cooldown_s:.0f}s · 시간당 ≤{args.max_per_hour}"
        f"{' · DRY-RUN' if args.dry_run else ''}")

    cycles: list[float] = []
    last_cycle = 0.0
    grace_until = time.time() + args.boot_grace_s     # 시작 직후에도 한 번 유예
    prev_on: bool | None = None
    off_since: float | None = None
    off_notified = False
    done_at_cycle: int | None = None
    consecutive_fail = 0
    pause_until = 0.0
    server_missing = 0
    plug_unknown = 0

    while True:
        now = time.time()
        st = status(status_url, args.expected)
        if st is None:
            server_missing += 1
            if server_missing == 15:
                log("서버 /status 응답 없음 5분 — 서버가 죽었나? (M5 는 건드리지 않음)")
            time.sleep(args.poll)
            continue
        server_missing = 0
        # 회복 판단에 쓰는 진행 수: 이 플러그의 M5 중 가장 뒤처진 수. 합으로 보면 같은 플러그의 C 가 움직일 때
        # 멈춘 B 도 회복된 것으로 본다 (2026-11-02 21:32)
        done = min(st["roles"][r]["received"] for r in only) if only and st.get("roles") else st["done"]

        on = plug_state(args.node)
        if on is None:
            plug_unknown += 1
            if plug_unknown in (3, 30):
                log(f"NOTIFY: 플러그 상태 읽기 실패 {plug_unknown}회 — Matter 서버(:5580) 확인")
            time.sleep(args.poll)
            continue
        plug_unknown = 0

        # 플러그 OFF → 예약 또는 사람. 켜지 않는다. 오래 꺼져 있으면 알림만
        if not on:
            if off_since is None:
                off_since = now
                log(f"플러그 OFF 관측 (완료 {done}/{st.get('total', args.expected or '?')})")
            elif now - off_since > 600 and not off_notified:
                log("NOTIFY: 플러그가 10분 넘게 OFF — 예약 ON 실패이거나 사람이 끈 것. 켜지 않음")
                off_notified = True
            prev_on = False
            time.sleep(args.poll)
            continue
        if prev_on is False:                      # OFF → ON 전이 = 콜드부팅 시작
            log(f"플러그 ON 전이 (OFF {((now - off_since) if off_since else 0):.0f}s) → 부팅 유예 {args.boot_grace_s:.0f}s")
            grace_until = now + args.boot_grace_s
            off_since, off_notified = None, False
        prev_on = True

        idle = effective_idle(st, args.expected, only)
        if st.get("remaining", 1) == 0 or idle is None:   # 다 끝났으면 판단하지 않는다
            time.sleep(args.poll * 3)
            continue
        # 사이클 뒤 결과가 늘었으면 회복으로 본다
        if done_at_cycle is not None and done > done_at_cycle:
            log(f"회복 확인: 사이클 뒤 결과 재개 (완료 {done}) · {(now - last_cycle) / 60:.1f}분")
            event("recovered", done=done, after_s=round(now - last_cycle))
            done_at_cycle, consecutive_fail = None, 0

        if idle < args.idle_s or now < grace_until or now - last_cycle < args.cooldown_s or now < pause_until:
            time.sleep(args.poll)
            continue

        cycles = [c for c in cycles if now - c < 3600]
        if len(cycles) >= args.max_per_hour:
            log(f"NOTIFY: 정지 {idle / 60:.0f}분이지만 시간당 {args.max_per_hour}회 한도 — 30분 쉼")
            pause_until = now + 1800
            continue
        if done_at_cycle is not None and done <= done_at_cycle:
            consecutive_fail += 1
            if consecutive_fail >= 3:
                log(f"NOTIFY: 사이클 3회 연속 회복 없음 (완료 {done}) — 30분 쉼. SD/전원/모듈 점검 필요")
                event("give_up_pause", done=done)
                pause_until = now + 1800
                consecutive_fail = 0
                continue

        log(f"정지 {idle / 60:.1f}분 (완료 {done}/{st.get('total', args.expected or '?')}) → 전원 사이클{' (dry-run)' if args.dry_run else ''}")
        ok = True if args.dry_run else plug_cycle(args.node, args.off_s)
        # 2026-10-30 런타임: 사이클의 ON 명령이 한 번 안 먹어 M5_A 가 꺼진 채 남았다 (OFF 는 사람 몫이라 위에서 안 켠다)
        for attempt in range(3):
            if ok:
                break
            time.sleep(10)
            ok = plug_on(args.node)
            log(f"  ON 재시도 {attempt + 1}/3: 플러그 {'ON' if ok else 'OFF/불명'}")
        log(f"  사이클 결과: 플러그 {'ON' if ok else 'OFF/불명'}")
        event("cycle", done=done, idle_s=round(idle), ok=bool(ok), dry_run=args.dry_run)
        if not ok:
            log("NOTIFY: 사이클 뒤 플러그가 ON 으로 읽히지 않음 — 확인 필요")
        last_cycle = now
        grace_until = now + args.boot_grace_s
        done_at_cycle = done
        cycles.append(now)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
