#!/usr/bin/env python3
"""Tapo P100M 을 Matter 로 LAN 제어 — python-matter-server 의 WebSocket API 를 직접 두드린다 (2026-09-18).

왜 Matter 인가: 플러그 펌웨어가 9/11 자동 업데이트로 TPAP 암호화가 되어 python-kasa/tapo 가 LAN 제어를 못 한다.
P100M 은 Matter 인증 기기라서 TP-Link 프로토콜과 무관하게 **Matter 컨트롤러(= 이 서버)** 가 LAN 에서 켜고 끌 수 있다.
서버는 Jetson 에서 `python3 -m matter_server.server --storage-path ~/matter-data --port 5580` 로 상주한다.
(fabric 자격은 storage-path 에 저장되므로 그 폴더를 지우면 다시 페어링해야 한다.)

    matter_plug.py commission <설정코드>       # Tapo 앱 "Matter 설정 코드" 화면이 열려 있는 동안(≈15분) 1회
    matter_plug.py nodes                        # 등록된 노드와 OnOff 상태
    matter_plug.py state 1                      # 노드 1 의 OnOff 읽기
    matter_plug.py off 1 / on 1                 # 끄기 / 켜기 (M5 전원이므로 신중히)
    matter_plug.py cycle 1 --off-seconds 10     # 전원 사이클

API 참고: 메시지 {"message_id","command","args"} → 응답 {"message_id","result"|"error_code","details"}.
OnOff 클러스터 id 6, 속성 0(OnOff), 명령 On/Off. 플러그의 릴레이 엔드포인트는 보통 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp

WS_URL = "ws://127.0.0.1:5580/ws"
ONOFF_CLUSTER = 6


class Matter:
    def __init__(self, url: str = WS_URL):
        self.url = url
        self._id = 0

    async def __aenter__(self):
        self._sess = aiohttp.ClientSession()
        self.ws = await self._sess.ws_connect(self.url, timeout=30)
        self.info = await self.ws.receive_json()          # ServerInfoMessage
        return self

    async def __aexit__(self, *a):
        await self.ws.close()
        await self._sess.close()

    async def call(self, command: str, timeout: float = 120, **args):
        self._id += 1
        mid = str(self._id)
        await self.ws.send_json({"message_id": mid, "command": command, "args": args})
        t0 = time.time()
        while True:
            m = await asyncio.wait_for(self.ws.receive_json(), timeout=max(1.0, timeout - (time.time() - t0)))
            if m.get("message_id") != mid:
                continue                                   # 이벤트(event) 또는 다른 응답
            if "error_code" in m:
                raise RuntimeError(f"{command}: error {m.get('error_code')} {m.get('details')}")
            return m.get("result")


def onoff_of(node: dict) -> tuple[int | None, bool | None]:
    """노드 dict 의 attributes 에서 OnOff 엔드포인트와 상태를 찾는다."""
    attrs = node.get("attributes") or {}
    for key, val in attrs.items():
        try:
            ep, cl, at = (int(x) for x in key.split("/"))
        except ValueError:
            continue
        if cl == ONOFF_CLUSTER and at == 0:
            return ep, bool(val)
    return None, None


async def cmd_commission(code: str) -> int:
    code = code.replace("-", "").replace(" ", "")
    async with Matter() as m:
        print("서버:", {k: m.info.get(k) for k in ("sdk_version", "schema_version", "fabric_id", "bluetooth_enabled")})
        nodes = await m.call("start_listening")
        print("기존 노드:", [n.get("node_id") for n in nodes or []])
        print(f"페어링 시작 code={code[:4]}… (네트워크 전용) — 30~90초")
        node = await m.call("commission_with_code", timeout=240, code=code, network_only=True)
        nid = node.get("node_id")
        ep, st = onoff_of(node)
        print(f"완료: node_id={nid} · OnOff 엔드포인트={ep} · 현재={'ON' if st else 'OFF' if st is not None else '?'}")
        print("  vendor/product:", node.get("attributes", {}).get("0/40/1"), node.get("attributes", {}).get("0/40/3"))
        return 0


async def cmd_nodes() -> int:
    async with Matter() as m:
        nodes = await m.call("start_listening")
        for n in nodes or []:
            ep, st = onoff_of(n)
            print(f"node {n.get('node_id')} · available={n.get('available')} · OnOff ep={ep} state={'ON' if st else 'OFF' if st is not None else '?'}"
                  f" · {n.get('attributes', {}).get('0/40/3')}")
        if not nodes:
            print("등록된 노드 없음 — 먼저 commission")
        return 0


async def _node(m: Matter, nid: int) -> dict:
    await m.call("start_listening")
    return await m.call("get_node", node_id=nid)


async def cmd_state(nid: int) -> int:
    async with Matter() as m:
        node = await _node(m, nid)
        ep, _ = onoff_of(node)
        val = await m.call("read_attribute", node_id=nid, attribute_path=f"{ep}/{ONOFF_CLUSTER}/0")
        st = val.get(f"{ep}/{ONOFF_CLUSTER}/0") if isinstance(val, dict) else val
        print(f"node {nid} ep {ep}: {'ON' if st else 'OFF'}")
        return 0


async def _switch(m: Matter, nid: int, ep: int, on: bool) -> None:
    await m.call("device_command", node_id=nid, endpoint_id=ep, cluster_id=ONOFF_CLUSTER,
                 command_name="On" if on else "Off", payload={})


async def cmd_switch(nid: int, on: bool) -> int:
    async with Matter() as m:
        node = await _node(m, nid)
        ep, _ = onoff_of(node)
        await _switch(m, nid, ep, on)
        await asyncio.sleep(1.0)
        val = await m.call("read_attribute", node_id=nid, attribute_path=f"{ep}/{ONOFF_CLUSTER}/0")
        st = val.get(f"{ep}/{ONOFF_CLUSTER}/0") if isinstance(val, dict) else val
        print(f"node {nid}: {'ON' if on else 'OFF'} 명령 → 읽음 {'ON' if st else 'OFF'}")
        return 0 if bool(st) == on else 1


async def cmd_cycle(nid: int, off_seconds: float) -> int:
    async with Matter() as m:
        node = await _node(m, nid)
        ep, _ = onoff_of(node)
        t0 = time.time()
        await _switch(m, nid, ep, False)
        print(f"OFF ({time.strftime('%H:%M:%S')}) … {off_seconds:.0f}s")
        await asyncio.sleep(off_seconds)
        await _switch(m, nid, ep, True)
        await asyncio.sleep(1.0)
        val = await m.call("read_attribute", node_id=nid, attribute_path=f"{ep}/{ONOFF_CLUSTER}/0")
        st = val.get(f"{ep}/{ONOFF_CLUSTER}/0") if isinstance(val, dict) else val
        print(f"ON  ({time.strftime('%H:%M:%S')}) 읽음 {'ON' if st else 'OFF'} · 왕복 {time.time() - t0:.1f}s")
        return 0 if st else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("commission"); s.add_argument("code")
    sub.add_parser("nodes")
    for name in ("state", "on", "off"):
        s = sub.add_parser(name); s.add_argument("node", type=int)
    s = sub.add_parser("cycle"); s.add_argument("node", type=int); s.add_argument("--off-seconds", type=float, default=10)
    a = ap.parse_args()
    try:
        if a.cmd == "commission":
            rc = asyncio.run(cmd_commission(a.code))
        elif a.cmd == "nodes":
            rc = asyncio.run(cmd_nodes())
        elif a.cmd == "state":
            rc = asyncio.run(cmd_state(a.node))
        elif a.cmd in ("on", "off"):
            rc = asyncio.run(cmd_switch(a.node, a.cmd == "on"))
        else:
            rc = asyncio.run(cmd_cycle(a.node, a.off_seconds))
    except Exception as e:  # noqa: BLE001
        print(f"실패: {type(e).__name__}: {e}", file=sys.stderr)
        rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
