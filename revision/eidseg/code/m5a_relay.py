#!/usr/bin/env python3
"""M5_A 결과 받기(:8010) 중계 — Jetson 에서 상주 (2026-11-02, EIDSeg 런타임).

왜: M5_B(192.168.0.139)가 와이파이에서 M5_A(192.168.0.77:8010)로 연결을 못 연다 (connect_fail, errno 113).
Mac·Jetson·C 는 A 에 닿고, Mac·Jetson 은 B 에도 닿는다. 그래서 B 는 Jetson:8020 으로 보내고, 이 스크립트가
바이트를 그대로 A 로 넘긴다 (HTTP 를 해석하지 않음). B 펌웨어는 `M5A_HOST=192.168.0.92`, `M5A_PORT=8020` 으로
다시 올린다. Jetson 방화벽은 8000·8020·5000·9000 만 연다 (8010 은 막힘, 8020 은 예전 워치독 자리).

    nohup python3 m5a_relay.py 192.168.0.77 8020 > ~/m5_watchdog/m5a_relay.out 2>&1 < /dev/null &
"""
import asyncio
import sys
from datetime import datetime

TARGET = [sys.argv[1] if len(sys.argv) > 1 else "192.168.0.77", 8010]
LISTEN_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8020
SUBNET = "192.168.0."   # M5_A 의 DHCP 주소가 바뀐다 (.30 → .77 → .30). 연결이 안 되면 이 대역에서 :8010 을 다시 찾는다


async def _open(host: str, timeout: float):
    return await asyncio.wait_for(asyncio.open_connection(host, TARGET[1]), timeout)


async def find_m5a() -> str | None:
    async def probe(host: str) -> str | None:
        try:
            _, w = await _open(host, 1.5)
            w.close()
            return host
        except (OSError, asyncio.TimeoutError):
            return None
    hits = [h for h in await asyncio.gather(*(probe(f"{SUBNET}{i}") for i in range(2, 255))) if h]
    return hits[0] if len(hits) == 1 else None   # 8010 을 여는 기기는 M5_A 뿐이어야 한다


def log(msg: str) -> None:
    print(f"{datetime.now():%m-%d %H:%M:%S}  {msg}", flush=True)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()


async def handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
    peer = client_w.get_extra_info("peername")[0]
    try:
        a_r, a_w = await _open(TARGET[0], 3)
    except (OSError, asyncio.TimeoutError) as e:
        found = await find_m5a()
        log(f"{peer} -> M5_A {TARGET[0]} 연결 실패 ({e!r}) → 다시 찾음: {found or '못 찾음'}")
        if not found:
            client_w.close()
            return
        TARGET[0] = found
        try:
            a_r, a_w = await _open(TARGET[0], 3)
        except (OSError, asyncio.TimeoutError):
            client_w.close()
            return
    await asyncio.gather(pipe(client_r, a_w), pipe(a_r, client_w))
    log(f"{peer} -> M5_A {TARGET[0]} 전달")


async def main() -> None:
    server = await asyncio.start_server(handle, "0.0.0.0", LISTEN_PORT)
    log(f"중계 시작 :{LISTEN_PORT} -> {TARGET[0]}:{TARGET[1]}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
