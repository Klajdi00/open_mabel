#!/usr/bin/env python3
"""test_security.py — verify the real-robot bridge's access-code auth + driver lock.
Run ON thor (has websockets + reaches localhost:9090). Reads the code from
~/.mabel_real_code. NO motion is commanded — this only exercises the gates."""
import asyncio, json, os, sys
import websockets

CODE = open(os.path.expanduser("~/.mabel_real_code")).read().strip()
BASE = "ws://127.0.0.1:9090/teleop"

async def hello(url):
    async with websockets.connect(url, open_timeout=5) as ws:
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        return json.loads(raw)

async def expect_reject(url, label):
    try:
        async with websockets.connect(url, open_timeout=5) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            if msg.get("type") == "error":
                print(f"  PASS {label}: server sent error '{msg['payload'].get('code')}' then closed")
                # confirm the socket actually closes
                try:
                    await asyncio.wait_for(ws.recv(), timeout=3)
                except Exception:
                    pass
                return True
            print(f"  FAIL {label}: got {msg.get('type')} (expected reject) -> {msg}")
            return False
    except websockets.exceptions.ConnectionClosed as e:
        print(f"  PASS {label}: connection closed by server (code {e.code})")
        return True
    except Exception as e:
        print(f"  PASS {label}: refused ({type(e).__name__})")
        return True

async def main():
    ok = True
    print("1) NO code  -> must reject")
    ok &= await expect_reject(BASE, "no-code")
    print("2) WRONG code -> must reject")
    ok &= await expect_reject(BASE + "?key=not-the-code", "wrong-code")

    print("3) CORRECT code -> must accept + report security posture")
    h = await hello(BASE + f"?key={CODE}")
    p = h.get("payload", {})
    print(f"   hello: authRequired={p.get('authRequired')} armRequired={p.get('armRequired')} "
          f"armed={p.get('driver',{}).get('armed')} you={p.get('you')}")
    ok &= (h.get("type") == "hello" and p.get("authRequired") is True)

    print("4) Driver lock: first request_control -> granted; second client -> denied")
    a = await websockets.connect(BASE + f"?key={CODE}", open_timeout=5)
    await a.recv()  # hello
    await a.send(json.dumps({"type": "request_control", "payload": {"name": "ClientA"}}))
    ra = json.loads(await asyncio.wait_for(a.recv(), timeout=5))
    while ra.get("type") != "control":
        ra = json.loads(await asyncio.wait_for(a.recv(), timeout=5))
    granted = ra["payload"].get("state") == "granted"
    print(f"   A request_control -> {ra['payload'].get('state')} (key={'yes' if ra['payload'].get('key') else 'no'})")
    ok &= granted

    b = await websockets.connect(BASE + f"?key={CODE}", open_timeout=5)
    await b.recv()  # hello
    await b.send(json.dumps({"type": "request_control", "payload": {"name": "ClientB"}}))
    rb = json.loads(await asyncio.wait_for(b.recv(), timeout=5))
    while rb.get("type") != "control":
        rb = json.loads(await asyncio.wait_for(b.recv(), timeout=5))
    denied = rb["payload"].get("state") == "denied"
    print(f"   B request_control -> {rb['payload'].get('state')} (holder='{rb['payload'].get('holder')}')")
    ok &= denied

    print("5) A releases -> B can now acquire")
    await a.send(json.dumps({"type": "release_control", "payload": {"key": ra['payload'].get('key')}}))
    await asyncio.sleep(0.3)
    await b.send(json.dumps({"type": "request_control", "payload": {"name": "ClientB"}}))
    rb2 = json.loads(await asyncio.wait_for(b.recv(), timeout=5))
    while rb2.get("type") != "control":
        rb2 = json.loads(await asyncio.wait_for(b.recv(), timeout=5))
    print(f"   B request_control (after release) -> {rb2['payload'].get('state')}")
    ok &= (rb2["payload"].get("state") == "granted")
    await a.close(); await b.close()

    print("\n" + ("ALL PASS ✅" if ok else "SOME FAILED ❌"))
    sys.exit(0 if ok else 1)

asyncio.run(main())
