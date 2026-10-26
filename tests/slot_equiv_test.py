#!/usr/bin/env python3
"""Slot-mapping equivalence test between vanilla and patched redis-server.

Starts two standalone servers (cluster-enabled but single-node, no cluster
meet needed for CLUSTER KEYSLOT), then for a large key corpus compares the
slot reported by CLUSTER KEYSLOT on both. Any difference is a hard failure.

This exercises the real server path: server main() -> crc16_init() ->
keyHashSlot() -> crc16_fast()/crc16().
"""
import argparse
import os
import random
import socket
import subprocess
import sys
import tempfile
import time


def wait_port(port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def start_server(binary, port, workdir, extra_env=None):
    conf = os.path.join(workdir, f"redis-{port}.conf")
    with open(conf, "w") as f:
        f.write(f"port {port}\n"
                f"dir {workdir}\n"
                "save ''\n"
                "appendonly no\n"
                "daemonize no\n"
                "logfile ''\n"
                "cluster-enabled yes\n"
                f"cluster-config-file nodes-{port}.conf\n"
                "cluster-node-timeout 15000\n")
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    p = subprocess.Popen([binary, conf], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, env=env)
    if not wait_port(port):
        p.kill()
        raise RuntimeError(f"server on {port} did not start")
    return p


def resp_query(port, *args):
    """Minimal RESP client: send command, read the first reply line."""
    out = f"*{len(args)}\r\n".encode()
    for a in args:
        b = str(a).encode()
        out += f"${len(b)}\r\n".encode() + b + b"\r\n"
    with socket.create_connection(("127.0.0.1", port), 2) as s:
        s.sendall(out)
        buf = b""
        while b"\r\n" not in buf:
            chunk = s.recv(64)
            if not chunk:
                break
            buf += chunk
    return buf.strip().decode()


def gen_keys(n, rng):
    """Corpus mimicking redis-benchmark traffic + adversarial shapes."""
    keys = []
    # 1. benchmark-style keys with slot hashtags
    for i in range(n // 4):
        keys.append("key:%d:__rand_int__" % rng.getrandbits(31))
    # 2. hash-tag keys of every length class around the 64B boundary
    for L in list(range(1, 130)) * (n // 4 // 130 + 1):
        body = "".join(chr(33 + rng.randrange(90)) for _ in range(L))
        keys.append("{" + body + "}:tail")
    # 3. binary keys (all byte values)
    for i in range(n // 4):
        keys.append(bytes(rng.getrandbits(8) for _ in range(
            rng.choice([3, 16, 63, 64, 65, 127, 128, 200]))).decode(
                "latin-1"))
    # 4. plain keys, no tags
    for i in range(n - len(keys)):
        keys.append("user:%d:profile" % rng.getrandbits(31))
    return keys[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-bin", required=True)
    ap.add_argument("--patched-bin", required=True)
    ap.add_argument("--keys", type=int, default=50000)
    ap.add_argument("--port-a", type=int, default=7111)
    ap.add_argument("--port-b", type=int, default=7112)
    args = ap.parse_args()

    rng = random.Random(20261026)
    keys = gen_keys(args.keys, rng)
    print(f"corpus: {len(keys)} keys")

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        pa = start_server(args.vanilla_bin, args.port_a, d1)
        # Patched server with the Kunpeng gate relaxed (CI hosts are Ampere).
        pb = start_server(args.patched_bin, args.port_b, d2,
                          extra_env={"REDIS_CRC16_ANY_PMULL_CPU": "1"})
        try:
            bad = 0
            for i, k in enumerate(keys):
                ka = resp_query(args.port_a, "CLUSTER", "KEYSLOT", k)
                kb = resp_query(args.port_b, "CLUSTER", "KEYSLOT", k)
                if ka != kb:
                    bad += 1
                    if bad < 10:
                        print(f"MISMATCH key={k!r}: vanilla={ka} patched={kb}")
            print(f"slot equivalence: {len(keys) - bad}/{len(keys)} keys match")
            if bad:
                print("RESULT: FAIL")
                return 1
            print("RESULT: PASS")
            return 0
        finally:
            pa.kill()
            pb.kill()


if __name__ == "__main__":
    sys.exit(main())
