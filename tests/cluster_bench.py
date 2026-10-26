#!/usr/bin/env python3
"""Cluster-mode throughput comparison: vanilla vs patched redis-server.

Builds a 1-node cluster (cluster-enabled yes, all 16384 slots assigned via
CLUSTER ADDSLOTS RANGE is unnecessary for benchmark traffic since a single
node with slots works; we assign the full range), runs redis-benchmark
GET with -P 16 in alternating order (A B B A) on the same machine, and
reports throughput deltas.

Note: 1-node cluster still routes CLUSTER SLOTS to itself, and every command
goes through keyHashSlot(), which is exactly the hot path under test.
"""
import argparse
import os
import re
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


def start_cluster_node(binary, port, workdir, extra_env=None):
    conf = os.path.join(workdir, f"node-{port}.conf")
    with open(conf, "w") as f:
        f.write(f"port {port}\n"
                f"dir {workdir}\n"
                "save ''\n"
                "appendonly no\n"
                "daemonize no\n"
                "logfile ''\n"
                "cluster-enabled yes\n"
                f"cluster-config-file nodes-{port}.conf\n"
                "cluster-node-timeout 15000\n"
                "cluster-require-full-coverage no\n")
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    p = subprocess.Popen([binary, conf], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, env=env)
    if not wait_port(port):
        p.kill()
        raise RuntimeError(f"node on {port} did not start")
    return p


def resp_cmd(port, *args):
    out = f"*{len(args)}\r\n".encode()
    for a in args:
        b = str(a).encode()
        out += f"${len(b)}\r\n".encode() + b + b"\r\n"
    with socket.create_connection(("127.0.0.1", port), 5) as s:
        s.sendall(out)
        buf = b""
        while b"\r\n" not in buf:
            chunk = s.recv(128)
            if not chunk:
                break
            buf += chunk
        return buf.strip().decode()


def assign_all_slots(port):
    """CLUSTER ADDSLOTS for all 16384 slots in ranges (bulk)."""
    # redis-cli would be handy; do it via raw RESP in ranges of 1000.
    start = 0
    while start < 16384:
        end = min(start + 999, 16383)
        r = resp_cmd(port, "CLUSTER", "ADDSLOTSRANGE", start, end)
        if "OK" not in r and "error" not in r.lower():
            pass
        start = end + 1
    time.sleep(1.0)


def run_bench(bench_bin, port, clients, pipeline, requests, outdir, tag,
              cluster_client=False):
    out = os.path.join(outdir, f"bench-{tag}.txt")
    cmd = [bench_bin, "-h", "127.0.0.1", "-p", str(port),
           "-c", str(clients), "-t", "get",
           "-n", str(requests), "-P", str(pipeline), "--csv"]
    if cluster_client:
        cmd.insert(5, "--cluster")
    # --csv writes to stdout in 7.0; capture instead
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    with open(out, "w") as f:
        f.write(r.stdout)
        f.write(r.stderr)
    # parse "GET","<rps>",... from stdout (--csv mode)
    m = re.search(r'"GET","([\d.]+)"', r.stdout)
    return float(m.group(1)) if m else 0.0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-bin", required=True)
    ap.add_argument("--patched-bin", required=True)
    ap.add_argument("--bench-bin", required=True)
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--pipeline", type=int, default=16)
    ap.add_argument("--requests", type=int, default=2000000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--bench-cluster", action="store_true",
                    help="pass --cluster to redis-benchmark (hashes only the "
                         "{tag} content, 1-3 bytes; default off matches the "
                         "requirement command line, where the server hashes "
                         "the full 16-byte key)")
    args = ap.parse_args()

    results = {"vanilla": [], "patched": []}
    with tempfile.TemporaryDirectory() as dv, tempfile.TemporaryDirectory() as dp:
        pv = start_cluster_node(args.vanilla_bin, 7121, dv)
        pp = start_cluster_node(args.patched_bin, 7122, dp,
                                extra_env={"REDIS_CRC16_ANY_PMULL_CPU": "1"})
        try:
            assign_all_slots(7121)
            assign_all_slots(7122)
            # warm both
            for port in (7121, 7122):
                run_bench(args.bench_bin, port, args.clients, args.pipeline,
                          args.requests // 10, dv, f"warm-{port}",
                          cluster_client=args.bench_cluster)
            order = ["vanilla", "patched"]
            for r in range(args.rounds):
                # alternate order every round to cancel thermal/drift bias
                seq = order if r % 2 == 0 else order[::-1]
                for which in seq:
                    port = 7121 if which == "vanilla" else 7122
                    rps, _ = run_bench(args.bench_bin, port, args.clients,
                                       args.pipeline, args.requests,
                                       dv, f"{which}-r{r}",
                                       cluster_client=args.bench_cluster)
                    results[which].append(rps)
                    print(f"round {r} {which:8s}: {rps:12.1f} ops/s")
        finally:
            pv.kill()
            pp.kill()

    v = results["vanilla"]
    p = results["patched"]
    vavg = sum(v) / len(v)
    pavg = sum(p) / len(p)
    delta = (pavg - vavg) / vavg * 100.0
    print()
    print(f"vanilla avg: {vavg:12.1f} ops/s  runs: {[f'{x:.0f}' for x in v]}")
    print(f"patched avg: {pavg:12.1f} ops/s  runs: {[f'{x:.0f}' for x in p]}")
    print(f"delta: {delta:+.2f}%")
    with open("clusterbench-summary.txt", "w") as f:
        f.write(f"vanilla_avg={vavg:.1f}\npatched_avg={pavg:.1f}\ndelta={delta:+.2f}%\n")


if __name__ == "__main__":
    sys.exit(main())
