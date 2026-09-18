"""Multi-host raft-rs cluster benchmark over real LXD container network.

Runs on the xrserver host. Starts one raft_node_server per LXD container,
elects a leader across the network, measures sustained proposal throughput,
kills the leader container (real failover), verifies quorum writes on the
remaining two nodes, restarts the killed node and reports log catch-up.
"""
from __future__ import annotations
import json
import socket
import subprocess
import sys
import time

VENV_PY = "/srv/ai/workspaces/llm-lora/.venv/bin/python"
PROJECT = "/home/xrey/neural-pods"
SERVER = f"{PROJECT}/research/raft_node_server.py"
NODES = ["np-node1", "np-node2", "np-node3"]
CONTROL_PORT = 45400


def lxc(*args: str) -> str:
    return subprocess.run(["lxc", *args], capture_output=True, text=True, check=True).stdout


def node_ips() -> dict[str, str]:
    ips: dict[str, str] = {}
    for row in lxc("list", "--format", "csv", "-c", "n4").splitlines():
        name, addr = row.split(",", 1)
        ips[name.strip()] = addr.strip().split(" ")[0]
    return {name: ips[name] for name in NODES if name in ips}


def control(ip: str, cmd: dict, timeout: float = 5.0) -> dict:
    with socket.create_connection((ip, CONTROL_PORT), timeout=timeout) as conn:
        conn.settimeout(timeout)
        conn.sendall(json.dumps(cmd).encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            part = conn.recv(4096)
            if not part:
                break
            buf += part
    return json.loads(buf)


def launch_server(ident: int, peers: dict[int, str]) -> None:
    # pkill runs in its own exec: its target pattern must not appear in the
    # launching shell's own command line.
    subprocess.run(["lxc", "exec", NODES[ident - 1], "--", "pkill", "-f",
                    "raft_node_server"], capture_output=True)
    time.sleep(0.3)
    peer_spec = ",".join(f"{i}={host}" for i, host in peers.items())
    script = (f"nohup {VENV_PY} {SERVER} --ident {ident} --peers '{peer_spec}' "
              f">/tmp/raft-node-{ident}.log 2>&1 &")
    lxc("exec", NODES[ident - 1], "--", "bash", "-c", script)


def stop_server(ip: str) -> None:
    try:
        control(ip, {"cmd": "shutdown"}, timeout=2.0)
    except OSError:
        pass


def wait_leader(ips: list[str], timeout: float) -> tuple[str | None, float]:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        for ip in ips:
            try:
                status = control(ip, {"cmd": "status"}, timeout=2.0)
            except OSError:
                continue
            if status.get("state") == "Leader":
                return ip, time.perf_counter() - started
        time.sleep(0.1)
    return None, time.perf_counter() - started


def wait_applied(ips: list[str], target: int, timeout: float) -> bool:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        counts = []
        for ip in ips:
            try:
                counts.append(control(ip, {"cmd": "status"}, timeout=2.0).get("applied", 0))
            except OSError:
                counts.append(-1)
        if all(c >= target for c in counts):
            return True
        time.sleep(0.1)
    return False


def run(count: int = 100, failover_count: int = 50) -> dict:
    ips_by_name = node_ips()
    if len(ips_by_name) != 3:
        raise SystemExit(f"expected 3 running nodes, found: {ips_by_name}")
    ips = list(ips_by_name.values())  # index i -> ident i+1
    result: dict = {"nodes": 3, "ips": ips_by_name}

    try:
        for ident in (1, 2, 3):
            peers = {i: NODES[i - 1] for i in (1, 2, 3) if i != ident}
            launch_server(ident, peers)
        deadline = time.time() + 15
        ready = []
        while time.time() < deadline and len(ready) < 3:
            ready = []
            for ip in ips:
                try:
                    state = control(ip, {"cmd": "status"}, timeout=1.0).get("state")
                except OSError:
                    continue
                if state in ("Leader", "Follower", "Candidate"):
                    ready.append(ip)
            time.sleep(0.3)

        control(ips[0], {"cmd": "campaign"})
        leader_ip, election_s = wait_leader(ips, 10.0)
        if leader_ip is None:
            raise SystemExit("no leader elected across hosts")
        result["election_s"] = round(election_s, 3)

        started = time.perf_counter()
        for i in range(count):
            control(leader_ip, {"cmd": "propose", "data": f"p-{i}"}, timeout=10.0)
        replicated = wait_applied(ips, count, 30.0)
        elapsed = time.perf_counter() - started
        result["proposals"] = count
        result["replicated_all"] = replicated
        result["proposals_per_s"] = round(count / max(elapsed, 1e-9), 1)

        # Real failover: stop the whole leader container.
        leader_name = next(n for n, ip in ips_by_name.items() if ip == leader_ip)
        follower_ips = [ip for ip in ips if ip != leader_ip]
        lxc("stop", leader_name)
        new_leader, failover_s = wait_leader(follower_ips, 15.0)
        result["failover"] = {
            "killed": leader_name,
            "new_leader_ip": new_leader,
            "failover_s": round(failover_s, 3),
            "ok": new_leader is not None,
        }
        if new_leader is None:
            return result
        for i in range(failover_count):
            control(new_leader, {"cmd": "propose", "data": f"f-{i}"}, timeout=10.0)
        quorum_ok = wait_applied(follower_ips, count + failover_count, 30.0)
        result["failover"]["quorum_writes"] = failover_count if quorum_ok else 0

        # Restart the killed node and check log catch-up.
        lxc("start", leader_name)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                addr = node_ips().get(leader_name, "")
                if addr:
                    break
            except Exception:
                pass
            time.sleep(1)
        time.sleep(3)
        peers = {i: NODES[i - 1] for i in (1, 2, 3) if i != NODES.index(leader_name) + 1}
        launch_server(NODES.index(leader_name) + 1, peers)
        catchup = wait_applied([node_ips()[leader_name]], count + failover_count, 60.0)
        result["failover"]["node_rejoined"] = bool(catchup)
        if not catchup:
            diag = {}
            for name, ip in node_ips().items():
                try:
                    diag[name] = control(ip, {"cmd": "status"}, timeout=2.0)
                except OSError as exc:
                    diag[name] = f"unreachable: {exc}"
            result["failover"]["diagnosis"] = diag

        for ip in ips:
            stop_server(ip)
        for name, ip in node_ips().items():
            try:
                control(ip, {"cmd": "shutdown"}, timeout=1.0)
            except OSError:
                pass
        return result
    finally:
        for name in NODES:
            subprocess.run(["lxc", "start", name], capture_output=True)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
