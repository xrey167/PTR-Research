"""Mesh presence + RTT benchmark.

1. Discovery: this host endpoint discovers a peer process running inside
   LXD node np-node2 purely via MQTT presence (no static peer config).
2. RTT: two host endpoints exchange ping/pong over the remote broker in
   np-node1; 100 sequential round trips, p50/p99 reported.

WHAT THE RTT IS BETWEEN, recorded as a field and not only as prose. The two
endpoints both run ON THE HOST; only the broker is remote. The 0.30 ms this
produces has been quoted as a cross-node mesh latency in the master document
and in three design documents, and it is not one. `rtt_scope` now says so
inside the evidence, where a reader of the number will see it.

STRUCTURE. `collect()` needs the broker and LXD; `summarise()` needs neither.
The percentiles and the "did every round come back" verdict are arithmetic
over the samples, and arithmetic that cannot be run without a broker is
arithmetic nobody checks.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.mesh import MeshEndpoint  # noqa: E402
from research.evidence import write as write_evidence  # noqa: E402

#: The modules these numbers are evidence ABOUT.
SUBJECT = ["neural_pods/mesh.py"]

BROKER = "10.50.0.121"
PEER_NODE = "np-node2"
PEER_ID = "mesh-node2"
ROUNDS = 100


def run_discovery(endpoint: MeshEndpoint) -> dict:
    """Start the container peer, wait for its live ready signal."""
    subprocess.run(["lxc", "exec", PEER_NODE, "--", "pkill", "-f", "mesh_peer"],
                   capture_output=True)
    ready_event = threading.Event()
    seen = {"state": None}

    def on_presence(topic: str, envelope: dict) -> None:
        if envelope.get("pod_id") == PEER_ID and envelope.get("state") in ("online", "ready"):
            seen["state"] = envelope["state"]
            if envelope.get("state") == "ready":
                ready_event.set()

    endpoint.subscribe(f"np/presence/{PEER_ID}", on_presence)
    peer_cmd = (f"nohup {sys.executable} /home/xrey/neural-pods/research/mesh_peer.py "
                f"--broker {BROKER} --pod-id {PEER_ID} --lifetime-s 60 "
                f">/tmp/mesh-peer.log 2>&1 &")
    subprocess.run(["lxc", "exec", PEER_NODE, "--", "bash", "-c", peer_cmd], check=True)
    ok = ready_event.wait(timeout=60.0)
    return {"discovered": ok, "peer_state": seen["state"]}


def percentile(values: list[float], p: float) -> float | None:
    """The repository's convention, used identically in every benchmark.

    None for an empty sample: the gate must be able to tell "fast" from
    "never measured", and a 0.0 here would read as the former.
    """
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * p))], 2)


def summarise(*, discovery: dict, rtts: list[float], rounds_sent: int,
              endpoint_stats: dict, responder_stats: dict) -> dict:
    """Turn the observations into the evidence the gate reads. Pure."""
    return {
        "status": "completed",
        "discovery": discovery,
        "rounds_sent": rounds_sent,
        "rounds_ok": len(rtts),
        "rounds_lost": rounds_sent - len(rtts),
        "rtt_p50_ms": percentile(rtts, 0.5),
        "rtt_p99_ms": percentile(rtts, 0.99),
        # Stated in the evidence, not only in the note below: both endpoints
        # run on the host and only the broker is remote. This number has been
        # quoted as a cross-node mesh latency, and it is not one.
        "rtt_scope": "host endpoint -> remote broker -> host endpoint",
        "rtt_is_cross_node": False,
        "endpoint_stats": endpoint_stats,
        "responder_stats": responder_stats,
        "note": ("Presence discovery host<->container peer (np-node2); "
                 "RTT between two host endpoints over the remote broker "
                 "in np-node1"),
    }


def collect() -> dict:
    """Discover the peer and measure the round trips. Needs broker and LXD."""
    endpoint = MeshEndpoint(BROKER, "mesh-host", manifest_hash="host-manifest")
    responder = None
    try:
        discovery = run_discovery(endpoint)
        print(json.dumps({"discovery": discovery}), flush=True)
        if not discovery["discovered"]:
            raise SystemExit("peer was not discovered via presence")

        # RTT: responder endpoint answers pings; requester measures.
        responder = MeshEndpoint(BROKER, "mesh-responder", manifest_hash="host-manifest")
        pongs: list[float] = []
        got_ping = threading.Event()

        def on_ping(topic: str, envelope: dict) -> None:
            responder.publish("pong", {"ping_ts_ms": envelope["body"]["ts_ms"]},
                              target_pod="mesh-host")
            got_ping.set()

        responder.subscribe("np/mesh-responder/ping", on_ping)
        time.sleep(0.5)

        rtts = []
        for i in range(ROUNDS):
            got_ping.clear()
            start = time.perf_counter()
            endpoint.publish("ping", {"ts_ms": int(start * 1000), "seq": i},
                             target_pod="mesh-responder")
            while not got_ping.is_set() and time.perf_counter() - start < 2.0:
                time.sleep(0.0002)
            if got_ping.is_set():
                rtts.append((time.perf_counter() - start) * 1000)
            else:
                time.sleep(0.3)  # let a lost ping recover before the next
        return {"discovery": discovery, "rtts": rtts, "rounds_sent": ROUNDS,
                "endpoint_stats": endpoint.stats(),
                "responder_stats": responder.stats()}
    finally:
        if responder is not None:
            responder.close()
        endpoint.close()


def main() -> None:
    result = summarise(**collect())
    write_evidence(result, Path("research/runs/mesh-presence-20260920.json"),
                   __file__, subject=SUBJECT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
