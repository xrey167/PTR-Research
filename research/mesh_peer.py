"""Mesh peer process: runs inside an LXD node, announces itself and answers
pings on its own channel. Used by benchmark_mesh.py as the far endpoint.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/xrey/neural-pods")
from neural_pods.mesh import MeshEndpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="10.50.0.121")
    parser.add_argument("--pod-id", default="mesh-node2")
    parser.add_argument("--lifetime-s", type=float, default=60.0)
    args = parser.parse_args()

    endpoint = MeshEndpoint(args.broker, args.pod_id, manifest_hash="peer-manifest")
    pings = []

    def on_ping(topic: str, envelope: dict) -> None:
        try:
            pings.append(time.perf_counter())
            endpoint.publish("pong", {"ping_ts_ms": envelope["body"]["ts_ms"],
                                      "peer": args.pod_id},
                             target_pod=envelope["body"]["reply_to"])
        except Exception as error:
            print(f"ping handler error: {error!r}", flush=True)

    # Subscribe BEFORE the (re-)announce so pings cannot outrun the
    # subscription; the constructor announce happened before imports.
    endpoint.subscribe(f"np/{args.pod_id}/ping", on_ping)
    endpoint.announce()
    # Non-retained live signal: retained would be overwritten by the
    # heartbeat's "online" before the benchmark necessarily sees it.
    endpoint._publish_raw(f"np/presence/{args.pod_id}",
                          {"pod_id": args.pod_id, "state": "ready",
                           "principal": "local",
                           "manifest_hash": "peer-manifest",
                           "protocol_version": 1}, retain=False)
    print("peer ready", flush=True)
    deadline = time.time() + args.lifetime_s
    while time.time() < deadline:
        time.sleep(0.2)
    endpoint.close()
    print(f"answered {len(pings)} pings")


if __name__ == "__main__":
    main()
