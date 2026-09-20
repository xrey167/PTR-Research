"""P3 perception stream benchmark: emit 2000 synthetic detector events over
the mesh (host endpoint -> consumer), measure throughput, lossless delivery
at bounded queue, and backpressure behavior when the queue overflows.

Neither stream sets `max_events_s`: this measures QUEUE backpressure, and
the rate limiter (opt-in since the pod audit, previously a parameter that
did nothing) would drop events for an unrelated reason and hide it.
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.mesh import MeshEndpoint  # noqa: E402
from neural_pods.perception import PerceptionStream, synthetic_detector_event  # noqa: E402
from research.evidence import write as write_evidence  # noqa: E402

#: The modules these numbers are evidence ABOUT. research/evidence.py
#: hashes them into the report, and the gate refuses the file once any
#: of them changes: a measurement of code that no longer exists is not
#: evidence, however carefully it was recorded.
SUBJECT = [
    "neural_pods/perception.py",
    "neural_pods/mesh.py",
]

BROKER = "10.50.0.121"
COUNT = 2000


def main() -> None:
    endpoint = MeshEndpoint(BROKER, "perception-host", manifest_hash="p-manifest")
    stream = PerceptionStream(pod_id="vision-pod", max_queue=COUNT)
    mesh_received = []
    mesh_done = threading.Event()

    endpoint.subscribe("np/vision-host/detections",
                       lambda t, e: (mesh_received.append(e["body"]), mesh_done.set()
                                     if len(mesh_received) >= COUNT else None))
    time.sleep(0.5)

    # Stream -> mesh: every consumed event is published natively.
    stream.set_consumer(lambda event: endpoint.publish("detections", event,
                                                       target_pod="vision-host"))
    started = time.perf_counter()
    for seq in range(COUNT):
        stream.emit(synthetic_detector_event(seq))
    # The return value decides the status. Discarding it meant a run that
    # timed out after delivering half its events was still recorded as
    # "completed" — and this evidence file has to be re-recorded on the
    # server, so the defect would have produced the replacement.
    delivered = mesh_done.wait(timeout=120.0)
    elapsed = time.perf_counter() - started

    # Backpressure: overflow a tiny queue, expect drops not deadlock.
    # No consumer on purpose: with nothing draining, the queue fills and the
    # overflow is deterministic. The drain thread used to pop events even
    # without a consumer, which made this race-dependent.
    tiny = PerceptionStream(pod_id="vision-tiny", max_queue=5)
    time.sleep(0.5)
    drop_result = [tiny.emit(synthetic_detector_event(i)) for i in range(50)]
    dropped = drop_result.count("dropped")

    lossless = bool(delivered) and len(mesh_received) == COUNT
    result = {
        # "degraded" is the same distinction benchmark_native_tcp draws: a
        # partial run is not a smaller success, and the gate must be able to
        # tell the two apart from the status alone.
        "status": "completed" if lossless else "degraded",
        "events": COUNT, "mesh_received": len(mesh_received),
        "delivery_completed": bool(delivered),
        "lossless": lossless,
        "throughput_events_s": round(COUNT / max(elapsed, 1e-9), 1),
        "elapsed_s": round(elapsed, 2),
        "stream_stats": stream.stats(),
        "backpressure": {"dropped": dropped, "no_deadlock": dropped == 45,
                         "queue_cap_respected": tiny.stats()["queued"] <= 5,
                         "rate_dropped": tiny.stats()["rate_dropped"]},
    }
    write_evidence(result, Path("research/runs/perception-20260920.json"), __file__, subject=SUBJECT)
    print(json.dumps(result, indent=2))
    endpoint.close()
    stream.close()
    tiny.close()


if __name__ == "__main__":
    main()
