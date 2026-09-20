"""P3 perception stream benchmark: emit 2000 synthetic detector events over
the mesh (host endpoint -> consumer), measure throughput, lossless delivery
at bounded queue, and backpressure behavior when the queue overflows.

Neither stream sets `max_events_s`: this measures QUEUE backpressure, and
the rate limiter (opt-in since the pod audit, previously a parameter that
did nothing) would drop events for an unrelated reason and hide it.

Split into `collect()` (needs the broker) and `summarise()` (pure), like the
other gate-relevant benchmarks: the verdict is the part that was wrong twice
here, and it is the part that could not be tested while it only existed
inside a function that opens an MQTT connection.
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
#: The overflow probe: 50 events into a queue of 5, nothing draining.
OVERFLOW_ATTEMPTS = 50
TINY_QUEUE = 5


def summarise(*, count: int, delivered: bool, mesh_received: int,
              elapsed_s: float, stream_stats: dict, dropped: int,
              tiny_queued: int, tiny_rate_dropped: int,
              overflow_attempts: int = OVERFLOW_ATTEMPTS,
              queue_cap: int = TINY_QUEUE) -> dict:
    """Turn the observations into the evidence the gate reads. Pure.

    Every number here is derived from what ARRIVED, never from the number of
    events the run set out to send. `throughput_events_s` was `COUNT /
    elapsed`, which reports a run that delivered half its events at the full
    rate — the one figure a degraded run must not be allowed to keep.
    """
    lossless = bool(delivered) and mesh_received == count
    return {
        # "degraded" is the same distinction benchmark_native_tcp draws: a
        # partial run is not a smaller success, and the gate must be able to
        # tell the two apart from the status alone.
        "status": "completed" if lossless else "degraded",
        "events": count,
        "mesh_received": mesh_received,
        "delivery_completed": bool(delivered),
        "lossless": lossless,
        "throughput_events_s": round(mesh_received / max(elapsed_s, 1e-9), 1),
        "elapsed_s": round(elapsed_s, 2),
        "stream_stats": stream_stats,
        "backpressure": {
            "dropped": dropped,
            # What the queue cannot hold has to be dropped, not swallowed and
            # not blocked on: attempts minus capacity, both recorded.
            "no_deadlock": dropped == overflow_attempts - queue_cap,
            "overflow_attempts": overflow_attempts,
            "queue_cap": queue_cap,
            "queue_cap_respected": tiny_queued <= queue_cap,
            "rate_dropped": tiny_rate_dropped,
        },
    }


def collect(*, broker: str = BROKER, count: int = COUNT) -> dict:
    """Run the stream over the mesh and return the raw observations.

    This is the half that needs a broker. It measures and returns; it draws
    no conclusions.
    """
    endpoint = MeshEndpoint(broker, "perception-host", manifest_hash="p-manifest")
    stream = PerceptionStream(pod_id="vision-pod", max_queue=count)
    mesh_received = []
    mesh_done = threading.Event()

    endpoint.subscribe("np/vision-host/detections",
                       lambda t, e: (mesh_received.append(e["body"]), mesh_done.set()
                                     if len(mesh_received) >= count else None))
    time.sleep(0.5)

    # Stream -> mesh: every consumed event is published natively.
    stream.set_consumer(lambda event: endpoint.publish("detections", event,
                                                       target_pod="vision-host"))
    started = time.perf_counter()
    for seq in range(count):
        stream.emit(synthetic_detector_event(seq))
    # The return value decides the status. Discarding it meant a run that
    # timed out after delivering half its events was still recorded as
    # "completed" — and this evidence file has to be re-recorded on the
    # server, so the defect would have produced the replacement.
    delivered = mesh_done.wait(timeout=120.0)
    elapsed = time.perf_counter() - started
    # ONE snapshot, taken here. The count was read twice at two different
    # moments, so the delivered figure and the figure the status was computed
    # from could disagree by whatever arrived in between.
    received = len(mesh_received)

    # Backpressure: overflow a tiny queue, expect drops not deadlock.
    # No consumer on purpose: with nothing draining, the queue fills and the
    # overflow is deterministic. The drain thread used to pop events even
    # without a consumer, which made this race-dependent.
    tiny = PerceptionStream(pod_id="vision-tiny", max_queue=TINY_QUEUE)
    time.sleep(0.5)
    drop_result = [tiny.emit(synthetic_detector_event(i))
                   for i in range(OVERFLOW_ATTEMPTS)]
    observations = {
        "count": count,
        "delivered": delivered,
        "mesh_received": received,
        "elapsed_s": elapsed,
        "stream_stats": stream.stats(),
        "dropped": drop_result.count("dropped"),
        "tiny_queued": tiny.stats()["queued"],
        "tiny_rate_dropped": tiny.stats()["rate_dropped"],
    }
    endpoint.close()
    stream.close()
    tiny.close()
    return observations


def main() -> None:
    result = summarise(**collect())
    write_evidence(result, Path("research/runs/perception-20260920.json"),
                   __file__, subject=SUBJECT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
