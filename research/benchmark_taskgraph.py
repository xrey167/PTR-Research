"""Task graph benchmark: a 6-node DAG executed across mesh endpoints.

Four independent nodes call a remote mesh responder (each call includes a
~100 ms remote processing delay); two dependent nodes consume their
outputs as token context. Correctness requires every node to echo its
inputs.

The responder handles each call on its OWN thread. paho delivers messages
on a single network loop thread and MeshEndpoint invokes handlers inline,
so sleeping in the handler serialises every call — the earlier version of
this benchmark did exactly that, which made the four "parallel" ingest
nodes finish 100 ms apart and inflated the reported ratio while nothing
ran concurrently.

Reported metrics: `mean_concurrency` (average in-flight nodes) and
`critical_path_ratio` (wall time against the DAG's lower bound). Neither
is a speedup — see neural_pods/taskgraph.py.

STRUCTURE. `collect()` needs the broker; `summarise()` does not. The verdict
this file feeds the gate — `correct`, `wall_within_bound` — is arithmetic
over observations, and arithmetic that only ever runs on a machine with an
MQTT broker is arithmetic nobody checks. That is how the 2.59x "speedup"
survived: it was a plausible number computed in a place no test could reach.
`summarise()` is pure and is exercised in tests/test_benchmark_taskgraph.py
against both a parallel and a serialised observation set, so the verdict is
shown to FLIP rather than merely to come out green once.
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.mesh import MeshEndpoint  # noqa: E402
from neural_pods.taskgraph import TaskGraph, TaskNode  # noqa: E402
from research.evidence import write as write_evidence  # noqa: E402

#: The modules these numbers are evidence ABOUT. research/evidence.py
#: hashes them into the report, and the gate refuses the file once any
#: of them changes: a measurement of code that no longer exists is not
#: evidence, however carefully it was recorded.
SUBJECT = [
    "neural_pods/taskgraph.py",
    "neural_pods/mesh.py",
]

BROKER = "10.50.0.121"
REMOTE_DELAY_S = 0.1


def summarise(run: dict, *, remote_delay_s: float = REMOTE_DELAY_S) -> dict:
    """Turn one TaskGraph run into the evidence the gate reads.

    Pure: takes the run's observations, returns the report. `run["results"]`
    may hold TaskResult dataclasses (a real run) or plain dicts (a recorded
    or constructed one); both are read the same way, because a summary that
    only accepts live objects cannot be tested against a known answer.
    """
    from dataclasses import asdict, is_dataclass

    def field(result, name):
        if is_dataclass(result):
            return getattr(result, name)
        return result.get(name)

    results = run["results"]
    # A node is correct when it neither raised nor timed out. The timeout is
    # reported INSIDE the output (the remote call returns a marker rather
    # than raising), so checking `error` alone would count it as a success.
    correct = all(
        field(r, "error") is None
        and field(r, "output")
        and not (isinstance(field(r, "output"), dict)
                 and field(r, "output").get("error") == "timeout")
        for r in results.values())

    # The DAG is two levels deep, so the delay this benchmark injects puts a
    # hard floor of 2 x remote_delay_s on the wall time. Real parallelism
    # lands near that floor; a responder that answers one call at a time
    # needs 6 x remote_delay_s and misses it. This is the check that actually
    # distinguishes the two — a ratio built from node durations does not,
    # because queueing inflates those durations.
    bound = round(2 * remote_delay_s, 3)
    report = dict(run)
    report["correct"] = correct
    report["nodes"] = len(results)
    report["dag_delay_bound_s"] = bound
    report["wall_within_bound"] = run["wall_s"] <= 2.0 * bound
    report["results"] = {
        node_id: (asdict(r) | {"output": r.output}) if is_dataclass(r) else dict(r)
        for node_id, r in results.items()}
    return report


def collect(*, broker: str = BROKER) -> dict:
    """Run the DAG across the mesh and return the raw TaskGraph result.

    This is the half that needs a broker. It measures and returns; it draws
    no conclusions.
    """
    endpoint = MeshEndpoint(broker, "tg-host", manifest_hash="tg-manifest")
    responder = MeshEndpoint(broker, "tg-responder", manifest_hash="tg-manifest")
    pongs: dict[int, dict] = {}
    pong_events: dict[int, threading.Event] = {}

    def _process(seq: int, intent: str) -> None:
        time.sleep(REMOTE_DELAY_S)  # simulated remote processing
        pongs[seq] = {"echo": intent, "processed_by": "tg-responder"}
        pong_events[seq].set()

    def on_call(topic: str, envelope: dict) -> None:
        # Off the paho loop thread: sleeping here would serialise every call.
        body = envelope["body"]
        threading.Thread(target=_process, args=(body["seq"], body["intent"]),
                         daemon=True).start()

    responder.subscribe("np/tg-responder/call", on_call)

    seq_lock = threading.Lock()

    def remote_call(intent: str):
        with seq_lock:
            seq = len(pong_events)
            pong_events[seq] = threading.Event()
        endpoint.publish("call", {"seq": seq, "intent": intent},
                         target_pod="tg-responder")
        pong_events[seq].wait(timeout=5.0)
        return pongs.get(seq, {"error": "timeout"})

    try:
        nodes = [
            TaskNode("ingest_a", lambda p: remote_call("ingest-a")),
            TaskNode("ingest_b", lambda p: remote_call("ingest-b")),
            TaskNode("ingest_c", lambda p: remote_call("ingest-c")),
            TaskNode("ingest_d", lambda p: remote_call("ingest-d")),
            TaskNode("plan", lambda p: remote_call(
                         f"plan:{p['context']['ingest_a']['echo']}+{p['context']['ingest_b']['echo']}"),
                     depends_on=("ingest_a", "ingest_b"), context_key="context"),
            TaskNode("verify", lambda p: remote_call(
                         f"verify:{p['context']['ingest_c']['echo']}+{p['context']['ingest_d']['echo']}"),
                     depends_on=("ingest_c", "ingest_d"), context_key="context"),
        ]
        graph = TaskGraph(nodes, max_parallel=6)
        result = graph.run({"task": "mesh-dag"})
        for node_id, res in result["results"].items():
            if res.error or (res.output and isinstance(res.output, dict)
                             and res.output.get("error") == "timeout"):
                print(f"NODE {node_id}: error={res.error} output={res.output}",
                      flush=True)
        return result
    finally:
        endpoint.close()
        responder.close()


def main() -> None:
    report = summarise(collect())
    write_evidence(report, Path("research/runs/taskgraph-20260920.json"),
                   __file__, subject=SUBJECT, default=str)
    print(json.dumps({key: report[key] for key in (
        "wall_s", "dag_delay_bound_s", "wall_within_bound",
        "node_elapsed_sum_s", "critical_path_s", "mean_concurrency",
        "critical_path_ratio", "correct", "nodes")}, indent=2))


if __name__ == "__main__":
    main()
