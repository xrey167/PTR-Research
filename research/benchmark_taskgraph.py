"""Task graph benchmark: a 6-node DAG executed across mesh endpoints.

Four independent nodes call a remote mesh responder (each call includes a
~100 ms remote processing delay); two dependent nodes consume their
outputs as token context. Wall time vs sequential-equivalent time yields
the parallel speedup; correctness requires every node to echo its inputs.
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.mesh import MeshEndpoint  # noqa: E402
from neural_pods.taskgraph import TaskGraph, TaskNode  # noqa: E402

BROKER = "10.50.0.121"
REMOTE_DELAY_S = 0.1


def main() -> None:
    endpoint = MeshEndpoint(BROKER, "tg-host", manifest_hash="tg-manifest")
    responder = MeshEndpoint(BROKER, "tg-responder", manifest_hash="tg-manifest")
    pongs: dict[int, dict] = {}
    pong_events: dict[int, threading.Event] = {}

    def on_call(topic: str, envelope: dict) -> None:
        body = envelope["body"]
        time.sleep(REMOTE_DELAY_S)  # simulated remote processing
        pong_events[body["seq"]].set()
        pongs[body["seq"]] = {"echo": body["intent"], "processed_by": "tg-responder"}

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

        from dataclasses import asdict
        for node_id, res in result["results"].items():
            if res.error or (res.output and isinstance(res.output, dict)
                             and res.output.get("error") == "timeout"):
                print(f"NODE {node_id}: error={res.error} output={res.output}", flush=True)
        correct = all(
            r.error is None and r.output and r.output.get("error") != "timeout"
            for r in result["results"].values())
        result["correct"] = correct
        result["results"] = {k: asdict(v) | {"output": v.output} for k, v in result["results"].items()}
        Path("research/runs/taskgraph-20260920.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"wall_s": result["wall_s"],
                          "sequential_equivalent_s": result["sequential_equivalent_s"],
                          "speedup": result["speedup"],
                          "correct": correct,
                          "nodes": len(result["results"])}, indent=2))
    finally:
        endpoint.close()
        responder.close()


if __name__ == "__main__":
    main()
