"""N5 end-to-end run: two pods on different mesh nodes exchange dialect
frames — Pod A (host) publishes an answer as a native MQTT frame through
the NativeCommExecutor; Pod B (np-node2) receives, validates and
acknowledges it. What it measures is the MESH path and the executor: both
pods online via presence, every frame delivered and acknowledged, pod B's
validation, total latency.

WHAT IT DOES NOT MEASURE, contrary to what this docstring used to say. No
dialect LoRA is loaded here and no model produces a frame. "Driven by the
native communication model" was not true of this script: the frames are
built with an f-string over json.dumps, exactly like the sequential path in
benchmark_traced_pipeline.py, and the variable that held the natural-language
intent a model would have been given was assembled and then never used.

A model's ability to produce these frames is measured in
research/train_native_comm.py and recorded as `exact_rate` (0.55 against a
0.50 threshold) in native-comm-eval-20260920-report.json. Nothing in this
file may be read as adding to that number, which is why the report now
carries `frames_produced_by: "serialiser"`.

Gate `mesh_e2e` reads the mesh facts only: acks_received, pod_b_validated_all
and the executor's refusal count.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.mesh import MeshEndpoint  # noqa: E402
from neural_pods.native_comm import EgressACL, NativeCommExecutor  # noqa: E402
from research.evidence import write as write_evidence  # noqa: E402
from research.train_reader import file_sha, load_bundle  # noqa: E402

#: The modules these numbers are evidence ABOUT.
SUBJECT = [
    "neural_pods/mesh.py",
    "neural_pods/native_comm.py",
]

BROKER = "10.50.0.121"
PEER_NODE = "np-node2"
PEER_ID = "mesh-e2e-b"
ROUNDS = 20

PEER_CODE = '''
import sys, time, json
sys.path.insert(0, "/home/xrey/neural-pods")
from neural_pods.mesh import MeshEndpoint
endpoint = MeshEndpoint("10.50.0.121", "mesh-e2e-b", manifest_hash="peer-manifest")
answers = []
def on_answer(topic, envelope):
    body = envelope["body"]
    # Pod B validates natively: the answer must parse and carry a value.
    ok = isinstance(body, dict) and "value" in body and isinstance(body["value"], int)
    endpoint.publish("ack", {"validated": ok, "seq": body.get("seq")}, target_pod="mesh-e2e-a")
    answers.append(ok)
endpoint.subscribe("np/mesh-e2e-b/answer", on_answer)
deadline = time.time() + 60
while time.time() < deadline and len(answers) < {ROUNDS}:
    time.sleep(0.05)
endpoint.close()
print(json.dumps({{"validated": sum(answers), "total": len(answers)}}))
'''


def summarise(*, acks: dict, rounds: int, elapsed_s: float,
              executor_stats: dict, peer_report: dict) -> dict:
    """Turn the observed acknowledgements into evidence. Pure.

    `pod_b_validated_all` requires BOTH that every ack says validated and
    that there are as many acks as rounds: `all()` over an empty dict is
    True, so the count is what stops a run where nothing arrived from
    reporting perfect validation.
    """
    return {
        "status": "completed",
        "rounds": rounds,
        "acks_received": len(acks),
        "acks_missing": rounds - len(acks),
        "pod_b_validated_all": bool(
            len(acks) == rounds and rounds > 0
            and all(ack.get("validated") for ack in acks.values())),
        "elapsed_s": round(elapsed_s, 2),
        "executor_stats": executor_stats,
        # Stated in the evidence, so no reader has to go back to the source
        # to find out whether a model was involved. It was not.
        "frames_produced_by": "serialiser",
        "model_dialect_measured_in": "native-comm-eval-20260920-report.json",
        "peer_report": peer_report,
    }


def main() -> None:
    protocol, _config, _ = load_bundle(Path("runs/reader-training-inputs-generation7"))
    rows = json.loads((Path("runs/reader-training-inputs-generation7") / "inputs" / "test.json")
                      .read_text(encoding="utf-8"))[:ROUNDS]
    # Frozen Gen-7 eval answers = what the main model pod would speak.
    gen7 = json.loads((Path("runs/qwen3b-eval-test-gen7-20260920-report.json"))
                      .read_text(encoding="utf-8"))
    answers_by_id = {r["id"]: r["text"] for r in gen7["rows"]}

    endpoint = MeshEndpoint(BROKER, "mesh-e2e-a", manifest_hash="host-manifest")
    acl = EgressACL(topics=("np/mesh-e2e-b/#",), hosts=("10.50.0.121",), ports=(1883,))
    executor = NativeCommExecutor(mesh=endpoint, acl=acl)
    validated = []
    ack_events: dict[int, threading.Event] = {}
    acks: dict[int, dict] = {}

    def on_ack(topic: str, envelope: dict) -> None:
        body = envelope["body"]
        acks[body["seq"]] = body
        ack_events[body["seq"]].set()

    endpoint.subscribe("np/mesh-e2e-a/ack", on_ack)

    # Start pod B in np-node2.
    subprocess.run(["lxc", "exec", PEER_NODE, "--", "pkill", "-f", "mesh-e2e-b"],
                   capture_output=True)
    peer_code = PEER_CODE.replace("{ROUNDS}", str(ROUNDS))
    peer_cmd = (f"nohup {sys.executable} -c '{peer_code}' > /tmp/mesh-e2e-b.log 2>&1 &")
    subprocess.run(["lxc", "exec", PEER_NODE, "--", "bash", "-c", peer_cmd], check=True)

    started = time.perf_counter()
    try:
        for i, row in enumerate(rows):
            ack_events[i] = threading.Event()
            answer = answers_by_id.get(row["id"], "")
            # Pod A speaks the answer natively: intent -> dialect LoRA would
            # emit the frame; the executor path is identical to the trained
            # dialect (frame parsed, ACL-checked, published via MQTT).
            import re
            value_match = re.search(r"\b(\d+)\s*(?:days|tage)?", answer, re.IGNORECASE)
            value = int(value_match.group(1)) if value_match else -1
            # The natural-language intent a dialect model would receive lived
            # here, was assembled, and was never passed to anything. Removed
            # rather than left standing: an unused variable shaped like the
            # missing half of the experiment reads as if that half existed.
            frame = (f'PUB np/mesh-e2e-b/answer '
                     f'{json.dumps({"value": value, "seq": i, "case": row["id"]}, separators=(",", ":"))}')
            assert frame.startswith("PUB ")  # dialect shape identical to training
            result = executor.execute(frame)
            if result.refused or result.invalid:
                continue
            ack_events[i].wait(timeout=5.0)
        # Retry lost answers (app-level reliability over QoS-0 style mesh):
        import re
        for _round in range(3):
            for i, row in enumerate(rows):
                if ack_events[i].is_set():
                    continue
                answer = answers_by_id.get(row["id"], "")
                value_match = re.search(r"\b(\d+)\s*(?:days|tage)?", answer, re.IGNORECASE)
                value = int(value_match.group(1)) if value_match else -1
                frame = (f'PUB np/mesh-e2e-b/answer '
                         f'{json.dumps({"value": value, "seq": i, "case": row["id"]}, separators=(",", ":"))}')
                executor.execute(frame)
            if all(e.is_set() for e in ack_events.values()):
                break
            time.sleep(1.0)
        elapsed = time.perf_counter() - started

        # Pod B's own validation report (peer exits after ROUNDS acks):
        deadline = time.perf_counter() + 45.0
        peer_report = {}
        while time.perf_counter() < deadline:
            peer_out = subprocess.run(
                ["lxc", "exec", PEER_NODE, "--", "cat", "/tmp/mesh-e2e-b.log"],
                capture_output=True, text=True).stdout.strip().splitlines()
            if peer_out and peer_out[-1].startswith("{"):
                peer_report = json.loads(peer_out[-1])
                break
            time.sleep(1.0)

        result = summarise(acks=acks, rounds=ROUNDS, elapsed_s=elapsed,
                           executor_stats=executor.stats() | {
                               "refused": executor.acl.violations},
                           peer_report=peer_report)
        write_evidence(result, Path("research/runs/mesh-e2e-20260920.json"),
                       __file__, subject=SUBJECT)
        print(json.dumps(result, indent=2))
    finally:
        endpoint.close()


if __name__ == "__main__":
    main()
