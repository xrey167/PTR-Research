"""Three-node raft-rs/PyO3 loopback benchmark."""
from __future__ import annotations
import json, time
from neural_pods_raft import RaftNode


def run(count: int = 100) -> dict:
    nodes = {i: RaftNode(i, [1, 2, 3]) for i in (1, 2, 3)}
    queue: list[tuple[int, bytes]] = []
    applied = {i: [] for i in nodes}

    def emit(src: int) -> None:
        for target, msg in zip(nodes[src].pending_message_targets(), nodes[src].pending_messages()):
            queue.append((target, msg))
        applied[src].extend(nodes[src].ack_ready())

    def drain() -> int:
        steps = 0
        while queue:
            target, payload = queue.pop(0)
            out = nodes[target].step_message(payload)
            steps += 1
            if out is not None:
                emit(target)
        return steps

    nodes[1].campaign_pending(); emit(1); election_steps = drain()
    started = time.perf_counter()
    for i in range(count):
        nodes[1].propose_pending(f"entry-{i}".encode())
        emit(1); drain()
    elapsed = time.perf_counter() - started
    return {"nodes": 3, "proposals": count, "election_steps": election_steps,
            "elapsed_s": elapsed, "proposals_per_s": count / max(elapsed, 1e-9),
            "applied": {str(k): len(v) for k, v in applied.items()},
            "status": {str(k): n.status() for k, n in nodes.items()}}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))

