"""Optional in-process raft-rs binding.

The extension is deliberately optional: Python placement, leases and WAL
remain usable when Rust is unavailable.  When installed, the binding removes
the Python/RPC hop for the consensus state machine; durable Ready handling
still belongs to the control plane.
"""
from __future__ import annotations
import base64
from typing import Any

try:  # pragma: no cover - exercised on the Linux GPU host
    from neural_pods_raft import RaftNode
except ImportError:  # pragma: no cover
    RaftNode = None  # type: ignore[assignment,misc]


def raft_binding_available() -> bool:
    return RaftNode is not None


class DurableRaftNode:
    """Small persistence gate around the native binding.

    The native node exposes a Ready tuple, while this class writes that tuple
    to the configured WAL before acknowledging it.  `SnapshotStore` is kept
    duck-typed so PostgreSQL/object-storage implementations can be plugged in.
    """

    def __init__(self, node_id: int, namespace: str, wal: Any):
        if RaftNode is None:
            raise RuntimeError("neural_pods_raft extension is not installed")
        self.node = RaftNode(node_id)
        self.namespace = namespace
        self.wal = wal

    @staticmethod
    def _event(ready: tuple) -> dict[str, Any]:
        entries, hard_state, committed = ready
        return {
            "kind": "raft_ready",
            "entries": [
                {"index": i, "term": t, "data_b64": base64.b64encode(data).decode("ascii")}
                for i, t, data in entries
            ],
            "hard_state": None if hard_state is None else {
                "term": hard_state[0], "vote": hard_state[1], "commit": hard_state[2]
            },
            "committed": [
                {"index": i, "term": t, "data_b64": base64.b64encode(data).decode("ascii")}
                for i, t, data in committed
            ],
        }

    def propose(self, data: bytes) -> list[bytes]:
        ready = self.node.propose_pending(data)
        if ready is None:
            return []
        self.wal.append_wal(self.namespace, self._event(ready))
        return list(self.node.ack_ready())

    def pending_messages(self) -> list[bytes]:
        return list(self.node.pending_messages())
