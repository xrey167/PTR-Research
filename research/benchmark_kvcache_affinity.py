"""Does session affinity actually buy anything? Measure TTFT, don't assume.

ADR-3 chose KV-cache session affinity over KV transfer, following Mooncake's
KVCache-centric design, and PodStorage.kv_session implements it. What was
never measured is the thing the ADR claims: that a session returning to the
replica holding its warm prefix starts generating sooner. The affinity
object only counted its own reuses and failovers, which says how often the
mapping held, not what holding it was worth.

This benchmark issues repeated turns per session against two vLLM replicas,
once with affinity and once round-robin, and compares time-to-first-token —
the part of latency a warm prefix can change. Everything after the first
token is decode and is not what affinity is about.

The routing and the statistics are pure and unit-tested
(tests/test_kvcache_affinity.py); only `measure()` needs the replicas.

Usage:
    research/run_vllm_replicas.sh start
    python research/benchmark_kvcache_affinity.py \\
        --replica http://127.0.0.1:18000 --replica http://127.0.0.1:18001
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.storage import SessionAffinity  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs" / "kvcache-affinity-20260920.json"


class Router:
    """Picks a replica per turn, with or without session affinity.

    The control arm has to be an actual control. A global round-robin counter
    is not: with the session loop inside the turn loop, an even number of
    sessions over two replicas puts every session back on the same replica
    every turn, so both arms route identically and the comparison measures
    nothing. The control therefore rotates PER SESSION — a session's
    consecutive turns land on different replicas, which is exactly the warm
    prefix that affinity is supposed to preserve.
    """

    def __init__(self, replicas: list[str], *, affinity: bool):
        if not replicas:
            raise ValueError("need at least one replica")
        self.replicas = list(replicas)
        self.affinity = SessionAffinity() if affinity else None
        self._home: dict[str, int] = {}      # session -> its first replica
        self._visits: dict[str, int] = {}    # session -> turns taken so far

    def route(self, session_id: str) -> str:
        if session_id not in self._home:
            # Sessions still spread across replicas; both arms start a session
            # in the same place, so the first turn is cold either way.
            self._home[session_id] = len(self._home) % len(self.replicas)
        visit = self._visits.get(session_id, 0)
        self._visits[session_id] = visit + 1
        home = self._home[session_id]
        if self.affinity is not None:
            # A session sticks to the replica that already holds its prefix.
            return self.affinity.bind(session_id, self.replicas[home])
        return self.replicas[(home + visit) % len(self.replicas)]

    def stats(self) -> dict:
        return (self.affinity.stats() if self.affinity is not None
                else {"sessions": None, "reuses": 0, "failovers": 0})


def summarize(name: str, ttft_ms: list[float], router: Router) -> dict:
    ordered = sorted(ttft_ms)
    return {
        "mode": name,
        "turns": len(ordered),
        "ttft_p50_ms": round(statistics.median(ordered), 3) if ordered else None,
        "ttft_p95_ms": (round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3)
                        if ordered else None),
        "ttft_mean_ms": round(statistics.fmean(ordered), 3) if ordered else None,
        "affinity": router.stats(),
    }


def compare(with_affinity: dict, round_robin: dict) -> dict:
    """What affinity was worth, stated as a ratio and a signed delta."""
    a, b = with_affinity.get("ttft_p50_ms"), round_robin.get("ttft_p50_ms")
    if not a or not b:
        return {"ttft_p50_speedup": None, "ttft_p50_delta_ms": None,
                "affinity_helps": None}
    return {"ttft_p50_speedup": round(b / a, 3),
            "ttft_p50_delta_ms": round(b - a, 3),
            "affinity_helps": a < b}


def measure(replicas: list[str], *, sessions: int, turns: int, affinity: bool,
            model: str, prefix_tokens: int) -> tuple[list[float], Router]:
    """One pass over every session's turns; returns per-turn TTFT in ms."""
    import urllib.request

    router = Router(replicas, affinity=affinity)
    ttft_ms: list[float] = []
    prefix = "Context line. " * prefix_tokens
    for turn in range(turns):
        for index in range(sessions):
            session_id = f"session-{index}"
            replica = router.route(session_id)
            payload = json.dumps({
                "model": model, "stream": True, "max_tokens": 1,
                "messages": [{"role": "user",
                              "content": f"{prefix}\nTurn {turn}: reply with OK."}],
            }).encode("utf-8")
            request = urllib.request.Request(
                f"{replica}/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=60) as response:
                for line in response:                # first streamed chunk
                    if line.strip() and line.startswith(b"data:"):
                        break
            ttft_ms.append((time.perf_counter() - started) * 1000)
    return ttft_ms, router


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", action="append", required=True,
                        help="replica base URL; pass twice for two replicas")
    parser.add_argument("--model", default="NeoHorse-1-4B")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    args = parser.parse_args()

    runs = []
    for affinity in (True, False):
        ttft, router = measure(args.replica, sessions=args.sessions,
                               turns=args.turns, affinity=affinity,
                               model=args.model, prefix_tokens=args.prefix_tokens)
        runs.append(summarize("affinity" if affinity else "round_robin", ttft, router))

    result = {"status": "completed", "replicas": args.replica, "model": args.model,
              "sessions": args.sessions, "turns": args.turns,
              "prefix_tokens": args.prefix_tokens,
              "runs": runs, "comparison": compare(runs[0], runs[1]),
              "scope": ("time-to-first-token only; affinity cannot change decode "
                        "speed, and this says nothing about KV transfer, which "
                        "ADR-3 declined for different reasons")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
