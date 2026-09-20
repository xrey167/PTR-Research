"""Mesh cache benchmark: Pod A on the host writes cache entries via the
Redis instance in np-node1; Pod B inside LXD node np-node2 reads them
across the network, and an invalidation is verified across the node
boundary.

WHAT CHANGED, AND WHY. The previous version recorded
`principal_isolated: true` from this step:

    cache_b.get("reader:case:42", principal="tenant-other")   # -> None

Nothing had ever been written under `tenant-other`, so the read was a miss
whether or not the cache isolates anything — the assertion was true by
construction. Worse, the same script proved the opposite two lines earlier:
it constructed `private_b` for `tenant-a-private` inside np-node2 and read
that principal's entry straight out, because MeshCache derives the key from
the principal the caller names.

So this benchmark now measures the two statements separately:

  * `cross_principal_refused_by_client` — a MeshCache built for one
    principal refuses to serve another (client-side check, `PrincipalRefused`).
  * `cross_principal_reachable_via_redis` — the very same entry is readable
    with a raw Redis GET from the peer node, because every principal shares
    one database behind one credential.

The second one is expected to be TRUE. That is the honest finding: the
principal scope is key derivation, not a security boundary. Recording it is
the point — an isolation claim nobody can falsify is worth nothing.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.mesh_cache import MeshCache  # noqa: E402
from research.evidence import write as write_evidence  # noqa: E402

REDIS_HOST = "10.50.0.121"
PEER_NODE = "np-node2"
PEER_ID = "mesh-cache-b"
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "research" / "runs" / "mesh-cache-20260920.json"


SHARED_VALUE = {"value": 42, "unit": "days"}


def summarise(peer: dict, *, cache_a_stats: dict | None = None) -> dict:
    """Turn the peer node's observations into evidence. Pure.

    `principal_isolated` is deliberately absent. It used to be produced by
    reading a principal nothing had ever been written under, so it was true
    whatever the cache did; the two statements that replace it are checkable
    in opposite directions, and one of them is expected to be True because
    the isolation does NOT hold.
    """
    return {
        "status": "completed",
        "cross_node_read": peer.get("shared") == SHARED_VALUE,
        "invalidation_works": peer.get("after_invalidate") is None,
        # The two halves of what "principal isolation" used to assert.
        "cross_principal_refused_by_client": bool(peer.get("cross_principal_refused")),
        "cross_principal_reachable_via_redis": bool(
            peer.get("cross_principal_reachable_via_redis")),
        "isolation": "client-side key derivation",
        "isolation_note": (
            "A MeshCache refuses principals it was not built for. The entries "
            "themselves are not isolated: one Redis database, one credential, "
            "so the peer node reads another principal's key with a raw GET. "
            "Real isolation needs Redis ACLs per principal."),
        "cache_a_stats": cache_a_stats or {},
        "peer_result": peer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-host", default=REDIS_HOST)
    parser.add_argument("--peer-node", default=PEER_NODE)
    args = parser.parse_args()

    import redis
    cache_a = MeshCache(redis_client=redis.Redis(host=args.redis_host, socket_timeout=3),
                        pod_id="mesh-cache-a", principal="tenant-a", namespace="mesh-bench")
    private = MeshCache(redis_client=redis.Redis(host=args.redis_host, socket_timeout=3),
                        pod_id="mesh-cache-a", principal="tenant-a-private",
                        namespace="mesh-bench")

    cache_a.put("reader:case:42", {"value": 42, "unit": "days"})
    private.put("reader:case:secret", {"value": 9999, "unit": "days"})
    # The exact Redis key the private entry lives under. Handing it to the
    # peer is what makes the raw-GET probe below a real test instead of a
    # guess about the key layout.
    private_key = private._key("reader:case:secret", "tenant-a-private")

    peer_code = f'''
import sys, json
sys.path.insert(0, "/home/xrey/neural-pods")
import redis
from neural_pods.mesh_cache import MeshCache, PrincipalRefused

#: The modules these numbers are evidence ABOUT. research/evidence.py
#: hashes them into the report, and the gate refuses the file once any
#: of them changes: a measurement of code that no longer exists is not
#: evidence, however carefully it was recorded.
SUBJECT = [
    "neural_pods/mesh_cache.py",
]
client = redis.Redis(host="{args.redis_host}", socket_timeout=3)
cache_b = MeshCache(redis_client=client, pod_id="{PEER_ID}",
                    principal="tenant-a", namespace="mesh-bench")
shared = cache_b.get("reader:case:42")

# (1) client-side check: this instance serves tenant-a only
try:
    cache_b.get("reader:case:secret", principal="tenant-a-private")
    refused = False
except PrincipalRefused:
    refused = True

# (2) the same entry, straight out of Redis with the credential this node
#     already holds. No MeshCache involved, no principal asked for.
raw = client.get({private_key!r})
reachable = raw is not None
leaked = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)["value"] if raw else None

cache_b.invalidate("reader:case:42")
after = cache_b.get("reader:case:42")
print(json.dumps({{"shared": shared, "cross_principal_refused": refused,
                   "cross_principal_reachable_via_redis": reachable,
                   "cross_principal_value": leaked,
                   "after_invalidate": after, "peer_stats": cache_b.stats()}}))
'''
    result = json.loads(subprocess.run(
        ["lxc", "exec", args.peer_node, "--", sys.executable, "-c", peer_code],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])

    outcome = summarise(result, cache_a_stats=cache_a.stats())
    write_evidence(outcome, OUT, __file__, subject=SUBJECT)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
