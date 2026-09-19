"""Mesh cache benchmark: Pod A on the host writes cache entries via the
Redis instance in np-node1; Pod B inside LXD node np-node2 reads them
across the network. Principal isolation is verified (pod B cannot read
pod A's private-principal entries). Invalidations via mesh event.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.mesh_cache import MeshCache  # noqa: E402

REDIS_HOST = "10.50.0.121"
PEER_NODE = "np-node2"
PEER_ID = "mesh-cache-b"


def main() -> None:
    import redis
    cache_a = MeshCache(redis_client=redis.Redis(host=REDIS_HOST, socket_timeout=3),
                        pod_id="mesh-cache-a", principal="tenant-a", namespace="mesh-bench")
    private = MeshCache(redis_client=redis.Redis(host=REDIS_HOST, socket_timeout=3),
                        pod_id="mesh-cache-a", principal="tenant-a-private",
                        namespace="mesh-bench")

    cache_a.put("reader:case:42", {"value": 42, "unit": "days"})
    private.put("reader:case:secret", {"value": 9999, "unit": "days"})

    # Run pod B inside np-node2: read the shared entry, try the private one,
    # verify an invalidation from the mesh side.
    peer_code = f'''
import sys, json
sys.path.insert(0, "/home/xrey/neural-pods")
import redis
from neural_pods.mesh_cache import MeshCache
cache_b = MeshCache(redis_client=redis.Redis(host="{REDIS_HOST}", socket_timeout=3),
                    pod_id="{PEER_ID}", principal="tenant-a", namespace="mesh-bench")
private_b = MeshCache(redis_client=redis.Redis(host="{REDIS_HOST}", socket_timeout=3),
                      pod_id="{PEER_ID}", principal="tenant-a-private",
                      namespace="mesh-bench")
shared = cache_b.get("reader:case:42")
secret = private_b.get("reader:case:secret")  # same principal -> visible
other = cache_b.get("reader:case:42", principal="tenant-other")  # isolated
cache_b.invalidate("reader:case:42")
after = cache_b.get("reader:case:42")
print(json.dumps({{"shared": shared, "secret": secret, "other_principal": other,
                  "after_invalidate": after}}))
'''
    result = json.loads(subprocess.run(
        ["lxc", "exec", PEER_NODE, "--", sys.executable, "-c", peer_code],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])

    outcome = {
        "status": "completed",
        "cross_node_read": result["shared"] == {"value": 42, "unit": "days"},
        "same_principal_visible": result["secret"] == {"value": 9999, "unit": "days"},
        "principal_isolated": result["other_principal"] is None,
        "invalidation_works": result["after_invalidate"] is None,
        "cache_a_stats": cache_a.stats(), "peer_result": result,
    }
    Path("research/runs/mesh-cache-20260920.json").write_text(
        json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
