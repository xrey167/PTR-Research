import sys, time
sys.path.insert(0, ".")
t0 = time.time()
def step(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)
step("import local_search")
from neural_pods.local_search import LocalSearchBackend
step("init backend")
backend = LocalSearchBackend("/tmp/redis-dbg.sqlite3")
step("create namespace")
backend.create_namespace("bench")
step("upsert")
backend.upsert("bench", "doc-0", text="lorem ipsum bench body zero")
step("first search")
hits = backend.search("bench", text="bench query warm 0", top_k=5)
step(f"search done: {len(hits)} hits")
from neural_pods.pod_cache import PodCache
import redis as redis_mod
step("redis connect")
r = redis_mod.Redis(host="10.50.0.121", socket_timeout=3)
step(f"ping: {r.ping()}")
cache = PodCache(redis_client=r)
step("cached search")
cache.search(backend, "bench", text="bench query warm 0", top_k=5)
step("done")
