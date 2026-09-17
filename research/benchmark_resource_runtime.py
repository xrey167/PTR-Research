"""Small benchmark for Colibri-inspired hardware/request accounting."""
from __future__ import annotations
import concurrent.futures, json, tempfile, time
from neural_pods.resource_runtime import probe_hardware, ResourceBudget, ResourceGovernor, RequestTracker

snap = probe_hardware()
budget = ResourceBudget.from_snapshot(snap)
gov = ResourceGovernor(budget)
tracker = RequestTracker(max_records=10000)

def one(i):
    rid=f"r{i}"; tracker.start(rid, f"t{i}", "bench", "g1")
    tracker.phase(rid,"queue",0.01)
    lease=gov.choose(1024, preferred="vram")
    time.sleep(0.00005)
    if lease: gov.release(lease)
    tracker.finish(rid, success=True, input_tokens=8, output_tokens=16, cache_hit=(i%2==0))
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
    t=time.perf_counter(); list(ex.map(one, range(10000))); elapsed=time.perf_counter()-t
print(json.dumps({"elapsed_s":elapsed,"requests_per_s":10000/elapsed, "hardware":snap.to_dict(), "tracker":tracker.stats(), "governor":gov.stats()}, indent=2, sort_keys=True))
