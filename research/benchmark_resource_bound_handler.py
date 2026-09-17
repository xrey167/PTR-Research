import concurrent.futures, time, json
from neural_pods.resource_runtime import ResourceBudget, ResourceGovernor, ResourceBoundHandler

g = ResourceGovernor(ResourceBudget(ram_bytes=10**12, vram_bytes={}, disk_bytes=None))
h = ResourceBoundHandler(lambda p,r: p, g, amount_bytes=1024, preferred='ram')
def one(i): return h(i,None)
t=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex: vals=list(ex.map(one,range(10000)))
elapsed=time.perf_counter()-t
print(json.dumps({'requests':len(vals),'requests_per_s':len(vals)/elapsed,'stats':h.stats(),'used':g.stats()['used']}))
