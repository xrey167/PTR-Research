import concurrent.futures,json,time
from neural_pods.resource_runtime import ResourceBudget,ResourceGovernor

g=ResourceGovernor(ResourceBudget(ram_bytes=10**12,vram_bytes={},disk_bytes=None))
def one(i):
 r=g.activate(f'pod-{i}', 'g1', 1024, preferred='ram')
 ok=r is not None
 if ok: g.deactivate(f'pod-{i}','g1')
 return ok
s=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex: vals=list(ex.map(one,range(10000)))
e=time.perf_counter()
print(json.dumps({'requests':len(vals),'qps':len(vals)/(e-s),'activated':sum(vals),'stats':g.stats()},indent=2))
