import json,time
from neural_pods.placement import PlacementDriver,FencedLeader
pd=PlacementDriver()
for i,z in enumerate(('a','b','c','d')): pd.register(f'n{i}',zone=z)
s=time.perf_counter()
for i in range(10000):
 p=pd.assign('ns',f'r{i}',replication_factor=3); pd.route('ns',f'r{i}',write=True)
elapsed=time.perf_counter()-s
l=FencedLeader(); lease=l.acquire('n0'); valid=l.validate('n0',lease.fence)
print(json.dumps({'regions':10000,'placement_ops_s':10000/elapsed,'unique_zones':len({pd.metadata()['nodes'][n]['zone'] for n in p.replicas}),'leader_fence_valid':valid,'epoch':p.epoch},indent=2))
