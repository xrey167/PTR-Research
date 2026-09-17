import time,json,random,statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neural_pods.block_postings import BlockPostings,ClusterPostings
random.seed(1); ids=list(range(100000)); selected=[f'p{i}' for i in range(5000)]
b=BlockPostings(256); c=ClusterPostings()
for j,t in enumerate(selected):
 vals=sorted(random.sample(ids,100)); b.add(t,vals); c.add(t,[x%50 for x in vals],vals)
res={}
for name,obj in [('block',b),('cluster',c)]:
 times=[]
 for _ in range(20):
  s=time.perf_counter(); n=len(obj.union(selected)); times.append((time.perf_counter()-s)*1000)
 res[name]={'p50_ms':statistics.median(times),'p95_ms':sorted(times)[-2],'hits':n,'stats':obj.stats()}
out=Path('runs/fts-block-vs-cluster-latency-001.json');out.write_text(json.dumps({'schema':'fts-block-vs-cluster-latency:v1','terms':len(selected),'runs':20,'results':res},indent=2));print(json.dumps(res,indent=2))
