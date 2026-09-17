import json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neural_pods.block_postings import BlockPostings,ClusterPostings
random.seed(42); docs=list(range(100000)); rows=[]
for rank in [1,10,100,1000,10000]:
 freq=max(50,100000//rank); ids=sorted(random.sample(docs,freq)); clusters=[i%max(1,freq//57) for i in range(freq)]
 b=BlockPostings(256);b.add('t',ids);c=ClusterPostings();c.add('t',clusters,ids)
 rows.append({'rank':rank,'postings':freq,'block':b.stats(),'cluster':c.stats(),'size_ratio_cluster_over_block':c.stats()['packed_bytes']/b.stats()['packed_bytes']})
print(json.dumps({'schema':'fts-partition-frequency-sweep:v1','rows':rows},indent=2)); open('runs/fts-partition-frequency-sweep-001.json','w').write(json.dumps({'schema':'fts-partition-frequency-sweep:v1','rows':rows},indent=2))

