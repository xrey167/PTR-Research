import json,time,statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neural_pods.local_search import LocalSearchBackend
out=Path('runs/permission-filter-benchmark-001');out.mkdir(exist_ok=True); b=LocalSearchBackend(out/'search.sqlite3'); b.create_namespace('perm')
for i in range(20000): b.upsert('perm',str(i),text=f'document {i} bm25 procurement',metadata={'permissions':[f'p{i%10000}',f'p{(i+1)%10000}'],'status':'active','acl':['*']})
filt={'permissions':{'contains': [f'p{x}' for x in range(0,10000,2)]}}
t=[]
for _ in range(5):
 s=time.perf_counter(); h=b.search('perm',text='procurement',filters=filt,principal='*',top_k=10); t.append((time.perf_counter()-s)*1000)
report={'rows':20000,'permission_values':5000,'results':len(h),'cold_or_warm_ms':t,'p50_ms':statistics.median(t),'filter_cache_hits':b.filter_cache_hits,'filter_cache_misses':b.filter_cache_misses}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));b.close()

