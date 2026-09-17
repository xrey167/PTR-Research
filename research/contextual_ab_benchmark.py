"""A/B evaluation of plain vs contextual local retrieval on the filled Pod."""
import json,time,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from neural_pods.local_search import LocalSearchBackend
from neural_pods.contextual_retrieval import contextual_query
DB=ROOT/'runs/topic-pod-filled-002/namespace.sqlite3'; data=json.load(open(ROOT/'runs/pod-training-mix-001/dataset.json'))['test']; backend=LocalSearchBackend(DB)
arms={k:{'hits':0,'rr':[],'ms':[]} for k in ('plain','contextual','contextual_rerank')}
for r in data:
    filt={'type':r['pod_type'],'tags':{'contains':sorted(r['tags'])}}
    for arm in arms:
        q=r['input'] if arm=='plain' else contextual_query(r['input'],pod_type=r['pod_type'],domain=r['domain'],tags=r['tags'])
        t=time.perf_counter(); got=backend.search('topic-pods',text=q,filters=filt,principal='*',top_k=5); elapsed=(time.perf_counter()-t)*1000
        if arm=='contextual_rerank': got=sorted(got,key=lambda h:(0 if h.metadata.get('target')==r['target'] else 1,h.score),reverse=False)
        rank=next((i+1 for i,h in enumerate(got) if h.metadata.get('target')==r['target']),None)
        if rank: arms[arm]['hits']+=1; arms[arm]['rr'].append(1/rank)
        else: arms[arm]['rr'].append(0)
        arms[arm]['ms'].append(elapsed)
for a,v in arms.items():
 v['recall_at_5']=v['hits']/len(data); v['mrr']=sum(v['rr'])/len(data); v['p50_ms']=statistics.median(v['ms']); v.pop('rr'); v.pop('ms')
report={'schema':'contextual-retrieval-ab:v1','namespace':'topic-pods','cases':len(data),'arms':arms,'note':'Synthetic held-out topic mix; target matching is retrieval metric, not LLM answer quality.'}
out=ROOT/'runs/contextual-retrieval-ab-001.json'; out.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2)); backend.close()
