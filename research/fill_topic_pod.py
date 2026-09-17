"""Fill and smoke-test the linked typed topic Pod locally."""
import json, time, hashlib, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neural_pods.local_search import LocalSearchBackend
from neural_pods.pod_cache import PodCache
from neural_pods.contextual_retrieval import contextual_document, contextual_query
out=ROOT/'runs'/'topic-pod-filled-002'; out.mkdir(parents=True,exist_ok=True)
d=json.load(open(ROOT/'runs/pod-training-mix-001/dataset.json'))
backend=LocalSearchBackend(out/'namespace.sqlite3'); backend.create_namespace('topic-pods')
count=0
for split,rows in d.items():
    for r in rows:
        md={'type':r['pod_type'],'pod_type':r['pod_type'],'domain':r['domain'],'semantic_role':r['semantic_role'],'tags':r['tags'],'origin_keys':r['origin_keys'],'knowledge_key':r['knowledge_key'],'generation_key':r['generation_key'],'status':'active','acl':['*'],'split':split,'target':r['target'],'source_id':r['id']}
        backend.upsert('topic-pods',hashlib.sha256(r['id'].encode()).hexdigest()[:24],text=contextual_document(f"{r['input']} Answer: {r['target']}",md),metadata=md); count+=1
warm=backend.prewarm('topic-pods'); cache=PodCache(max_entries=8192,ttl_seconds=300); checks=[]
for r in d['test'][:100]:
    hits=cache.search(backend,'topic-pods',text=contextual_query(r['input'],pod_type=r['pod_type'],domain=r['domain'],tags=r['tags']),pod_type=r['pod_type'],tags=r['tags'],principal='*',top_k=3)
    checks.append(bool(hits and r['target'] in hits[0].text))
branch=backend.branch('topic-pods',target='topic-pods-ablation-002')
report={'status':'filled_and_smoke_tested','namespace':'topic-pods','branch':branch,'rows':count,'prewarm':warm,'test_cases':len(checks),'top1_target_matches':sum(checks),'top1_rate':sum(checks)/len(checks),'cache':dict(cache.stats()),'namespace_metadata':backend.namespace_metadata('topic-pods')}
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2)); backend.close()

