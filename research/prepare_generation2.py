import json,shutil,hashlib,math,sys
from pathlib import Path
src,out=Path(sys.argv[1]),Path(sys.argv[2]); shutil.copytree(src,out)
items=[
('How does contextual retrieval improve a Pod?','It adds typed context to embeddings and BM25 while preserving the raw fact.'),
('Why use fixed-size posting blocks?','They reduce metadata overhead and enable batched decoding for high-cardinality filters.'),
('What does a spaCy lookup candidate require?','It must resolve through the Registry before a current generation can be used.'),
('Which profile fits source code memory?','The programming profile with technical tags and a dedicated cache namespace.'),
('What database is intended for production namespaces?','PostgreSQL with JSONB metadata, GIN indexes and dedicated posting blocks.'),
('Can a revoked alias activate a Pod?','No. Registry lifecycle validation rejects revoked or stale generations.'),
('What is kept stable when a Pod branches?','The Pod identity remains stable while the artifact generation changes.'),
('What should a model Pod contain?','A verifiable adapter artifact, typed metadata, provenance and lifecycle references.'),]
for name in ('train','dev','test'):
 p=out/'inputs'/f'{name}.json'; rows=json.loads(p.read_text());
 for i,(q,a) in enumerate(items): rows.append({'id':f'{name}:generation2:{i}','task':'model_pod_generation2','pod_type':'model' if i in (3,7) else ('retrieval' if i in (0,1,4) else 'reasoning'),'language':'en','evidence':None,'question':q,'history':[],'target':a,'assessment':{'generation2':True}})
 p.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
protocol=json.loads((out/'protocol.json').read_text()); protocol['rows']={k:len(json.loads((out/'inputs'/f'{k}.json').read_text())) for k in ('train','dev','test')}; t=json.loads((out/'inputs/config.json').read_text())['training']; protocol['optimizer_updates_per_epoch']=math.ceil(protocol['rows']['train']/t['gradient_accumulation_steps']); protocol['optimizer_updates']=protocol['optimizer_updates_per_epoch']*t['epochs']; protocol['warmup_updates']=math.ceil(protocol['optimizer_updates']*t['warmup_ratio']); protocol['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'inputs').iterdir()}; protocol['status']='prepared_not_trained_generation2'; (out/'protocol.json').write_text(json.dumps(protocol,indent=2)); print(protocol['rows'],protocol['optimizer_updates'])
