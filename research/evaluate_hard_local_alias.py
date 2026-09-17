import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neural_pods.local_search import LocalSearchBackend
from neural_pods.alias_resolver import AliasResolver
from research.generate_hard_benchmark import build
def run():
 d=build(); out=ROOT/'runs/hard-local-alias-001';out.mkdir(exist_ok=True); b=LocalSearchBackend(out/'search.sqlite');b.create_namespace('hard')
 aliases={f'partner-{i:02d}-alias': next(x['text'].split()[3] for x in d['corpus'] if x['doc_id']==f'doc-{i}-0') for i in range(20,30)}
 for row in d['corpus']:
  b.upsert('hard',row['doc_id'],text=row['text'],metadata={'target_doc':row['doc_id'],'supplier':row['supplier'],'split':row['split'],'status':'active','acl':['*']})
 resolver=AliasResolver(aliases); plain=resolved=0
 for q in d['queries']:
  if any(h.metadata.get('target_doc')==q['target_doc'] for h in b.search('hard',text=q['question'],principal='*',top_k=3)):plain+=1
  rq=resolver.resolve(q['question']); hits=b.search('hard',text=rq,principal='*',top_k=3)
  if any(h.metadata.get('target_doc')==q['target_doc'] for h in hits):resolved+=1
 report={'schema':'hard-local-alias-resolution:v1','queries':len(d['queries']),'plain_recall_at_3':plain/len(d['queries']),'alias_resolved_recall_at_3':resolved/len(d['queries']),'aliases':len(aliases),'resolver':'exact longest-match advisory rewrite'}
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));b.close()
if __name__=='__main__':run()

