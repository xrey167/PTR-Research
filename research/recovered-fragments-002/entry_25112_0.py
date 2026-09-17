"""Standard-library token/count audit; does not import model or candidate."""
import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent.parent;here=Path(__file__).parent
if __name__=='__main__':
 seal=json.loads((here/'seal.json').read_text())
 assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in seal.items())
 folder=here/'validation';rows=[json.loads(x) for x in (folder/'answers.jsonl').read_text().splitlines()]
 result=json.loads((folder/'results.json').read_text());assert len(rows)==96
 counts={};pairs={};questions={};injected=0
 for mode in ['none','cc8','opposite','text']:
  batch=[r for r in rows if r['mode']==mode];assert len(batch)==24
  for r in batch:
   c=r['case'];expected='Yes' if bool(c['value'])!=c['negated'] else 'No'
   assert c['expected']==expected
   correct=r['text']==expected and r['eos_seen'];assert correct==r['correct']
   if mode in ('cc8','opposite'):
    assert r['hook_called']==1 and r['delta_norm']>0;injected+=1
  counts[mode]=sum(r['correct'] for r in batch)
  groups={}
  for r in batch:groups.setdefault(r['case']['question'],[]).append(r)
  assert len(groups)==12 and all({r['case']['value'] for r in rs}=={0,1} for rs in groups.values())
  pairs[mode]=sum(all(r['correct'] for r in rs) for rs in groups.values())
 assert counts==result['correct'];assert pairs==result['counterfactual_pairs_correct']
 assert result['candidate_gate']==(counts['cc8']==24 and pairs['cc8']==12)
 output={'passed':True,'answers_recounted':len(rows),'injected_answers_with_nonzero_state_change':injected,'correct':counts,'counterfactual_pairs_correct':pairs,'candidate_gate':result['candidate_gate'],'external_replication':False}
 (folder/'audit.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
