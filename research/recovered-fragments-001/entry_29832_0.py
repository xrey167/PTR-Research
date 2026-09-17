from pathlib import Path
import json,hashlib,re
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'binding_transfer/run_bt2'
def read(f):return json.loads((OUT/f).read_text())
p=read('protocol.json');r=read('results.json');rows=read('rows.json');seal=read('candidate_seal.json')
assert len(rows)==48 and rows[:32]==seal['rows'] and seal['target_joint_compiles']==0
assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in p['sources'].items())
model=ROOT.parent.parent/'models/Qwen2.5-3B-Instruct'
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
eos=json.loads((model/'generation_config.json').read_text())['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
colors=('black','white','purple','orange');totals={};paired={};seen=set()
for row in rows:
 w=tuple(row['world']);op=row['op'];m=row['mode'];a,b,s=w;key=(w,op)
 assert (key,m) not in seen;seen.add((key,m))
 v={'direct_a':a,'direct_b':b,'selected':(a,b)[s],'other':(a,b)[1-s]}[op]
 assert row['gold']==colors[v]
 end=next((i for i,t in enumerate(row['ids']) if t in eos),None)
 text=tok.decode(row['ids'] if end is None else row['ids'][:end],skip_special_tokens=False)
 assert row['text']==text
 ok=end is not None and re.fullmatch(re.escape(colors[v])+r'[."'!?]?'"',text.strip().lower()) is not None
 assert ok==row['correct']
 totals.setdefault(m,dict(n=0,correct=0));totals[m]['n']+=1;totals[m]['correct']+=int(ok)
 paired.setdefault(key,{})[m]=row
assert totals==r['totals'] and len(paired)==16
assert len(read('compilation.json'))==12
g=read('execution_gate.json');assert g['cached']==g['full'] and g['logit_error']==0
cmp={m:dict(reference_only_correct=sum(x['exact']['correct'] and not x[m]['correct'] for x in paired.values()),candidate_only_correct=sum(x[m]['correct'] and not x['exact']['correct'] for x in paired.values()),identical_sequences=sum(x[m]['ids']==x['exact']['ids'] for x in paired.values())) for m in ('target1','hybrid2')}
out=dict(passed=True,scope='same-assistant saved-token and source consistency audit; no independent replication',totals=totals,comparison=cmp,generation_count=56,full_dod_pass=False)
(OUT/'audit.json').write_text(json.dumps(out,indent=2)+'\\n');print(json.dumps(out))
