"""Separate saved-output audit; no independent external reviewer is claimed."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'binding_transfer/run'
def read(n):return json.loads((OUT/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=read('rows.json');seal=read('candidate_seal.json');p=read('protocol.json');res=read('results.json')
assert len(rows)==240 and rows[:192]==seal['rows'] and seal['target_nonanchor_compiles']==0
# atomic_json serializes with its own whitespace: verify semantic sealed rows,
# plus source hashes, rather than guessing serialization to recreate prior hash.
assert all(sha(ROOT/f)==h for f,h in p['sources'].items())
seen=set();totals={};paired={}
colors=('red','blue','green','yellow')
for r in rows:
    key=(tuple(r['pair']),tuple(r['world']),r['op'])
    assert (key,r['mode']) not in seen;seen.add((key,r['mode']))
    a,b,s=r['world'];vals={'direct_a':a,'direct_b':b,'selected':(a,b)[s],'other':(a,b)[1-s]}
    assert r['gold']==colors[vals[r['op']]]
    # EOS IDs from fixed Qwen tokenizer configuration, not assumed numeric constants.
    cfg=json.loads((ROOT.parent.parent/'models/Qwen2.5-3B-Instruct/generation_config.json').read_text())
    eos=cfg['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
    eos_seen=any(t in eos for t in r['ids'])
    ok=eos_seen and re.fullmatch(re.escape(r['gold'])+r'[.!?]?',r['text'].strip().lower()) is not None
    assert ok==r['correct']
    totals.setdefault(r['mode'],dict(n=0,correct=0));totals[r['mode']]['n']+=1;totals[r['mode']]['correct']+=int(ok)
    paired.setdefault(key,{})[r['mode']]=r
assert totals==res['totals'] and len(paired)==48
comparison={}
for mode in ('anchor','donor2','rebase1','rebase2'):
    comparison[mode]=dict(reference_only_correct=sum(x['exact']['correct'] and not x[mode]['correct'] for x in paired.values()),candidate_only_correct=sum(x[mode]['correct'] and not x['exact']['correct'] for x in paired.values()),identical_sequences=sum(x[mode]['ids']==x['exact']['ids'] for x in paired.values()))
assert len(read('execution_gate.json'))==3 and all(x['cached']==x['full'] and x['logit_error']==0 for x in read('execution_gate.json'))
assert len(read('compilation.json'))==15
out=dict(passed=True,scope='same-assistant saved-output consistency and source audit, not independent replication',qa_cases=240,gate_sequences=24,totals=totals,comparison=comparison,full_dod_pass=False)
(OUT/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
