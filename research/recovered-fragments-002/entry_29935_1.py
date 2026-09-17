from pathlib import Path
import sys,json,re,time
import torch
from safetensors.torch import load_file
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));import run_bt1 as bt
im=bt.im
COLORS=('black','white','purple','orange');PAIR=('X','Y');ANCHOR=(0,0,0)
def render(tok,w,q):
 a,b,s=w
 facts=f'Stored facts:\nThe color of item X is {COLORS[a]}.\nThe color of item Y is {COLORS[b]}.\nThe selected item is {PAIR[s]}.\n'
 return tok.apply_chat_template([dict(role='system',content=im.SYSTEM),dict(role='user',content=facts+bt.rename(q,PAIR))],tokenize=False,add_generation_prompt=True)
def definition(tok,w):
 prefix,closing=render(tok,w,im.MARKER).split(im.MARKER);ids=tok(prefix,add_special_tokens=False)['input_ids']
 for _,q in bt.OPS:
  assert tok(render(tok,w,q),add_special_tokens=False)['input_ids']==ids+tok(bt.rename(q,PAIR)+closing,add_special_tokens=False)['input_ids']
 return dict(prefix_text=prefix,prefix_ids=ids,closing=closing)
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 out=HERE/'run_bt2';out.mkdir(exist_ok=False)
 src=[Path(__file__),HERE/'PROTOCOL_BT2.md',HERE/'run_bt1.py']+[ROOT/f for f in json.loads((HERE/'run/protocol.json').read_text())['sources'] if ROOT/f!=HERE/'run_bt1.py']
 sources={str(p.relative_to(ROOT)):im.kvc.sha(p) for p in src}
 old_protocol=json.loads((HERE/'run/protocol.json').read_text())
 assert {n:im.kvc.sha(bt.MODEL/n) for n in im.kvc.EXPECTED}==im.kvc.EXPECTED
 assert old_protocol['assets']=={n:im.kvc.sha(bt.MODEL/n) for n in old_protocol['assets']}
 basis=[w for w in im.WORLDS if im.support(w)<=1]
 im.put(out/'protocol.json',dict(sources=sources,assets=old_protocol['assets'],weights=im.kvc.EXPECTED,colors=COLORS,pair=PAIR,basis=basis,worlds=bt.WORLDS,questions=bt.OPS,full_dod_pass=False))
 engine=im.kvc.Engine(bt.MODEL);started=time.perf_counter();calls=[0]
 def count(m,a):calls[0]+=1
 hook=engine.model.register_forward_pre_hook(count)
 defs={w:definition(engine.tokenizer,w) for w in basis+list(bt.WORLDS)}
 assert all(len(d['prefix_ids'])==49 for d in defs.values())
 bank={};comp=[]
 for w in basis:
  bank[w],sec=im.compile_image(engine,defs[w]);comp.append(dict(world=w,seconds=sec,sha256=im.digest_tensor(bank[w])))
 target=bt.InteractionMemory(bank,1)
 old=load_file(ROOT/'interaction_memory/run/basis.safetensors')
 donor=bt.InteractionMemory({tuple(map(int,k)):v for k,v in old.items()},2)
 cs=[dict(value=0,query=bt.rename(q,PAIR)) for _,q in bt.OPS]
 cap=im.capsule(engine,bank[ANCHOR],defs[ANCHOR])
 ca,lc,_=im.kvc.cached_generate(engine,cs,cap);full,lf=im.kvc.full_reference(engine,cs,cap)
 assert ca==full and float((lc-lf).abs().max())==0
 im.put(out/'execution_gate.json',dict(cached=ca,full=full,logit_error=0))
 records=[]
 def generate(t,w,mode):
  seq,_,_=im.kvc.cached_generate(engine,cs,im.capsule(engine,t,defs[w]))
  a,b,s=w;goldvals={'direct_a':a,'direct_b':b,'selected':(a,b)[s],'other':(a,b)[1-s]}
  for (op,q),ids in zip(bt.OPS,seq,strict=True):
   end=next((i for i,v in enumerate(ids) if v in engine.eos),None)
   txt=engine.tokenizer.decode(ids if end is None else ids[:end],skip_special_tokens=False);gold=COLORS[goldvals[op]]
   ok=end is not None and re.fullmatch(re.escape(gold)+r'[.!?]?',txt.strip().lower()) is not None
   records.append(dict(world=w,mode=mode,op=op,ids=ids,text=txt,gold=gold,correct=ok))
  im.put(out/'rows.json',records);print('BT2',w,mode,sum(r['correct'] for r in records if r['mode']==mode),flush=True)
 for w in bt.WORLDS:
  t=target.base.clone()
  for key,v in target.terms.items():
   if all(a==0 or a==b for a,b in zip(key,w)):t+=v
  generate(t.bfloat16(),w,'target1')
  for key,v in donor.terms.items():
   if im.support(key)==2 and all(a==0 or a==b for a,b in zip(key,w)):t+=v
  generate(t.bfloat16(),w,'hybrid2')
 im.put(out/'candidate_seal.json',dict(rows=records,target_joint_compiles=0))
 for w in bt.WORLDS:
  t,sec=im.compile_image(engine,defs[w]);comp.append(dict(world=w,seconds=sec,sha256=im.digest_tensor(t)));generate(t,w,'exact')
 assert all(im.kvc.sha(ROOT/f)==h for f,h in sources.items())
 assert all(not p.requires_grad for p in engine.model.parameters())
 totals={m:dict(n=16,correct=sum(r['correct'] for r in records if r['mode']==m)) for m in ('target1','hybrid2','exact')}
 im.put(out/'compilation.json',comp)
 im.put(out/'results.json',dict(status='complete',totals=totals,hybrid_gate=totals['hybrid2']['correct']==16,hybrid_gain=totals['hybrid2']['correct']-totals['target1']['correct'],forward_calls=calls[0],qa_generations=48,execution_gate_generations=8,seconds=time.perf_counter()-started,sources_unchanged=True,full_dod_pass=False))
 print(json.dumps(totals),flush=True);hook.remove()
if __name__=='__main__':main()
