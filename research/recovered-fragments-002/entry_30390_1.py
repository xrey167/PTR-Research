from pathlib import Path
import sys, json, re, time
import torch
from safetensors.torch import load_file
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'interaction_memory'))
import run_im1 as im
from tensor_memory import InteractionMemory
PAIRS=(('X','Y'),('C','D'),('K','M'))
WORLDS=((1,2,0),(1,2,1),(3,1,0),(3,1,1))
OPS=im.QUESTIONS[:4]
MODEL=ROOT.parent.parent/'models/Qwen2.5-3B-Instruct'
def rename(s,pair):
    return re.sub(r'\b[AB]\b',lambda m: pair[m.group()=='B'],s)
def definition(tok,pair,w):
    s=rename(im.render(tok,w,im.MARKER),pair)
    prefix,closing=s.split(im.MARKER)
    ids=tok(prefix,add_special_tokens=False)['input_ids']
    for _,q in OPS:
        assert tok(rename(im.render(tok,w,q),pair),add_special_tokens=False)['input_ids']==ids+tok(rename(q,pair)+closing,add_special_tokens=False)['input_ids']
    return dict(prefix_text=prefix,prefix_ids=ids,closing=closing)
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=HERE/'run';out.mkdir(exist_ok=False)
    src=[Path(__file__),HERE/'PROTOCOL.md',ROOT/'interaction_memory/run_im1.py',ROOT/'interaction_memory/tensor_memory.py',ROOT/'interaction_memory/schema.py',ROOT/'value_capsule/run_capsule.py',ROOT/'entity_port/run_entity.py',ROOT/'late_cut/run_cut.py',ROOT/'stronger/latent_3b.py',ROOT/'cases.py']
    sources={str(p.relative_to(ROOT)):im.kvc.sha(p) for p in src}
    assets={n:im.kvc.sha(MODEL/n) for n in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')}
    assert {n:im.kvc.sha(MODEL/n) for n in im.kvc.EXPECTED}==im.kvc.EXPECTED
    bank_path=ROOT/'interaction_memory/run/basis.safetensors'
    bank_hash=im.kvc.sha(bank_path)
    assert bank_hash=='6e4d0a2d89688357607134b790a4512e6306d5b33f59eccaca7bee686bf6afc1'
    im.put(out/'protocol.json',dict(sources=sources,assets=assets,weights=im.kvc.EXPECTED,bank_sha256=bank_hash,pairs=PAIRS,worlds=WORLDS,ops=OPS,full_dod_pass=False))
    t=time.perf_counter();engine=im.kvc.Engine(MODEL)
    calls=[0]
    def count(m,a):calls[0]+=1
    hook=engine.model.register_forward_pre_hook(count)
    bank={tuple(map(int,k)):v for k,v in load_file(bank_path).items()}
    memories={o:InteractionMemory({w:v for w,v in bank.items() if im.support(w)<=o},o) for o in (1,2)}
    defs={(p,w):definition(engine.tokenizer,p,w) for p in PAIRS for w in ((0,0,0),)+WORLDS}
    assert all(len(d['prefix_ids'])==49 for d in defs.values())
    anchors={};gates=[];records=[];compile_rows=[]
    def generate(image,p,w,mode):
        cs=[dict(value=0,query=rename(q,p)) for _,q in OPS]
        seq,_,_=im.kvc.cached_generate(engine,cs,im.capsule(engine,image,defs[p,w]))
        for (op,q),ids in zip(OPS,seq,strict=True):
            end=next((i for i,v in enumerate(ids) if v in engine.eos),None)
            txt=engine.tokenizer.decode(ids if end is None else ids[:end],skip_special_tokens=False)
            gold=im.expected(w,op)
            ok=end is not None and re.fullmatch(re.escape(gold)+r'[.!?]?',txt.strip().lower()) is not None
            records.append(dict(pair=p,world=w,mode=mode,op=op,query=rename(q,p),ids=ids,text=txt,gold=gold,correct=ok))
        im.put(out/'rows.json',records)
        print('BT1',p,w,mode,sum(r['correct'] for r in records if r['mode']==mode),flush=True)
    for p in PAIRS:
        d=defs[p,(0,0,0)];anchors[p],secs=im.compile_image(engine,d)
        compile_rows.append(dict(pair=p,world=(0,0,0),seconds=secs,sha256=im.digest_tensor(anchors[p])))
        cs=[dict(value=0,query=rename(q,p)) for _,q in OPS];cap=im.capsule(engine,anchors[p],d)
        cached,lc,_=im.kvc.cached_generate(engine,cs,cap);full,lf=im.kvc.full_reference(engine,cs,cap)
        gates.append(dict(pair=p,cached=cached,full=full,logit_error=float((lc-lf).abs().max())))
        im.put(out/'execution_gate.json',gates)
        assert cached==full and gates[-1]['logit_error']==0
    for p in PAIRS:
        for w in WORLDS:
            generate(anchors[p],p,w,'anchor')
            donor,_=memories[2].compose(w);generate(donor,p,w,'donor2')
            for o in (1,2):
                image=anchors[p].double().clone()
                for key,v in memories[o].terms.items():
                    if all(a==0 or a==b for a,b in zip(key,w)):image+=v
                generate(image.bfloat16(),p,w,'rebase'+str(o))
    im.put(out/'candidate_seal.json',dict(rows_sha256=im.kvc.sha(out/'rows.json'),rows=records,target_nonanchor_compiles=0,forward_calls=calls[0]))
    for p in PAIRS:
        for w in WORLDS:
            exact,secs=im.compile_image(engine,defs[p,w]);compile_rows.append(dict(pair=p,world=w,seconds=secs,sha256=im.digest_tensor(exact)))
            generate(exact,p,w,'exact')
    totals={mode:dict(n=sum(r['mode']==mode for r in records),correct=sum(r['correct'] for r in records if r['mode']==mode)) for mode in ('anchor','donor2','rebase1','rebase2','exact')}
    sources_ok=sources=={str(p.relative_to(ROOT)):im.kvc.sha(p) for p in src}
    assert sources_ok and assets=={n:im.kvc.sha(MODEL/n) for n in assets} and im.kvc.sha(bank_path)==bank_hash
    assert all(not p.requires_grad for p in engine.model.parameters())
    im.put(out/'compilation.json',compile_rows)
    im.put(out/'results.json',dict(status='complete',totals=totals,gates={m:totals[m]['correct']==48 and totals['anchor']['correct']<48 for m in ('rebase1','rebase2')},qa_generations=len(records),execution_gate_generations=24,forward_calls=calls[0],seconds=time.perf_counter()-t,source_and_asset_hashes_unchanged=True,full_dod_pass=False))
    hook.remove();print(json.dumps(totals),flush=True)
if __name__=='__main__':main()
