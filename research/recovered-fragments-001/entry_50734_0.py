\"\"\"QC-C1: matched shared readers with opaque counterfactual worlds.\"\"\"
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from cases import datasets,question,gold,score,strict_ok,COLORS
from conditioned_operator import SharedAffineOperator


def main(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    data=datasets();(args.output/'cases.json').write_text(json.dumps(data,indent=2)+'\\n')
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    model.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    term=model.generation_config.eos_token_id;term=set(term if isinstance(term,list) else [term])
    eos=tok.eos_token_id;layer=args.layer;rank=16
    assert eos in term

    def render(c,text_memory=False):
        q=question(c)
        if text_memory:q=f\"Stored fact: the color of item {c['identity']} is {COLORS[c['value']]}.\\n\"+q
        return tok.apply_chat_template([{'role':'system','content':'Follow the requested answer format. Give only the requested single word, without explanation.'},
        {'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=False)

    def detach(x):
        if torch.is_tensor(x):return x.detach().clone()
        if isinstance(x,tuple):return tuple(detach(t) for t in x)
        if isinstance(x,dict):return {k:detach(v) for k,v in x.items()}
        return x

    def capture(inputs,keep):
        cap={}
        def pre(module,ins,kwargs):cap.update(hidden=detach(ins[0]),kwargs=detach(kwargs))
        handle=model.model.layers[layer+1].register_forward_pre_hook(pre,with_kwargs=True)
        try:
            with torch.no_grad():out=model(**inputs,use_cache=False,logits_to_keep=keep)
        finally:handle.remove()
        cap.update(reference=out.logits.detach(),inputs=inputs)
        return cap

    def batches(cases,teacher=False):
        bs=[]
        for off in range(0,len(cases),8):
            chunk=cases[off:off+8];inputs=tok([render(c) for c in chunk],padding=True,return_tensors='pt',add_special_tokens=False)
            labels=[tok.encode(gold(c['spec'],c['value']),add_special_tokens=False) for c in chunk]
            assert all(len(x)==1 for x in labels)
            labels=torch.tensor([x[0] for x in labels])
            if teacher:
                inputs['input_ids']=torch.cat([inputs['input_ids'],labels[:,None]],1)
                inputs['attention_mask']=torch.cat([inputs['attention_mask'],torch.ones((len(chunk),1),dtype=torch.long)],1)
            cap=capture(inputs,2 if teacher else 1)
            cap.update(cases=chunk,values=torch.tensor([c['value'] for c in chunk]),labels=labels,position=-2 if teacher else -1)
            bs.append(cap)
        return bs

    def suffix(batch,op=None,replacement=None):
        h=batch['hidden'];pos=batch['position']
        if op is not None or replacement is not None:
            h=h.clone();h[:,pos]=op(h[:,pos],batch['values']) if replacement is None else replacement
        for block in model.model.layers[layer+1:]:h=block(h,**batch['kwargs'])
        keep=2 if pos==-2 else 1
        return model.lm_head(model.model.norm(h)[:,-keep:])

    def hook_for(op,values):
        first=[True]
        def inject(module,ins,outs):
            if not first[0]:return outs
            first[0]=False
            h=outs[0] if isinstance(outs,tuple) else outs
            h=h.clone();h[:,-1]=op(h[:,-1],values)
            return (h,)+outs[1:] if isinstance(outs,tuple) else h
        return inject

    def generate(cases,op=None,wrong=False,text_memory=False):
        rows=[]
        for off in range(0,len(cases),16):
            chunk=cases[off:off+16]
            inputs=tok([render(c,text_memory) for c in chunk],padding=True,return_tensors='pt',add_special_tokens=False)
            values=torch.tensor([(c['value']+int(wrong))%4 for c in chunk])
            handle=None if op is None else model.model.layers[layer].register_forward_hook(hook_for(op,values))
            try:
                with torch.no_grad():seq=model.generate(**inputs,max_new_tokens=6,do_sample=False,use_cache=True,pad_token_id=eos)
            finally:
                if handle is not None:handle.remove()
            for c,ids in zip(chunk,seq[:,inputs['input_ids'].shape[1]:].tolist()):
                end=next((i for i,t in enumerate(ids) if t in term),None)
                text=tok.decode(ids if end is None else ids[:end],skip_special_tokens=False)
                expected=gold(c['spec'],c['value']);shifted=gold(c['spec'],(c['value']+1)%4)
                rows.append(dict(case=c,token_ids=ids,text=text,eos_seen=end is not None,expected=expected,
                                 correct=strict_ok(text,expected,end is not None),shifted_expected=shifted,
                                 shifted_correct=strict_ok(text,shifted,end is not None)))
        return dict(score=score(rows),rows=rows)

    print('CACHE_START',flush=True)
    train=batches(data['train'],True);valid=batches(data['valid']);test=batches(data['test'])
    bare=torch.cat([b['hidden'][:,b['position']] for b in train])
    center=bare.mean(0)
    _,_,vt=torch.linalg.svd(bare-center,full_matrices=False);Q=vt[:rank].T.contiguous()
    np.savez(args.output/'basis.npz',Q=Q.numpy(),center=center.numpy())
    with torch.no_grad():
        split=[float((suffix(b)-b['reference']).abs().max()) for b in [train[0],valid[0],test[0]]]
        assert max(split)<1e-5,split
        altered={k:v.clone() for k,v in train[0]['inputs'].items()};altered['input_ids'][:,-1]=tok.encode('square',add_special_tokens=False)[0]
        alternative=capture(altered,2)
        causal=float((alternative['hidden'][:,-2]-train[0]['hidden'][:,-2]).abs().max());assert causal<1e-6
    checks=dict(split_logit_errors=split,future_target_causal_error=causal)
    print('EXECUTION_CHECKS',json.dumps(checks),flush=True)

    def repair_check(op):
        b=test[0];base=b['hidden'][:,-1]
        rows=[]
        with torch.no_grad():
            for old in [None,0,1,2,3]:
                previous=op(base,old)
                for new in [None,0,1,2,3]:
                    fresh=op(base,new);repaired=op.repair(previous,old,new)
                    rel=(repaired-fresh).norm(dim=-1)/fresh.norm(dim=-1).clamp_min(1e-12)
                    fl=suffix(b,replacement=fresh)[:,-1];rl=suffix(b,replacement=repaired)[:,-1]
                    rows.append(dict(old=old,new=new,n=len(base),max_case_relative_error=float(rel.max()),
                        max_absolute_error=float((repaired-fresh).abs().max()),logit_error=float((fl-rl).abs().max()),
                        top_agreement=int((fl.argmax(-1)==rl.argmax(-1)).sum())))
            state=base.clone();old=None;history=[]
            for i in range(256):
                new=[0,3,1,None,2][i%5];state=op.repair(state,old,new);old=new
                fresh=op(base,new)
                rel=(state-fresh).norm(dim=-1)/fresh.norm(dim=-1).clamp_min(1e-12)
                if i in [0,15,63,255]:history.append(dict(replacements=i+1,max_case_relative_error=float(rel.max()),max_absolute_error=float((state-fresh).abs().max())))
            fl=suffix(b,replacement=op(base,old))[:,-1];rl=suffix(b,replacement=state)[:,-1]
        passed=all(r['max_case_relative_error']<1e-5 and r['top_agreement']==r['n'] for r in rows) and history[-1]['max_case_relative_error']<1e-5 and torch.equal(fl.argmax(-1),rl.argmax(-1))
        return dict(pairwise=rows,sequence=history,sequence_final_top_agreement=int((fl.argmax(-1)==rl.argmax(-1)).sum()),numerical_gate=bool(passed))

    controls={}
    for name,textmode in [('no_memory',False),('text_oracle',True)]:
        controls[name]=generate(data['test'],text_memory=textmode)
        print('CONTROL',name,json.dumps(controls[name]['score']),flush=True)
    (args.output/'controls.json').write_text(json.dumps(controls,indent=2)+'\\n')
    results=[]
    for seed in [17,29]:
        for name,rho in [('affine',None),('bounded',.9)]:
            torch.manual_seed(seed)
            op=SharedAffineOperator(Q.shape[0],rank,4,rho=rho,Q=Q,center=center)
            with torch.no_grad():op.W.normal_(0,1e-4)
            optim=torch.optim.Adam(op.parameters(),lr=.02)
            history=[];best=None;best_state=None;start=time.perf_counter()
            for step in range(args.steps+1):
                if step in [0,args.steps//3,2*args.steps//3,args.steps]:
                    validation=generate(data['valid'],op)
                    metric=(min(v['accuracy'] for v in validation['score']['operations'].values()),validation['score']['overall']['accuracy'])
                    record=dict(step=step,score=validation['score'],worst_family=metric[0],seconds=time.perf_counter()-start)
                    history.append(record)
                    if best is None or metric>best[0]:best=(metric,record);best_state=copy.deepcopy(op.state_dict())
                    print('VALID',name,seed,json.dumps(record),flush=True)
                if step==args.steps:break
                batch=train[int(torch.randint(len(train),(1,)))]
                optim.zero_grad(set_to_none=True);logits=suffix(batch,op)
                loss=torch.nn.functional.cross_entropy(logits[:,0],batch['labels'])+torch.nn.functional.cross_entropy(logits[:,1],torch.full_like(batch['labels'],eos))
                loss.backward();torch.nn.utils.clip_grad_norm_(op.parameters(),1.0);optim.step()
                if step%40==0:print('LOSS',name,seed,step,float(loss.detach()),flush=True)
            op.load_state_dict(best_state)
            primary=generate(data['test'],op);shifted=generate(data['test'],op,wrong=True)
            eligible=[i for i,c in enumerate(data['test']) if gold(c['spec'],c['value'])"'!=gold(c['"'spec'],(c['value']+1)%4)]
            pair=sum(primary['rows'][i]['correct'] and shifted['rows'][i]['shifted_correct'] for i in eligible)
            numerical=repair_check(op)
            # Compare teacher-forced full-prefix suffix to real KV autoregression for the same first token.
            with torch.no_grad():
                b=train[0];inp={k:v[:,:-1].clone() for k,v in b['inputs'].items()}
                handle=model.model.layers[layer].register_forward_hook(hook_for(op,b['values']))
                try:prefill=model(**inp,use_cache=True,logits_to_keep=1)
                finally:handle.remove()
                after=model(input_ids=b['labels'][:,None],attention_mask=b['inputs']['attention_mask'],past_key_values=prefill.past_key_values,use_cache=True,logits_to_keep=1)
                teacher=suffix(b,op)
                autoreg_error=float((after.logits[:,-1]-teacher[:,1]).abs().max())
            result=dict(method=name,rho=rho,seed=seed,layer=layer,steps=args.steps,selected=best[1],history=history,
                generation=primary,shifted_generation=shifted,causal_pairs=dict(n=len(eligible),correct=pair,accuracy=pair/len(eligible)),
                numerical=numerical,diagnostics=op.diagnostics(),teacher_vs_autoregressive_eos_logit_error=autoreg_error,
                seconds=time.perf_counter()-start,shared_parameters=sum(p.numel() for p in op.parameters()))
            result['interface_gate_pass']=all(s['accuracy']>=.9 for s in primary['score']['operations'].values()) and pair/len(eligible)>=.9 and numerical['numerical_gate']
            np.savez(args.output/f'{name}_seed{seed}.npz',**{k:v.detach().numpy() for k,v in op.state_dict().items()})
            (args.output/f'{name}_seed{seed}.json').write_text(json.dumps(result,indent=2)+'\\n')
            results.append(result)
            print('FINISHED',name,seed,json.dumps(primary['score']), 'PAIR',pair,len(eligible),'REPAIR',numerical['numerical_gate'],flush=True)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    output=dict(experiment='QC-C1',checks=checks,runs=results,controls=controls,backbone_frozen=True,
        model='Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',parameter_count=sum(p.numel() for p in model.parameters()),
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,device='cpu',dtype='float32'),
        source_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['run_experiment.py','cases.py','conditioned_operator.py']},
        per_fact_gradient_updates=0,unique_test_ids=len({c['identity'] for c in data['test']}),counterfactual_test_cases=len(data['test']),full_dod_pass=False)
    (args.output/'results.json').write_text(json.dumps(output,indent=2)+'\\n');print('ALL_FINISHED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'run_c1');parser.add_argument('--layer',type=int,default=23);parser.add_argument('--steps',type=int,default=240)
    main(parser.parse_args())
