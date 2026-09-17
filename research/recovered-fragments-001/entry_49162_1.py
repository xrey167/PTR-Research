\"\"\"QC-CR1: recovered counterfactual reader research with resumable checkpoints.\"\"\"
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from cases import datasets,question,gold,score,strict_ok,COLORS
from conditioned_operator import SharedAffineOperator
from coupling_operator import SharedCouplingOperator


def atomic_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\\n');os.replace(tmp,path)

def atomic_torch(path,data):
    tmp=path.with_suffix('.tmp');torch.save(data,tmp);os.replace(tmp,path)

class Boundary:
    def __init__(self,modelpath,layer):
        self.model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
        self.model.requires_grad_(False);self.layer=layer
        self.tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
        self.tok.padding_side='left';self.tok.pad_token=self.tok.eos_token
        self.eos=self.tok.eos_token_id;t=self.model.generation_config.eos_token_id
        self.term=set(t if isinstance(t,list) else [t]);assert self.eos in self.term
    def render(self,c,text_memory=False):
        q=question(c)
        if text_memory:q=f\"Stored fact: the color of item {c['identity']} is {COLORS[c['value']]}.\\n\"+q
        return self.tok.apply_chat_template([{'role':'system','content':'Follow the requested answer format. Give only the requested single word, without explanation.'},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    @staticmethod
    def detach(x):
        if torch.is_tensor(x):return x.detach().clone()
        if isinstance(x,tuple):return tuple(Boundary.detach(v) for v in x)
        if isinstance(x,dict):return {k:Boundary.detach(v) for k,v in x.items()}
        return x
    def capture(self,inputs,keep):
        cap={}
        def pre(module,ins,kwargs):cap.update(hidden=self.detach(ins[0]),kwargs=self.detach(kwargs))
        handle=self.model.model.layers[self.layer+1].register_forward_pre_hook(pre,with_kwargs=True)
        try:
            with torch.no_grad():ref=self.model(**inputs,use_cache=False,logits_to_keep=keep).logits.detach()
        finally:handle.remove()
        cap.update(reference=ref,inputs=inputs);return cap
    def batches(self,cases,teacher=False):
        result=[]
        for off in range(0,len(cases),8):
            chunk=cases[off:off+8];inp=self.tok([self.render(c) for c in chunk],padding=True,return_tensors='pt',add_special_tokens=False)
            labels=[self.tok.encode(gold(c['spec'],c['value']),add_special_tokens=False) for c in chunk];assert all(len(x)==1 for x in labels)
            labels=torch.tensor([x[0] for x in labels])
            if teacher:
                inp['input_ids']=torch.cat([inp['input_ids'],labels[:,None]],1)
                inp['attention_mask']=torch.cat([inp['attention_mask'],torch.ones((len(chunk),1),dtype=torch.long)],1)
            cap=self.capture(inp,2 if teacher else 1);cap.update(cases=chunk,values=torch.tensor([c['value'] for c in chunk]),labels=labels,position=-2 if teacher else -1)
            result.append(cap)
        return result
    def suffix(self,batch,op=None,replacement=None):
        h=batch['hidden'];pos=batch['position']
        if op is not None or replacement is not None:
            h=h.clone();h[:,pos]=op(h[:,pos],batch['values']) if replacement is None else replacement
        for block in self.model.model.layers[self.layer+1:]:h=block(h,**batch['kwargs'])
        return self.model.lm_head(self.model.model.norm(h)[:,-(2 if pos==-2 else 1):])
    def hook(self,op,values):
        first=[True]
        def inject(module,ins,outs):
            if not first[0]:return outs
            first[0]=False;h=outs[0] if isinstance(outs,tuple) else outs
            h=h.clone();h[:,-1]=op(h[:,-1],values)
            return (h,)+outs[1:] if isinstance(outs,tuple) else h
        return inject
    def generate(self,cases,op=None,shift=False,text_memory=False):
        rows=[]
        for off in range(0,len(cases),16):
            chunk=cases[off:off+16];inp=self.tok([self.render(c,text_memory) for c in chunk],padding=True,return_tensors='pt',add_special_tokens=False)
            values=torch.tensor([(c['value']+int(shift))%4 for c in chunk]);handle=None if op is None else self.model.model.layers[self.layer].register_forward_hook(self.hook(op,values))
            try:
                with torch.no_grad():seq=self.model.generate(**inp,max_new_tokens=6,do_sample=False,use_cache=True,pad_token_id=self.eos)
            finally:
                if handle is not None:handle.remove()
            for c,ids in zip(chunk,seq[:,inp['input_ids'].shape[1]:].tolist()):
                end=next((i for i,t in enumerate(ids) if t in self.term),None);text=self.tok.decode(ids if end is None else ids[:end],skip_special_tokens=False)
                expected=gold(c['spec'],c['value']);shifted=gold(c['spec'],(c['value']+1)%4)
                rows.append(dict(case=c,token_ids=ids,text=text,eos_seen=end is not None,expected=expected,correct=bool(strict_ok(text,expected,end is not None)),shifted_expected=shifted,shifted_correct=bool(strict_ok(text,shifted,end is not None))))
        return dict(score=score(rows),rows=rows)
    @torch.no_grad()
    def repair(self,op,batches):
        pairs=[];sequence=[];final_agree=0;total=0
        for b in batches:
            base=b['hidden'][:,-1];total+=len(base)
            for old in [None,0,1,2,3]:
                previous=op(base,old)
                for new in [None,0,1,2,3]:
                    fresh=op(base,new);repaired=op.repair(previous,old,new)
                    rel=(repaired-fresh).norm(dim=-1)/fresh.norm(dim=-1).clamp_min(1e-12)
                    fl=self.suffix(b,replacement=fresh)[:,-1];rl=self.suffix(b,replacement=repaired)[:,-1]
                    pairs.append(dict(old=old,new=new,specs=[c['spec'] for c in b['cases']],n=len(base),case_relative_errors=rel.tolist(),max_relative_error=float(rel.max()),max_absolute_error=float((repaired-fresh).abs().max()),max_logit_error=float((fl-rl).abs().max()),top_agreement=int((fl.argmax(-1)==rl.argmax(-1)).sum())))
            state=base.clone();old=None;history=[]
            for i in range(256):
                new=[0,3,1,None,2][i%5];state=op.repair(state,old,new);old=new;fresh=op(base,new)
                rel=(state-fresh).norm(dim=-1)/fresh.norm(dim=-1).clamp_min(1e-12)
                history.append(float(rel.max()))
            fl=self.suffix(b,replacement=op(base,old))[:,-1];rl=self.suffix(b,replacement=state)[:,-1]
            final_agree+=int((fl.argmax(-1)==rl.argmax(-1)).sum())
            sequence.append(dict(specs=[c['spec'] for c in b['cases']],all_step_max_relative_errors=history,maximum=max(history),final=history[-1],final_max_logit_error=float((fl-rl).abs().max())))
        passed=all(r['max_relative_error']<1e-5 and r['top_agreement']==r['n'] for r in pairs) and max(r['maximum'] for r in sequence)<1e-5 and final_agree==total
        return dict(unique_queries=total,pairwise=pairs,sequence=sequence,sequence_top_agreement=final_agree,numerical_gate=bool(passed))
    @torch.no_grad()
    def autoreg_check(self,b,op):
        inp={k:v[:,:-1].clone() for k,v in b['inputs'].items()};handle=self.model.model.layers[self.layer].register_forward_hook(self.hook(op,b['values']))
        try:prefill=self.model(**inp,use_cache=True,logits_to_keep=1)
        finally:handle.remove()
        after=self.model(input_ids=b['labels'][:,None],attention_mask=b['inputs']['attention_mask'],past_key_values=prefill.past_key_values,use_cache=True,logits_to_keep=1)
        error=float((after.logits[:,-1]-self.suffix(b,op)[:,1]).abs().max())
        return dict(max_logit_error=error,threshold=1e-3,passed=error<1e-3)

def main(args):
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=True)
    source={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['run_recovered.py','cases.py','operators.py']}
    data=datasets();atomic_json(args.output/'cases.json',data)
    print('LOAD_MODEL',flush=True);engine=Boundary(args.model,args.layer)
    print('CACHE_TRAIN',flush=True);train=engine.batches(data['train'],True)
    print('CACHE_VALID',flush=True);validation_batch=engine.batches(data['valid'][:8])
    representative=[next(c for c in data['test'] if c['spec']==s) for s in sorted({c['spec'] for c in data['test']})]
    print('CACHE_REPAIR',flush=True);repair_batches=engine.batches(representative)
    bare=torch.cat([b['hidden'][:,b['position']] for b in train]);center=bare.mean(0)
    _,_,vt=torch.linalg.svd(bare-center,full_matrices=False);Q=vt[:16].T.contiguous();scale=((bare-center)@Q).square().mean(0).sqrt().clamp_min(1e-3)
    np.savez(args.output/'basis.npz',Q=Q.numpy(),center=center.numpy(),scale=scale.numpy())
    with torch.no_grad():
        errors=[float((engine.suffix(b)-b['reference']).abs().max()) for b in [train[0],validation_batch[0],repair_batches[0]]];assert max(errors)<1e-5
        inp={k:v.clone() for k,v in train[0]['inputs'].items()};inp['input_ids'][:,-1]=engine.tok.encode('square',add_special_tokens=False)[0]
        alt=engine.capture(inp,2);causal=float((alt['hidden'][:,-2]-train[0]['hidden'][:,-2]).abs().max());assert causal<1e-6
    execution=dict(split_logit_errors=errors,future_target_causal_error=causal)
    atomic_json(args.output/'execution_checks.json',execution);print('EXECUTION_CHECKS',execution,flush=True)
    results=[]
    for method in ['affine','bounded','coupling']:
        for seed in [17,29]:
            tag=f'{method}_seed{seed}';result_path=args.output/f'{tag}.json';cp=args.output/f'{tag}_resume.pt'
            if result_path.exists():
                r=json.loads(result_path.read_text());assert r['source_hashes']==source;results.append(r);print('REUSE_COMPLETE',tag,flush=True);continue
            torch.manual_seed(seed)
            if method=='coupling':op=SharedCouplingOperator(1024,16,4,Q=Q,center=center,scale=scale,hidden=64)
            else:
                op=SharedAffineOperator(1024,16,4,rho=.9 if method=='bounded' else None,Q=Q,center=center)
                with torch.no_grad():op.W.normal_(0,1e-4)
            optim=torch.optim.Adam(op.parameters(),lr=.02);sampler=torch.Generator().manual_seed(seed+991)
            start=0;history=[];best=None;best_state=None
            if cp.exists():
                saved=torch.load(cp,weights_only=True);assert saved['source_hashes']==source and saved['total_steps']==args.steps
                op.load_state_dict(saved['current']);optim.load_state_dict(saved['optimizer']);sampler.set_state(saved['sampler'])
                start=saved['step'];history=saved['history'];best=saved['best'];best_state=saved['best_state'];print('RESUME',tag,start,flush=True)
            t0=time.perf_counter()
            for step in range(start,args.steps+1):
                if step in [0,args.steps//3,2*args.steps//3,args.steps] and not any(r['step']==step for r in history):
                    g=engine.generate(data['valid'],op);s=g['score'];metric=[min(v['accuracy'] for v in s['operations'].values()),s['overall']['accuracy']]
                    record=dict(step=step,score=s,worst_family=metric[0]);history.append(record)
                    if best is None or tuple(metric)>tuple(best['metric']):best=dict(metric=metric,record=record);best_state=copy.deepcopy(op.state_dict())
                    atomic_json(args.output/f'{tag}_validation.json',history);print('VALID',tag,step,metric,flush=True)
                if step%20==0:
                    atomic_torch(cp,dict(source_hashes=source,total_steps=args.steps,step=step,current=op.state_dict(),optimizer=optim.state_dict(),sampler=sampler.get_state(),history=history,best=best,best_state=best_state))
                if step==args.steps:break
                batch=train[int(torch.randint(len(train),(1,),generator=sampler))];optim.zero_grad(set_to_none=True);logits=engine.suffix(batch,op)
                loss=torch.nn.functional.cross_entropy(logits[:,0],batch['labels'])+torch.nn.functional.cross_entropy(logits[:,1],torch.full_like(batch['labels'],engine.eos))
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(op.parameters(),1.0);optim.step()
                if step%40==0:print('LOSS',tag,step,float(loss.detach()),flush=True)
            op.load_state_dict(best_state);atomic_torch(args.output/f'{tag}_selected.pt',op.state_dict())
            generation=engine.generate(data['test'],op);shifted=engine.generate(data['test'],op,shift=True)
            eligible=[i for i,c in enumerate(data['test']) if gold(c['spec'],c['value'])"'!=gold(c['"'spec'],(c['value']+1)%4)]
            pairs=sum(generation['rows'][i]['correct'] and shifted['rows'][i]['shifted_correct'] for i in eligible)
            numerical=engine.repair(op,repair_batches);autoregressive=engine.autoreg_check(train[0],op)
            r=dict(method=method,seed=seed,layer=args.layer,steps=args.steps,source_hashes=source,selected=best['record'],history=history,generation=generation,shifted_generation=shifted,causal_pairs=dict(n=len(eligible),correct=pairs,accuracy=pairs/len(eligible)),numerical=numerical,autoregressive_check=autoregressive,diagnostics=op.diagnostics(),parameters=sum(p.numel() for p in op.parameters()),seconds_this_invocation=time.perf_counter()-t0)
            r['interface_gate_pass']=all(s['accuracy']>=.9 for s in generation['score']['operations'].values()) and pairs/len(eligible)>=.9 and numerical['numerical_gate'] and autoregressive['passed']
            atomic_json(result_path,r);results.append(r);atomic_json(args.output/'progress.json',dict(completed=[f\"{r['method']}_seed{r['seed']}\" for r in results],full_dod_pass=False))
            print('FINISHED',tag,json.dumps(generation['score']), 'CAUSAL',pairs,len(eligible),'REPAIR',numerical['numerical_gate'],'AUTOREG',autoregressive,flush=True)
    control_path=args.output/'controls.json'
    if control_path.exists():controls=json.loads(control_path.read_text())
    else:
        controls={}
        for name,text_memory in [('no_memory',False),('text_oracle',True)]:
            controls[name]=engine.generate(data['test'],text_memory=text_memory);print('CONTROL',name,controls[name]['score'],flush=True)
        atomic_json(control_path,controls)
    assert all(p.grad is None and not p.requires_grad for p in engine.model.parameters())
    atomic_json(args.output/'results.json',dict(experiment='QC-CR1',source_hashes=source,runs=results,controls=controls,execution=execution,backbone_frozen=True,parameter_count=sum(p.numel() for p in engine.model.parameters()),per_fact_gradients=0,unique_test_ids=4,counterfactual_cases=240,runtime=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,dtype='float32',cpu_threads=4),full_dod_pass=False))
    print('ALL_FINISHED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).parent/'run');p.add_argument('--layer',type=int,default=23);p.add_argument('--steps',type=int,default=240);main(p.parse_args())
