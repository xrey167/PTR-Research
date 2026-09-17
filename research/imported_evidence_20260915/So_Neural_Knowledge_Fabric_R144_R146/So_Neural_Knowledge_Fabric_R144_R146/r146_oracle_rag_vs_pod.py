import sys,time,json,random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0,'/mnt/data/So_Neural_Knowledge_Fabric_R137_R143')
from r137_pod_native_fabric import PodFabric,N,R,random_world
OUT=Path('/mnt/data/So_Neural_Knowledge_Fabric_R144_R146'); OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
model=PodFabric(); ck=torch.load('/mnt/data/So_Neural_Knowledge_Fabric_R137_R143/r137_pod_native_fabric.pt',map_location='cpu'); model.load_state_dict(ck['model']); model.eval()
np.random.seed(146); torch.manual_seed(146); random.seed(146)
w=random_world(); active=torch.ones((N,R),dtype=torch.bool); keys,vals,_=model.compile_memory(torch.tensor(w),active)
S=N*R
# Exact oracle address distribution, Zipf-like hot set.
ranks=np.arange(1,S+1,dtype=float); probs=1/(ranks**1.15); probs/=probs.sum()
perm=np.random.permutation(S); probs2=np.zeros(S); probs2[perm]=probs

def sample_slots(B): return torch.tensor(np.random.choice(S,size=B,p=probs2),dtype=torch.long)

def truth_obj(slots,mapping):
    s=(slots//R).numpy(); r=(slots%R).numpy(); return torch.tensor(mapping[s,r],dtype=torch.long)

@torch.no_grad()
def pod_read(slots,vals):
    z=vals[slots]; return model.decoder(z).argmax(-1)
@torch.no_grad()
def oracle_rag_read(slots,mapping):
    # Strong lower-bound baseline: retrieval is exact and already returns structured object id.
    # It still has to transform the retrieved record into the model-specific neural representation on every read.
    o=truth_obj(slots,mapping)
    vraw=F.one_hot(o,num_classes=N+1).float(); z=model.value_enc(vraw); return model.decoder(z).argmax(-1)

def bench(fn,args,reps=400,warm=40):
    for _ in range(warm): fn(*args)
    ts=[]
    for _ in range(reps):
        t=time.perf_counter_ns(); fn(*args); ts.append((time.perf_counter_ns()-t)/1e6)
    return {'median_ms':float(np.median(ts)),'mean_ms':float(np.mean(ts)),'p95_ms':float(np.quantile(ts,.95))}

def main():
    results={}
    for B in [1,32,512,4096]:
        slots=sample_slots(B); truth=truth_obj(slots,w)
        pp=pod_read(slots,vals); rr=oracle_rag_read(slots,w)
        assert torch.equal(pp,truth) and torch.equal(rr,truth)
        # interleaved manually for fairness
        for _ in range(20): pod_read(slots,vals); oracle_rag_read(slots,w)
        p=[]; r=[]
        for i in range(240):
            order=[('p',pod_read,(slots,vals)),('r',oracle_rag_read,(slots,w))]
            if i%2: order.reverse()
            for name,fn,args in order:
                t=time.perf_counter_ns(); fn(*args); dt=(time.perf_counter_ns()-t)/1e6; (p if name=='p' else r).append(dt)
        results[str(B)]={'pod_median_ms':float(np.median(p)),'oracle_structured_rag_median_ms':float(np.median(r)),'speedup':float(np.median(r)/np.median(p)),'accuracy':1.0}
    # lifecycle update: old precompiled row becomes invalid, new row recompile only once
    slot=int(perm[0]); s=slot//R; rel=slot%R; old=int(w[s,rel]); new=(old+1)%N
    w2=w.copy(); w2[s,rel]=new
    vals2=vals.clone(); vraw=F.one_hot(torch.tensor([new]),num_classes=N+1).float(); vals2[slot]=model.value_enc(vraw)[0]
    q=torch.tensor([slot]); update_correct=int(pod_read(q,vals2)[0])==new
    stale_wrong=int(pod_read(q,vals)[0])==old
    # hybrid LRU analytical + measured hit rate: top-C neural images resident; cold misses use exact RAG path.
    Q=100000; qs=np.random.choice(S,size=Q,p=probs2)
    hybrid={}
    for C in [4,8,16,32,64]:
        hot=set(np.argsort(probs2)[-C:].tolist()); hit=float(np.mean([int(x in hot) for x in qs]))
        # Expected per-record cost from B=1 medians; conservative no batching.
        pr=results['1']['pod_median_ms']; rg=results['1']['oracle_structured_rag_median_ms']
        exp=hit*pr+(1-hit)*rg
        hybrid[str(C)]={'capacity':C,'hit_rate':hit,'expected_ms_per_read':exp,'speedup_vs_rag':rg/exp}
    out={'baseline_strength':'oracle exact-address structured RAG: no retrieval errors and no text parsing; only repeated model-specific compilation remains','results':results,'hot_update':{'slot':slot,'old':old,'new':new,'new_image_correct':bool(update_correct),'old_image_still_encodes_old':bool(stale_wrong),'runtime_rule':'old image is keyed by old revision and cannot authorize under new snapshot'},'zipf_hybrid':hybrid,'interpretation':'If a RAG system precompiles and reuses model-specific latent records, it has moved into the neural-memory design space (NeuralDB/Larimar/our Pods). The Fabric therefore treats RAG as cold backing store and Pods as resident compiled cache, rather than claiming universal accuracy dominance.'}
    json.dump(out,open(OUT/'r146_oracle_rag_vs_pod.json','w'),indent=2); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
