import sys, time, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,'/mnt/data/So_Neural_Knowledge_Fabric_R137_R143')
from r137_pod_native_fabric import PodFabric, N,R, random_world
OUT=Path('/mnt/data/So_Neural_Knowledge_Fabric_R144_R146'); OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
model=PodFabric(); sd=torch.load('/mnt/data/So_Neural_Knowledge_Fabric_R137_R143/r137_pod_native_fabric.pt',map_location='cpu'); model.load_state_dict(sd['model'] if isinstance(sd,dict) and 'model' in sd else sd); model.eval()
class Branch:
    def __init__(self,mapping,active):
        self.mapping=mapping.copy(); self.active=active.copy(); self.epoch=1
        self.keys,self.vals,_=model.compile_memory(torch.tensor(self.mapping),torch.tensor(self.active))
    def lease(self): return (self.epoch,self.keys,self.vals)
    def edit(self,s,r,o):
        self.mapping=self.mapping.copy(); self.active=self.active.copy(); self.mapping[s,r]=o; self.active[s,r]=True; self.epoch+=1
        slot=s*R+r; vr=F.one_hot(torch.tensor([o]),num_classes=N+1).float(); newv=model.value_enc(vr).detach()[0]
        vals=self.vals.clone(); vals[slot]=newv; self.vals=vals
    def revoke(self,s,r):
        self.mapping=self.mapping.copy(); self.active=self.active.copy(); self.active[s,r]=False; self.epoch+=1
        slot=s*R+r; vr=F.one_hot(torch.tensor([N]),num_classes=N+1).float(); newv=model.value_enc(vr).detach()[0]
        vals=self.vals.clone(); vals[slot]=newv; self.vals=vals
@torch.no_grad()
def run_base(keys,vals,starts,rels):
    cur=F.one_hot(starts,num_classes=N).float(); p=None
    for t in range(rels.shape[1]):
        logits,_,_=model.hop(cur,rels[:,t],keys,vals); p=logits.argmax(-1); cur=F.one_hot(torch.clamp(p,max=N-1),num_classes=N).float()
    return p
@torch.no_grad()
def run_fast(branch,lease,starts,rels):
    epoch,keys,vals=lease
    if epoch!=branch.epoch: raise RuntimeError('stale lease')
    cur=F.one_hot(starts,num_classes=N).float(); p=None
    for t in range(rels.shape[1]):
        if epoch!=branch.epoch: raise RuntimeError('revoked during request')
        logits,_,_=model.hop(cur,rels[:,t],keys,vals); p=logits.argmax(-1); cur=F.one_hot(torch.clamp(p,max=N-1),num_classes=N).float()
    return p
@torch.no_grad()
def run_naive(branch,lease,starts,rels):
    epoch,keys,vals=lease; cur=F.one_hot(starts,num_classes=N).float(); p=None; b=len(starts)
    for t in range(rels.shape[1]):
        ids=cur.argmax(-1).tolist(); rr=rels[:,t].tolist()
        for i in range(b):
            s=int(ids[i]); r=int(rr[i]); _=bool(branch.active[s,r]); _g=(s,r,branch.epoch)
        logits,_,_=model.hop(cur,rels[:,t],keys,vals); p=logits.argmax(-1); cur=F.one_hot(torch.clamp(p,max=N-1),num_classes=N).float()
    return p

def interleaved_bench(base_fn,base_args,test_fn,test_args,reps=300,warm=30):
    for _ in range(warm): base_fn(*base_args); test_fn(*test_args)
    b=[]; t=[]
    for i in range(reps):
        order=((base_fn,base_args,b),(test_fn,test_args,t)) if i%2==0 else ((test_fn,test_args,t),(base_fn,base_args,b))
        for fn,args,out in order:
            s=time.perf_counter_ns(); fn(*args); out.append((time.perf_counter_ns()-s)/1e6)
    def stats(x): return {'median_ms':float(np.median(x)),'mean_ms':float(np.mean(x)),'p95_ms':float(np.quantile(x,.95))}
    return stats(b),stats(t)

def main():
    np.random.seed(145); torch.manual_seed(145); random.seed(145)
    w=random_world(); branch=Branch(w,np.ones((N,R),bool)); lease=branch.lease(); B=512; H=6
    starts=torch.randint(0,N,(B,)); rels=torch.randint(0,R,(B,H)); assert torch.equal(run_base(branch.keys,branch.vals,starts,rels),run_fast(branch,lease,starts,rels))
    base,fast=interleaved_bench(run_base,(branch.keys,branch.vals,starts,rels),run_fast,(branch,lease,starts,rels))
    base2,naive=interleaved_bench(run_base,(branch.keys,branch.vals,starts,rels),run_naive,(branch,lease,starts,rels),reps=120,warm=15)
    fast_over=(fast['median_ms']/base['median_ms']-1)*100; naive_over=(naive['median_ms']/base2['median_ms']-1)*100
    edits=[(random.randrange(N),random.randrange(R),random.randrange(N)) for _ in range(800)]
    br=Branch(w,np.ones((N,R),bool)); s=time.perf_counter_ns()
    for a,r,o in edits: br.edit(a,r,o)
    partial=(time.perf_counter_ns()-s)/1e6/len(edits)
    map2=w.copy(); act=np.ones((N,R),bool); s=time.perf_counter_ns()
    for a,r,o in edits: map2[a,r]=o; model.compile_memory(torch.tensor(map2),torch.tensor(act))
    full=(time.perf_counter_ns()-s)/1e6/len(edits)
    old=branch.lease(); branch.edit(0,0,(int(w[0,0])+1)%N)
    rejected=False
    try: run_fast(branch,old,starts[:8],rels[:8])
    except RuntimeError: rejected=True
    new=branch.lease(); correct=torch.equal(run_fast(branch,new,starts[:8],rels[:8]),run_base(branch.keys,branch.vals,starts[:8],rels[:8]))
    out={'threads':1,'batch':B,'hops':H,'base_interleaved':base,'snapshot_fastpath':fast,'fastpath_overhead_pct_median':fast_over,'base_for_naive':base2,'naive_per_record':naive,'naive_overhead_pct_median':naive_over,'mutation_partial_ms_per_edit':partial,'mutation_full_recompile_ms_per_edit':full,'mutation_speedup':full/partial,'stale_lease_rejected':rejected,'new_lease_correct':bool(correct),'hotpath_design':['pin immutable snapshot once per request','one scalar branch epoch comparison per neural step','Neural Images addressed by snapshot/revision so no per-use authorization','derived caches keyed by snapshot/dep digest; invalidation happens at mutation boundary','compile only changed Pod value row']}
    json.dump(out,open(OUT/'r145_lifecycle_fastpath.json','w'),indent=2); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
