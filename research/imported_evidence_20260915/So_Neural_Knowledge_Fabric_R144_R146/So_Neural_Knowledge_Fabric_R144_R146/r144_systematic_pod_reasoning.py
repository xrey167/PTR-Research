import sys, json, random, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,'/mnt/data/So_Neural_Knowledge_Fabric_R137_R143')
from r137_pod_native_fabric import PodFabric, KnowledgeSnapshot, random_world, alias_centroids, sample_alias, N, R, ALIAS_D, REVOKED

torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
OUT=Path('/mnt/data/So_Neural_Knowledge_Fabric_R144_R146')
OUT.mkdir(exist_ok=True)

HELD_REL={(0,3),(1,2),(2,1),(3,0)}

def held_value_pair(a,b):
    # deterministic ~20% of ordered identity pairs, spread across all identities
    return ((a*7 + b*11 + 3) % 5)==0

def sample_seq(T, rng, training=True, force_held_rel=False):
    for _ in range(1000):
        rs=[rng.randrange(R) for _ in range(T)]
        has=any((rs[i],rs[i+1]) in HELD_REL for i in range(T-1))
        if force_held_rel and has: return rs
        if training and not has: return rs
        if not training and not force_held_rel: return rs
    raise RuntimeError('seq sample')

def make_batch(mapping,batch,rng,max_hops=3,training=True,force_held_rel=False,force_held_value=False,fixed_hops=None):
    starts=[]; rels=[]; targets=[]; slots=[]
    attempts=0
    while len(starts)<batch:
        attempts+=1
        if attempts>batch*10000: raise RuntimeError('batch rejection')
        s=rng.randrange(N); T=fixed_hops or rng.randint(1,max_hops)
        rs=sample_seq(T,rng,training=training,force_held_rel=force_held_rel)
        cur=s; targs=[]; sl=[]; valheld=False
        ok=True
        for r in rs:
            nxt=int(mapping[cur,r]);
            if held_value_pair(cur,nxt): valheld=True
            if training and held_value_pair(cur,nxt): ok=False; break
            sl.append(cur*R+r); targs.append(nxt); cur=nxt
        if not ok: continue
        if force_held_value and not valheld: continue
        starts.append(s); rels.append(rs); targets.append(targs); slots.append(sl)
    T=max(map(len,rels))
    ra=np.zeros((batch,T),dtype=np.int64); ta=np.full((batch,T),-100,dtype=np.int64); sa=np.full((batch,T),-100,dtype=np.int64); hops=[]
    for i,(rs,ts,ss) in enumerate(zip(rels,targets,slots)):
        ra[i,:len(rs)]=rs; ta[i,:len(ts)]=ts; sa[i,:len(ss)]=ss; hops.append(len(rs))
    return np.array(starts),np.array(hops),ra,ta,sa

def train_seed(seed,steps=500):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng=random.Random(seed+10000)
    model=PodFabric(); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4)
    for step in range(steps):
        w=random_world()
        starts,hops,rels,targs,slots=make_batch(w,128,rng,max_hops=3,training=True)
        keys,vals,_=model.compile_memory(torch.tensor(w),torch.ones((N,R),dtype=torch.bool))
        alias=sample_alias(torch.tensor(starts))
        al=model.resolve_alias(alias); start_loss=F.cross_entropy(al,torch.tensor(starts))
        cur=F.one_hot(al.argmax(-1),num_classes=N).float()
        loss=start_loss
        for t in range(rels.shape[1]):
            logits,scores,_=model.hop(cur,torch.tensor(rels[:,t]),keys,vals)
            trg=torch.tensor(targs[:,t]); slot=torch.tensor(slots[:,t]); mask=trg!=-100
            if mask.any():
                loss=loss+F.cross_entropy(logits[mask],trg[mask])+0.35*F.cross_entropy(scores[mask],slot[mask])
            pred=logits.argmax(-1); cur=F.one_hot(torch.clamp(pred,max=N-1),num_classes=N).float()
        # tombstones, as before
        ta=torch.ones((N,R),dtype=torch.bool); pairs=[]
        for _ in range(8):
            s=rng.randrange(N); r=rng.randrange(R); ta[s,r]=False; pairs.append((s,r))
        tk,tv,_=model.compile_memory(torch.tensor(w),ta)
        ss=torch.tensor([x[0] for x in pairs]); rr=torch.tensor([x[1] for x in pairs])
        lg,sc,_=model.hop(F.one_hot(ss,num_classes=N).float(),rr,tk,tv)
        loss=loss+F.cross_entropy(lg,torch.full((len(pairs),),REVOKED,dtype=torch.long))+0.2*F.cross_entropy(sc,ss*R+rr)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
    return model

@torch.no_grad()
def evaluate(model,seed,mode,worlds=20,per_world=160,hops=2):
    rng=random.Random(seed+90000+hash(mode)%10000)
    acc=[]; route=[]
    for wi in range(worlds):
        np.random.seed(seed*1000+wi+500000); w=random_world()
        kw={}
        if mode=='seen': kw=dict(training=False,force_held_rel=False,force_held_value=False)
        elif mode=='held_rel': kw=dict(training=False,force_held_rel=True,force_held_value=False)
        elif mode=='held_value': kw=dict(training=False,force_held_rel=False,force_held_value=True)
        elif mode=='held_both': kw=dict(training=False,force_held_rel=True,force_held_value=True)
        elif mode=='long_8': kw=dict(training=False,force_held_rel=True,force_held_value=True); hops=8
        elif mode=='long_16': kw=dict(training=False,force_held_rel=True,force_held_value=True); hops=16
        starts,hs,rels,targs,slots=make_batch(w,per_world,rng,fixed_hops=hops,**kw)
        keys,vals,_=model.compile_memory(torch.tensor(w),torch.ones((N,R),dtype=torch.bool))
        al=model.resolve_alias(sample_alias(torch.tensor(starts)))
        cur=F.one_hot(al.argmax(-1),num_classes=N).float()
        for t in range(hops):
            lg,sc,_=model.hop(cur,torch.tensor(rels[:,t]),keys,vals)
            pred=lg.argmax(-1); trg=torch.tensor(targs[:,t])
            if t==hops-1: acc.extend((pred==trg).float().tolist())
            top=sc.argmax(-1); route.extend((top==torch.tensor(slots[:,t])).float().tolist())
            cur=F.one_hot(torch.clamp(pred,max=N-1),num_classes=N).float()
    return {'accuracy':float(np.mean(acc)),'route_accuracy':float(np.mean(route)),'n':len(acc)}

def main():
    seeds=[1441,1442,1443,1444,1445]
    allr=[]; t0=time.time()
    for s in seeds:
        print('train',s,flush=True); m=train_seed(s)
        r={'seed':s}
        for mode,h in [('seen',2),('held_rel',2),('held_value',2),('held_both',2),('long_8',8),('long_16',16)]:
            r[mode]=evaluate(m,s,mode,hops=h)
            print(s,mode,r[mode],flush=True)
        allr.append(r)
    summary={}
    for mode in ['seen','held_rel','held_value','held_both','long_8','long_16']:
        vals=[x[mode]['accuracy'] for x in allr]; routes=[x[mode]['route_accuracy'] for x in allr]
        summary[mode]={'mean_accuracy':float(np.mean(vals)),'min_accuracy':float(np.min(vals)),'mean_route_accuracy':float(np.mean(routes)),'per_seed':vals}
    out={'design':{'held_relation_bigrams':sorted(map(list,HELD_REL)),'held_value_rule':'((a*7+b*11+3)%5)==0','training':'new random world each step; no training query may traverse held value pair or held relation bigram','steps_per_seed':500,'seeds':seeds},'runs':allr,'summary':summary,'seconds':time.time()-t0}
    json.dump(out,open(OUT/'r144_systematic_pod_reasoning.json','w'),indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
