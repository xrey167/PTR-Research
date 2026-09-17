import json, math, time
from pathlib import Path
import numpy as np
rng=np.random.default_rng(2111)
D=6

def givens(i,j,theta):
    A=np.eye(D); c,s=np.cos(theta),np.sin(theta)
    A[i,i]=c;A[i,j]=-s;A[j,i]=s;A[j,j]=c
    return A
planes=[(0,1),(2,3),(4,5),(0,2),(1,4),(3,5)]

def op(kind,coef): return givens(*planes[kind], 0.35*coef)

def qorth(d):
    q,r=np.linalg.qr(rng.standard_normal((d,d))); s=np.sign(np.diag(r));s[s==0]=1
    return q*s.reshape(1,-1)
abis={}
for name,d in [('A',12),('B',18)]:
    Q=qorth(d)
    def comp(A,Q=Q,d=d):
        B=np.eye(d);B[:D,:D]=A;return Q@B@Q.T
    def enc(x,Q=Q,d=d):z=np.zeros(d);z[:D]=x;return Q@z
    def dec(h,Q=Q):return (Q.T@h)[:D]
    abis[name]=(d,comp,enc,dec)
N=100_000;kinds=rng.integers(0,len(planes),N);coef=rng.uniform(-1,1,N)
# long programs semantic and cross ABI
long={}
for L in [1,8,32,128,512,2048]:
    errs=[];cross=[];orth=[]
    for _ in range(40):
        ids=rng.integers(0,N,L);P=np.eye(D)
        for idx in ids:P=op(int(kinds[idx]),float(coef[idx]))@P
        x=rng.standard_normal(D);truth=P@x;outs=[]
        for _,(d,comp,enc,dec) in abis.items():
            M=np.eye(d)
            for idx in ids:M=comp(op(int(kinds[idx]),float(coef[idx])))@M
            y=dec(M@enc(x));outs.append(y);errs.append(np.linalg.norm(y-truth)/(np.linalg.norm(truth)+1e-12))
        cross.append(np.linalg.norm(outs[0]-outs[1])/(np.linalg.norm(truth)+1e-12));orth.append(np.linalg.norm(P.T@P-np.eye(D)))
    long[str(L)]={'max_relerr':float(max(errs)),'mean_relerr':float(np.mean(errs)),'max_cross':float(max(cross)),'max_orth_defect':float(max(orth))}
# segment tree edit
L=4096;ids=rng.integers(0,N,L);pk=kinds[ids].copy();pc=coef[ids].copy();size=1
while size<L:size*=2
T=np.tile(np.eye(D),(2*size,1,1))
for i in range(L):T[size+i]=op(int(pk[i]),float(pc[i]))
for i in range(size-1,0,-1):T[i]=T[2*i+1]@T[2*i]
updates=[];full=[];gaps=[]
for _ in range(100):
    p=int(rng.integers(0,L));pk[p]=int(rng.integers(0,len(planes)));pc[p]=float(rng.uniform(-1,1))
    t=time.perf_counter_ns();T[size+p]=op(int(pk[p]),float(pc[p]));i=(size+p)//2
    while i:T[i]=T[2*i+1]@T[2*i];i//=2
    updates.append((time.perf_counter_ns()-t)/1e6)
    t=time.perf_counter_ns();P=np.eye(D)
    for j in range(L):P=op(int(pk[j]),float(pc[j]))@P
    full.append((time.perf_counter_ns()-t)/1e6);gaps.append(np.linalg.norm(T[1]-P)/np.linalg.norm(P))
res={'stage':'R211b','objects':N,'abis':{'A':12,'B':18},'exact_lane_policy':'IDs/numbers/time/revisions never enter long floating neural product','neural_algebra':'orthogonal Givens operators','long':long,'tree':{'median_update_ms':float(np.median(updates)),'median_full_ms':float(np.median(full)),'speedup':float(np.median(full)/np.median(updates)),'max_relative_product_gap':float(max(gaps))},'pass':bool(max(v['max_relerr'] for v in long.values())<1e-10 and max(gaps)<1e-10)}
Path(__file__).with_name('r211b_results.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
