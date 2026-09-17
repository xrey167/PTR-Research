import json, time
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
rng=np.random.default_rng(212)
D=6

def orth():
    q,r=np.linalg.qr(rng.standard_normal((D,D)));s=np.sign(np.diag(r));s[s==0]=1;return q*s
# canonical generators: generic orthogonal matrices => scalar commutant w.h.p.
G=[orth() for _ in range(5)]
models={}
for name in ['A','B']:
    Q=orth(); T=[Q@g@Q.T for g in G]; models[name]=(Q,T)

def solve_intertwiner(Ts):
    # S P = P T. vec convention F; vec(SP)=(I kron S)vec(P), vec(PT)=(T.T kron I)vec(P)
    blocks=[];I=np.eye(D)
    for S,T in zip(G,Ts):blocks.append(np.kron(I,S)-np.kron(T.T,I))
    A=np.vstack(blocks)
    t=time.perf_counter_ns(); ns=null_space(A,rcond=1e-11); ms=(time.perf_counter_ns()-t)/1e6
    if ns.shape[1]<1:raise RuntimeError('no intertwiner')
    # pick first; reshape Fortran; normalize to nearest orthogonal (scale/sign irrelevant)
    P=ns[:,0].reshape((D,D),order='F')
    u,_,vt=np.linalg.svd(P); P=u@vt
    # orient determinant not relevant; test residual
    return P,ns.shape[1],ms,float(np.linalg.norm(A@P.reshape(-1,order='F'))/np.linalg.norm(P))
cal={}
for name,(Q,T) in models.items():cal[name]=solve_intertwiner(T)
# unseen knowledge operators are arbitrary words in generators and inverses
results={}
for name,(Q,T) in models.items():
    P,dim,ms,res=cal[name]
    errs=[];state_err=[]
    for L in [1,8,32,128,256]:
        le=[]
        for _ in range(100):
            word=[];Sprod=np.eye(D);Tprod=np.eye(D)
            for i in rng.integers(0,len(G),size=L):
                inv=bool(rng.integers(0,2));S=G[int(i)].T if inv else G[int(i)];Tm=T[int(i)].T if inv else T[int(i)]
                Sprod=S@Sprod;Tprod=Tm@Tprod
            pred=P@Tprod@P.T  # canonical action recovered from model coords
            le.append(np.linalg.norm(pred-Sprod)/np.linalg.norm(Sprod))
            x=rng.standard_normal(D); hm=P.T@x # canonical -> model coordinates because P model->canonical
            got=P@(Tprod@hm); exp=Sprod@x
            state_err.append(np.linalg.norm(got-exp)/(np.linalg.norm(exp)+1e-12))
        errs.append((L,float(max(le)),float(np.mean(le))))
    results[name]={'nullspace_dim':dim,'calibration_ms':ms,'equation_residual':res,'program_errors':errs,'max_state_relerr':float(max(state_err)),'P_vs_true_up_to_sign_relerr':float(min(np.linalg.norm(P-Q.T),np.linalg.norm(P+Q.T))/np.linalg.norm(Q))}
res={'stage':'R212','generator_count':len(G),'dimension':D,'models':results,'zero_fact_gradients':True,'calibration_scope':'global ABI only','pass':all(v['nullspace_dim']==1 and max(e[1] for e in v['program_errors'])<1e-10 for v in results.values())}
Path(__file__).with_name('r212_results.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
