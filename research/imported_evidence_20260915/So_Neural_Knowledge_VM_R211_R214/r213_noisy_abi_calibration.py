import json
from pathlib import Path
import numpy as np
from scipy.linalg import svd
rng=np.random.default_rng(213)
D=6

def orth():
 q,r=np.linalg.qr(rng.standard_normal((D,D)));s=np.sign(np.diag(r));s[s==0]=1;return q*s
G=[orth() for _ in range(7)]; Q=orth(); Ttrue=[Q@g@Q.T for g in G]

def nearest_orth(A):u,_,vt=np.linalg.svd(A);return u@vt

def estimate(Ts):
 I=np.eye(D);A=np.vstack([np.kron(I,S)-np.kron(T.T,I) for S,T in zip(G,Ts)])
 # smallest right singular vector
 _,sv,vh=svd(A,full_matrices=False);P=vh[-1].reshape((D,D),order='F');P=nearest_orth(P)
 return P,float(sv[-1]),float(sv[-2]/max(sv[-1],1e-30))

def program_err(P,L,trials=80):
 es=[]
 for _ in range(trials):
  Sp=np.eye(D);Tp=np.eye(D)
  for ii in rng.integers(0,len(G),L):
   inv=bool(rng.integers(0,2));S=G[int(ii)].T if inv else G[int(ii)];T=Ttrue[int(ii)].T if inv else Ttrue[int(ii)]
   Sp=S@Sp;Tp=T@Tp
  pred=P@Tp@P.T;es.append(np.linalg.norm(pred-Sp)/np.linalg.norm(Sp))
 return float(np.mean(es)),float(np.max(es))
rows=[]
for noise in [0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]:
 # perturb each observed generator then project back to orthogonal: models approximate action, structure preserved but correspondence noisy
 obs=[]
 for T in Ttrue:obs.append(nearest_orth(T+noise*rng.standard_normal(T.shape)))
 P,smin,gap=estimate(obs)
 maperr=float(min(np.linalg.norm(P-Q.T),np.linalg.norm(P+Q.T))/np.linalg.norm(Q))
 item={'noise':noise,'abi_map_relerr':maperr,'smallest_singular':smin,'spectral_separation_ratio':gap}
 for L in [1,16,64,256]:
  m,x=program_err(P,L);item[f'L{L}_mean']=m;item[f'L{L}_max']=x
 rows.append(item)
res={'stage':'R213','rows':rows,'observation':'deep error is controlled by global ABI calibration error; no per-fact gradients used','pass_condition':'informational sweep, not binary'}
Path(__file__).with_name('r213_results.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
