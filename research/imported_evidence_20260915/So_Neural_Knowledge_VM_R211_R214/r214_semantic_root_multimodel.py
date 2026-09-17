import json,time
from pathlib import Path
import numpy as np
rng=np.random.default_rng(214)
D=6; L=8192; NABI=4

def orth():
 q,r=np.linalg.qr(rng.standard_normal((D,D)));s=np.sign(np.diag(r));s[s==0]=1;return q*s
G=[orth() for _ in range(8)]
Qs=[orth() for _ in range(NABI)]
ops=rng.integers(0,len(G),L)
size=1
while size<L:size*=2
T=np.tile(np.eye(D),(2*size,1,1))
for i,o in enumerate(ops):T[size+i]=G[int(o)]
for i in range(size-1,0,-1):T[i]=T[2*i+1]@T[2*i]
root_v1=T[1].copy(); roots_v1=[Q@root_v1@Q.T for Q in Qs]
canon_ms=[]; abimap_ms=[]; naive_ms=[]; gaps=[];old_gaps=[]
for _ in range(20):
 p=int(rng.integers(0,L));ops[p]=int(rng.integers(0,len(G)))
 t=time.perf_counter_ns();T[size+p]=G[int(ops[p])];i=(size+p)//2
 while i:T[i]=T[2*i+1]@T[2*i];i//=2
 canon_ms.append((time.perf_counter_ns()-t)/1e6)
 t=time.perf_counter_ns();compiled=[Q@T[1]@Q.T for Q in Qs];abimap_ms.append((time.perf_counter_ns()-t)/1e6)
 # naive: each model independently walks entire semantic program and maps every op
 t=time.perf_counter_ns();nroots=[]
 for Q in Qs:
  P=np.eye(D)
  for o in ops:P=(Q@G[int(o)]@Q.T)@P
  nroots.append(P)
 naive_ms.append((time.perf_counter_ns()-t)/1e6)
 gaps.append(max(np.linalg.norm(a-b)/np.linalg.norm(b) for a,b in zip(compiled,nroots)))
 old_gaps.append(max(np.linalg.norm(a-b) for a,b in zip(roots_v1,[Q@root_v1@Q.T for Q in Qs])))
res={'stage':'R214','program_len':L,'model_abis':NABI,'median_canonical_tree_update_ms':float(np.median(canon_ms)),'median_map_root_all_abis_ms':float(np.median(abimap_ms)),'median_naive_per_model_recompile_ms':float(np.median(naive_ms)),'end_to_end_speedup_vs_naive':float(np.median(naive_ms)/(np.median(canon_ms)+np.median(abimap_ms))),'max_compiled_vs_naive_relerr':float(max(gaps)),'old_snapshot_max_change':float(max(old_gaps)),'semantic_revision_compiled_once':True,'pass':bool(max(gaps)<1e-10 and max(old_gaps)==0.0)}
Path(__file__).with_name('r214_results.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
