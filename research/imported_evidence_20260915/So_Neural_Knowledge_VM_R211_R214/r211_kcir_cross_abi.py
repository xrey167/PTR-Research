import json, math, time
from pathlib import Path
import numpy as np

SEED=211
rng=np.random.default_rng(SEED)
SEM_D=5  # homogeneous 4D semantic state + 1

def qorth(d):
    q,r=np.linalg.qr(rng.standard_normal((d,d)))
    s=np.sign(np.diag(r)); s[s==0]=1
    return q*s.reshape(1,-1)

def semantic_op(kind, coeff):
    A=np.eye(SEM_D)
    if kind==0: # rotation plane 0/1
        th=coeff
        c,s=math.cos(th),math.sin(th)
        A[0,0]=c;A[0,1]=-s;A[1,0]=s;A[1,1]=c
    elif kind==1: # rotation 2/3
        th=coeff
        c,s=math.cos(th),math.sin(th)
        A[2,2]=c;A[2,3]=-s;A[3,2]=s;A[3,3]=c
    elif kind==2: # typed scale on semantic coordinate, bounded/invertible
        a=math.exp(0.25*coeff)
        A[0,0]=a;A[1,1]=1/a
    elif kind==3: # translation in homogeneous coordinates
        A[0,4]=coeff
    elif kind==4:
        A[2,4]=coeff
    else: raise ValueError
    return A

# two different model ABIs: same canonical semantic algebra embedded in larger spaces
abis={}
for name,d in [('A',12),('B',18)]:
    Q=qorth(d)
    def compile_fn(A,Q=Q,d=d):
        B=np.eye(d); B[:SEM_D,:SEM_D]=A
        return Q @ B @ Q.T
    def encode_fn(y,Q=Q,d=d):
        z=np.zeros(d); z[:SEM_D]=y
        return Q @ z
    def decode_fn(h,Q=Q):
        return (Q.T@h)[:SEM_D]
    abis[name]=(d,Q,compile_fn,encode_fn,decode_fn)

# post-training knowledge objects are only exact kind+coefficient pages
N_OBJECTS=100_000
kinds=rng.integers(0,5,size=N_OBJECTS)
coeffs=rng.uniform(-1.0,1.0,size=N_OBJECTS)

# 1) unseen object single-step cross-ABI semantic agreement
single_err=[]
for idx in rng.choice(N_OBJECTS,size=2000,replace=False):
    A=semantic_op(int(kinds[idx]),float(coeffs[idx]))
    y=rng.standard_normal(SEM_D); y[4]=1.0
    expected=A@y
    for name,(d,Q,comp,enc,dec) in abis.items():
        got=dec(comp(A)@enc(y))
        single_err.append(np.linalg.norm(got-expected)/(np.linalg.norm(expected)+1e-12))

# 2) long programs, new combinations, cross-model equivalence
lengths=[1,8,32,128,512]
long={}
for L in lengths:
    errs=[]; cross=[]
    for _ in range(120):
        ids=rng.integers(0,N_OBJECTS,size=L)
        Aprod=np.eye(SEM_D)
        for idx in ids: Aprod=semantic_op(int(kinds[idx]),float(coeffs[idx]))@Aprod
        y=rng.standard_normal(SEM_D); y[4]=1.0
        expected=Aprod@y
        outs=[]
        for name,(d,Q,comp,enc,dec) in abis.items():
            M=np.eye(d)
            for idx in ids: M=comp(semantic_op(int(kinds[idx]),float(coeffs[idx])))@M
            out=dec(M@enc(y)); outs.append(out)
            errs.append(np.linalg.norm(out-expected)/(np.linalg.norm(expected)+1e-12))
        cross.append(np.linalg.norm(outs[0]-outs[1])/(np.linalg.norm(expected)+1e-12))
    long[str(L)]={'max_semantic_relerr':float(np.max(errs)),'mean_semantic_relerr':float(np.mean(errs)),'max_cross_abi_relerr':float(np.max(cross))}

# 3) version edit: replace one position in 4096-op program, product tree update cost vs full rebuild
L=4096
ids=rng.integers(0,N_OBJECTS,size=L)
pkinds=kinds[ids].copy(); pcoeffs=coeffs[ids].copy()
size=1
while size<L:size*=2
T=np.tile(np.eye(SEM_D),(2*size,1,1))
for i in range(L):T[size+i]=semantic_op(int(pkinds[i]),float(pcoeffs[i]))
for i in range(size-1,0,-1):T[i]=T[2*i+1]@T[2*i]
update_times=[]; full_times=[]; max_gap=0.0; touched=[]
for _ in range(200):
    p=int(rng.integers(0,L)); newk=int(rng.integers(0,5)); newc=float(rng.uniform(-1,1))
    pkinds[p]=newk; pcoeffs[p]=newc
    t=time.perf_counter_ns()
    T[size+p]=semantic_op(newk,newc); i=(size+p)//2; n=1
    while i:
        T[i]=T[2*i+1]@T[2*i]; i//=2; n+=1
    update_times.append((time.perf_counter_ns()-t)/1e6); touched.append(n)
    t=time.perf_counter_ns(); P=np.eye(SEM_D)
    for j in range(L):P=semantic_op(int(pkinds[j]),float(pcoeffs[j]))@P
    full_times.append((time.perf_counter_ns()-t)/1e6)
    max_gap=max(max_gap,float(np.max(np.abs(P-T[1]))))

res={
 'stage':'R211','objects':N_OBJECTS,'semantic_dim':SEM_D,'abis':{k:v[0] for k,v in abis.items()},
 'single_max_relerr':float(np.max(single_err)),'single_mean_relerr':float(np.mean(single_err)),
 'long_programs':long,
 'product_tree':{'program_len':L,'median_update_ms':float(np.median(update_times)),'median_full_ms':float(np.median(full_times)),'speedup':float(np.median(full_times)/np.median(update_times)),'max_matrix_gap':max_gap,'median_nodes_touched':float(np.median(touched))},
 'zero_per_object_gradients':True,'object_payload':'typed opcode + exact coefficient','pass':bool(np.max(single_err)<1e-12 and max(v['max_semantic_relerr'] for v in long.values())<1e-10 and max_gap<1e-10)
}
Path(__file__).with_name('r211_results.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
