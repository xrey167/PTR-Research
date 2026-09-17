from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
P=2147483647
ROUNDS=40


def verify_product(t,output,rng):
    # Closed dyadic domain; no floating-point tolerance or modular alias allowed.
    vals={}
    for n,a in t.items():
        scaled=a*16
        if not np.isfinite(scaled).all() or np.max(np.abs(scaled))>32 or not np.array_equal(scaled,np.rint(scaled)):
            return False
        vals[n]=scaled.astype(np.int64)
    o=output*256
    # Conservative bound for all supported shapes and inputs: |output*256|<2^24.
    if not np.isfinite(o).all() or np.max(np.abs(o))>=2**24 or not np.array_equal(o,np.rint(o)):
        return False
    residual=o.astype(np.int64)-16*(vals['B']+vals['S']+vals['R'])
    shift=vals['Y']-vals['X'];c=vals['C'];d=shift.shape[1]
    z=rng.integers(0,2,size=(d,ROUNDS),dtype=np.int64)
    # Domain bounds keep every intermediate in signed int64 here.
    return bool(np.array_equal((c@(shift@z))%P,(residual@z)%P))


def main():
    out=HERE/'cert_results';out.mkdir(exist_ok=False)
    saved=np.load(HERE/'results/raw_tensors.npz',allow_pickle=False)
    rows=json.loads((HERE/'results/rows.json').read_text());checks=[]
    rng=np.random.default_rng(2026091401)
    for row in rows:
        seed=row['seed'];m,d,k=row['shape'];tag=f'{seed}_{m}_{d}_{k}'
        t={n:saved[tag+'_'+n] for n in ('B','S','R','C','Y','X')};o=saved[tag+'_output']
        bad=o.copy();bad[0,0]+=1/256
        alias=o.copy();alias[0,0]+=P/256
        r={'seed':seed,'shape':row['shape'],'fresh_accept':verify_product(t,o,rng),
          'unit_error_rejected':not verify_product(t,bad,rng),'modular_alias_rejected':not verify_product(t,alias,rng)}
        assert r['fresh_accept'] and r['unit_error_rejected'] and r['modular_alias_rejected']
        checks.append(r)
    costs=[]
    for name,m,k,d in [('TC1_all_pairs_local',15,6,256),('TC1_single_pair_local',1,6,256),('large_control',512,512,512)]:
        direct=m*k*d;verify=ROUNDS*(k*d+m*k+m*d)
        costs.append({'name':name,'m':m,'k':k,'d':d,'direct_mac_terms':direct,
                      'verify_mac_terms':verify,'verify_div_direct':verify/direct,'cost_gate':verify<direct})
    result={'checks':checks,'costs':costs,'rounds':ROUNDS,
      'soundness_bound_for_fixed_error':'<=2^-40 theoretical under secret independent post-commit challenges and valid bounds',
      'production_secret_challenges_implemented':False,'timing_claim':None,
      'tc1_cost_gate_pass':all(c['cost_gate'] for c in costs if c['name'].startswith('TC1')),
      'model_forwards':0,'full_dod_pass':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
