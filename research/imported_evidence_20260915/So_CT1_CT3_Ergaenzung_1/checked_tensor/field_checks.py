from pathlib import Path
import json
import numpy as np
from check_product import verify_product,P

HERE=Path(__file__).resolve().parent


class FixedZeros:
    def integers(self,low,high,size,dtype):return np.zeros(size,dtype=dtype)


def field_verify(t,output,rng):
    # Reuse strict domain validation. Zero vectors make this phase a bounds check.
    if not verify_product(t,output,FixedZeros()):return False
    v={n:(x*16).astype(np.int64) for n,x in t.items()}
    residual=(output*256).astype(np.int64)-16*(v['B']+v['S']+v['R'])
    z=rng.integers(0,P,size=(v['X'].shape[1],2),dtype=np.int64)
    return bool(np.array_equal((v['C']@((v['Y']-v['X'])@z %P))%P,(residual@z)%P))


def main():
    out=HERE/'field_results';out.mkdir(exist_ok=False)
    saved=np.load(HERE/'results/raw_tensors.npz',allow_pickle=False)
    rows=json.loads((HERE/'results/rows.json').read_text());rng=np.random.default_rng(2026091402);checks=[]
    for row in rows:
        seed=row['seed'];m,d,k=row['shape'];tag=f'{seed}_{m}_{d}_{k}'
        t={n:saved[tag+'_'+n] for n in ('B','S','R','C','Y','X')};o=saved[tag+'_output']
        bad=o.copy();bad[0,0]+=1/256;alias=o.copy();alias[0,0]+=P/256
        check=dict(seed=seed,shape=row['shape'],fresh=field_verify(t,o,rng),
          wrong_rejected=not field_verify(t,bad,rng),alias_rejected=not field_verify(t,alias,rng))
        assert check['fresh'] and check['wrong_rejected'] and check['alias_rejected'];checks.append(check)
    costs=[]
    for m in (1,3,15):
        k,d=6,256;direct=m*k*d;verification=2*(k*d+m*k+m*d)
        costs.append(dict(pair_count=m,direct_mac_terms=direct,verification_mac_terms=verification,
                          ratio=verification/direct,cheaper=verification<direct))
    result=dict(checks=checks,costs=costs,source_scope='reused CT1 mathematical development cases',
      probability_bound='<=1/(2147483647^2), theoretical for fixed error and independent secret post-commit field challenges',
      actual_secret_verifier=False,on_query_cost_gate=costs[1]['cheaper'],
      implementation_note='Harness deliberately performs redundant zero-challenge arithmetic during domain checking; MAC ledger models an optimized verifier, not actual harness runtime.',
      new_model_answers=0,full_dod_pass=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
