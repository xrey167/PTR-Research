from pathlib import Path
from fractions import Fraction
import copy
import json
import time
import numpy as np
from engine import build,verify,transport_program,tensor_digest,snapshot,canon,digest,Invalid

HERE=Path(__file__).resolve().parent


def put(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return digest(p.read_bytes())


def make(seed,shape):
    m,d,k=shape;rng=np.random.default_rng(seed)
    dims={'B':(m,d),'S':(m,d),'R':(m,d),'C':(m,k),'Y':(k,d),'X':(k,d)}
    t={n:rng.integers(-16,17,size=s).astype('<f8')/16 for n,s in dims.items()}
    t['C'][0,0]=1.0  # The preregistered Y edit has a guaranteed nonzero effect.
    return t,{n:1 for n in t}


def exact_reference(t):
    m,d=t['B'].shape;k=t['X'].shape[0]
    f=lambda a,i,j:Fraction.from_float(float(t[a][i,j]))
    return [[f('B',i,j)+f('S',i,j)+f('R',i,j)+sum((f('C',i,l)*(f('Y',l,j)-f('X',l,j)) for l in range(k)),Fraction())
             for j in range(d)] for i in range(m)]


def main():
    out=HERE/'results';out.mkdir(exist_ok=False)
    program=transport_program()
    for seed in (11,19):
        t,e=make(seed,(2,4,2));y,p=build(program,t,e);assert verify(program,t,e,y,p)
        exact=exact_reference(t)
        assert all(Fraction.from_float(float(y[i,j]))==exact[i][j] for i,j in np.ndindex(y.shape))
    sources={n:sha(HERE/n) for n in ('engine.py','run.py','PROTOCOL.md')}
    put(out/'seal.json',{'sources':sources,'development_seeds':[11,19],'validation_started':False})
    rows=[];artifacts={};started=time.perf_counter()
    for seed in (101,103,107,109,113,127,131,137,139,149,151,157):
        for shape in ((2,4,2),(3,8,3),(4,16,4)):
            t,e=make(seed,shape);y,proof=build(program,t,e);exact=exact_reference(t)
            exact_ok=all(Fraction.from_float(float(y[i,j]))==exact[i][j] for i,j in np.ndindex(y.shape))
            changed={k:v.copy() for k,v in t.items()};changed['Y'][0,0]+=1.;e2={**e,'Y':2}
            fresh,fp=build(program,changed,e2)
            assert not np.array_equal(y,fresh)
            relabelled=copy.deepcopy(fp);relabelled['output_digest']=tensor_digest(y)
            # An incomplete checker trusts its new manifest and merely hashes the old output.
            weak_accept=tensor_digest(y)==relabelled['output_digest'] and relabelled['snapshot_root']==snapshot(changed,e2)[1]
            aba={**e,'Y':3};same,ap=build(program,t,aba)
            undeclared=copy.deepcopy(program);undeclared[4]['name']='undeclared'
            row={'seed':seed,'shape':shape,'exact_fraction_match':exact_ok,
              'fresh_accept':verify(program,t,e,y,proof),
              'stale_rejected':not verify(program,changed,e2,y,proof),
              'relabelled_old_output_rejected':not verify(program,changed,e2,y,relabelled),
              'weak_hash_control_accepts_relabelled':bool(weak_accept),
              'new_output_accepted':verify(program,changed,e2,fresh,fp),
              'undeclared_input_rejected':not verify(undeclared,t,e,y,proof),
              'aba_old_evidence_rejected':not verify(program,t,aba,y,proof),
              'aba_new_evidence_accepted':verify(program,t,aba,same,ap),
              'aba_output_unchanged':bool(np.array_equal(y,same)),
              'output_sha256':tensor_digest(y),'new_output_sha256':tensor_digest(fresh)}
            assert all(v for k,v in row.items() if k not in ('seed','shape','output_sha256','new_output_sha256'))
            rows.append(row)
            tag=f'{seed}_{shape[0]}_{shape[1]}_{shape[2]}'
            for n,v in t.items():artifacts[tag+'_'+n]=v
            artifacts[tag+'_output']=y;artifacts[tag+'_new_output']=fresh
            put(out/'rows.json',rows)
    negatives=[]
    t,e=make(163,(2,4,2));y,p=build(program,t,e)
    for kind in ('operation','reference','shape','nonfinite','abi','boolean_epoch'):
        prog=copy.deepcopy(program);tt={k:v.copy() for k,v in t.items()};ep=dict(e);ev=copy.deepcopy(p)
        if kind=='operation':prog[-1]={'op':'unregistered'}
        elif kind=='reference':prog[-1]['a']=True
        elif kind=='shape':tt['C']=np.ones((1,1),dtype='<f8')
        elif kind=='nonfinite':tt['X'][0,0]=np.nan
        elif kind=='abi':ev['abi']='unverified-abi'
        else:ep['X']=True
        rejected=not verify(prog,tt,ep,y,ev);assert rejected
        negatives.append({'case':kind,'rejected':rejected})
    np.savez_compressed(out/'raw_tensors.npz',**artifacts)
    put(out/'structural_controls.json',negatives)
    assert sources=={n:sha(HERE/n) for n in sources}
    results={'status':'complete','exact_math_cases':len(rows),'all_case_gates_pass':True,
      'relabelled_old_outputs_rejected':sum(r['relabelled_old_output_rejected'] for r in rows),
      'weak_hash_false_accepts':sum(r['weak_hash_control_accepts_relabelled'] for r in rows),
      'structural_controls_pass':len(negatives),'new_model_answers':0,'archived_tensors_used':False,
      'scope':'dyadic mathematical tensors; deterministic replay, not cryptographic proof or neural generalization',
      'seconds':time.perf_counter()-started,'full_dod_pass':False}
    put(out/'summary.json',results);print(json.dumps(results,indent=2))


if __name__=='__main__':main()
