\"\"\"Local-only CQTA2-S E0 entrypoint. No automatic later-stage authorization.\"\"\"
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'serialized_atlas'))
from native_e0 import PINS,sha,dump
from gate import evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),HERE/'gate.py',HERE/'PROTOCOL_DE.md',
      ROOT/'serialized_atlas/native_e0.py',ROOT/'semantic_linker/run_tskv_prefix_quotient.py',
      ROOT/'semantic_linker/run_ds3_oracle.py',ROOT/'semantic_linker/PROTOCOL_TSKV_P1.md',
      ROOT/'semantic_linker/PROTOCOL_CQTA2_SYSTEM_DE.md',ROOT/'FROZEN_DOD.md',
      ROOT/'stronger/model_download_manifest.json']
    seals={str(x.relative_to(ROOT)):sha(x) for x in sources}
    actual={}
    for name in PINS:
        try: actual[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: actual[name]=None
    pre={'source_hashes':seals,'expected_packages':PINS,'actual_packages':actual,
         'q0_authorized':False,'compiler_training_authorized':False,'full_dod':False}
    dump(a.output/'preseal.json',pre)
    if actual"'!=PINS or not (a.model/'"'download_manifest.json').is_file():
        result=pre|{'status':'blocked_environment','new_model_sequences':0,
                    'cqta2_e0_gate':False,'reason':'Pinned packages or local model manifest unavailable'}
        dump(a.output/'results.json',result)
        print(json.dumps(result,indent=2));return 3
    sys.path.insert(0,str(ROOT/'semantic_linker'))
    import run_ds3_oracle as oracle
    import run_tskv_prefix_quotient as legacy
    torch,tok,model,att=oracle.load_model(a.model)
    torch.set_num_threads(4)
    selected=legacy.cases('engineering');dump(a.output/'cases.json',selected)
    start=time.perf_counter()
    rows=legacy.run_engineering(model,tok,selected)
    dump(a.output/'answers.json',rows)
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    result=evaluate(rows,selected,model.config.vocab_size,eos)
    unchanged=all(sha(ROOT/name)==digest for name,digest in seals.items())
    result=result|pre|{'sources_unchanged':unchanged,'new_model_sequences':len(rows),
      'seconds':time.perf_counter()-start,'model':att['model'],'revision':att['revision']}
    if not unchanged: result['cqta2_e0_gate']=result['engineering_gate']=False
    result['status']='pass' if result['cqta2_e0_gate'] else 'fail'
    dump(a.output/'results.json',result)
    dump(a.output/'seal.json',{'sources':seals,'answers_sha256':sha(a.output/'answers.json'),
      'results_sha256':sha(a.output/'results.json'),'q0_authorized':False})
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='pass' else 2

if __name__=='__main__': raise SystemExit(main())
