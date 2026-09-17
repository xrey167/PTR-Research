"""Bind a value-bearing LoRA artifact to a concrete generation and verify revocation."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neural_pods.registry import Registry, InvalidState

def sha(path): return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
def run(registry_path, adapter_run, generation, output):
    src=Path(registry_path); out=Path(output); shutil.copy2(src,out); reg=Registry(out); adapter=Path(adapter_run)/'adapter_model.safetensors';
    if not adapter.exists():
        adapter=Path(adapter_run)/'adapter'/'adapter_model.safetensors'
    training=reg.origin('value-lora-training',Path(adapter_run).name,'1',{'adapter_sha256':sha(adapter)},acl=['buyer'])
    artifact=reg.artifact('lora',{'schema':'value-bearing-qwen-lora:v1','adapter_sha256':sha(adapter),'adapter_path':str(adapter.resolve()),'knowledge_role':'value_reader','generation_key':generation,'training_origin':training,'activation_contract':'generation_and_training_lineage'},[generation,training],principal='buyer')
    reg.snapshot([artifact],'buyer'); reg.revoke(training)
    blocked=False
    try: reg.snapshot([artifact],'buyer')
    except InvalidState: blocked=True
    result={'generation_key':generation,'artifact_key':artifact,'training_origin':training,'adapter_sha256':sha(adapter),'revoked_activation_blocked':blocked,'scope':'value-bearing LoRA bound to generation; copied registry'}
    Path(output).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); reg.close(); return result
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--registry',required=True); p.add_argument('--adapter-run',required=True); p.add_argument('--generation',required=True); p.add_argument('--output',required=True); a=p.parse_args(); print(json.dumps(run(a.registry,a.adapter_run,a.generation,a.output),indent=2))
