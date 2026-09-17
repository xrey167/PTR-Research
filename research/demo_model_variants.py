"""Create and lifecycle-check all executable Model-Pod variants."""
from __future__ import annotations
import json
from pathlib import Path
from neural_pods.registry import Registry
from neural_pods.pod_types import ModelVariant, typed_artifact
from neural_pods.execution_manifest import ExecutionManifest
from neural_pods.model_pod import ModelPodRuntime

def main():
    out=Path('runs/model-pod-variants-001.json'); reg=Registry(':memory:')
    origin=reg.origin('model-lab','variant-suite','1',{'suite':'all-model-variants'})
    knowledge=reg.publish('model:variant-suite',{'subject':'reader','predicate':'executable_variant'},[origin])
    artifacts=[]
    for variant in ModelVariant:
        payload={'reader_identity':f'reader-{variant.value}','model_variant':variant.value,
                 'model_sha256':f'sha-{variant.value}','input_schema':'pod:v1','output_schema':'text:v1'}
        if variant is ModelVariant.LORA: payload['adapter_sha256']='adapter-lora'
        if variant is ModelVariant.DISTILLED: payload['teacher_identity']='teacher-qwen-v1'
        if variant is ModelVariant.MOE: payload['experts']=['expert-a','expert-b']
        artifacts.append(typed_artifact(reg,'model','model',payload,[knowledge]))
    nodes=[reg.node(a) for a in artifacts]
    variants=[n['payload']['payload']['model_variant'] for n in nodes]
    plans=[]
    runtime=ModelPodRuntime(reg)
    for artifact, variant in zip(artifacts, variants):
        manifest=ExecutionManifest.build(reg, knowledge, [artifact], reader_key=artifact)
        plans.append(runtime.prepare(manifest, base_model='Qwen/Qwen2.5-3B')['operation'])
    revoked=reg.revoke(origin)
    blocked=True
    for artifact in artifacts:
        try:
            reg.snapshot([artifact])
            blocked=False
        except Exception:
            pass
    report={'schema':'model-pod-variants-demo:v1','variants':variants,'artifacts':artifacts,
            'activation_operations':plans,
            'revoked_origin':origin,'revoked_count':revoked,'all_activation_blocked':blocked,
            'passed':set(variants)=={v.value for v in ModelVariant} and
                     plans==['attach_lora','load_student','load_moe','load_quantized','load_base'] and blocked}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8'); reg.close(); print(json.dumps(report,indent=2)); raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__': main()
