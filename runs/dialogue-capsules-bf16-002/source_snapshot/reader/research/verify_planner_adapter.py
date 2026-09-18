"""Reload the saved planner, reproduce held-out tokens and probe dialogue access.

Planning-only experiment. It does not publish an end-to-end knowledge answer.
"""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sentence_transformers import SentenceTransformer
from neural_pods.model import PodModel
from neural_pods.registry import Registry,InvalidState,verify_files
from research.dialogue_access import DialogueAccess,SYSTEM
from research.identity_dragonfly import IdentityDragonfly


def main():
    p=argparse.ArgumentParser();p.add_argument('training_run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.training_run.resolve()
    training=json.loads((source/'report.json').read_text(encoding='utf-8'))
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    data=json.loads((source/'dataset.json').read_text(encoding='utf-8'))
    assert training['status']=='completed' and protocol['system']==SYSTEM
    assert [r['id'] for r in training['trained']]==[r['id'] for r in data['test']]
    reg=Registry(source/'registry.sqlite3')
    try:
        reg.snapshot([training['planner_artifact']],'buyer')
        payload=reg.node(training['planner_artifact'])['payload']['payload']
        path=(source/'adapter'/payload['adapter']).resolve()
        assert path.parent==(source/'adapter').resolve()
        verify_files(path,payload['files'])
    finally:reg.close()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','reload':[],'dialogue':[],'planner_artifact':training['planner_artifact'],
        'training_source':str(source),'scope':'Saved adapter replay and guarded address regression, no factual answers or planner proof materialization',
        'full_research_goal_complete':False}
    start=time.perf_counter()
    def save():
        report['elapsed_s']=time.perf_counter()-start
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    r=None;router=None
    try:
        model=PodModel(protocol['model']['path']);assert model.frozen_hash()==payload['base_sha256']
        model.model.load_adapter(path,adapter_name='planner_reloaded',is_trainable=False)
        model.model.set_adapter('planner_reloaded');model.model.eval()
        def infer(system,content):
            prompt=model.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            inputs=model.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
            with torch.inference_mode():tokens=model.model.generate(**inputs,max_new_tokens=12,do_sample=False,pad_token_id=model.tokenizer.eos_token_id)[0,inputs.input_ids.shape[1]:].tolist()
            return {'text':model.tokenizer.decode(tokens,skip_special_tokens=True).strip(),'tokens':tokens}
        for row,old in zip(data['test'],training['trained']):
            prediction=infer(SYSTEM,json.dumps(row['input'],ensure_ascii=False))
            assert prediction['tokens']==old['tokens']
            report['reload'].append({'id':row['id'],'same_tokens':True,**prediction});save()
        fixture=Path('runs/identity-linked-001')
        src=sqlite3.connect((fixture/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
        dst=sqlite3.connect(out/'registry.sqlite3');src.backup(dst);src.close();dst.close()
        shutil.copytree(fixture/'qdrant',out/'qdrant')
        manifest=json.loads((fixture/'manifest.json').read_text(encoding='utf-8'))
        info=manifest['models']['encoder'];encoder=SentenceTransformer(info['path'],device='cpu',local_files_only=True)
        r=Registry(out/'registry.sqlite3');router=IdentityDragonfly(r,encoder,out,encoder_id=info['repo']+'@'+info['revision']);router.load_weights()
        access=DialogueAccess(router)
        cases=json.loads(Path('research/internal-knowledge-cases.json').read_text(encoding='utf-8'))['cases']
        for case in cases:
            raw={}
            def predict(system,content):
                raw.update(infer(system,content));return raw['text']
            try:
                chosen=access.prepare(case['question'],case['history'],predict)
                subject=chosen['selection']['query_semantics']['subject'];error=None
            except InvalidState as exc:subject=None;error=str(exc)
            report['dialogue'].append({'id':case['id'],'subject':subject,'expected_subject':case['expected_subject'],
                'guarded_correct':subject==case['expected_subject'],'error':error,'model':raw});save()
        report.update(status='completed',reload_equal=len(report['reload'])==len(data['test']),
            guarded_dialogue_correct=sum(r['guarded_correct'] for r in report['dialogue']),
            base_unchanged=model.frozen_hash()==payload['base_sha256'])
        assert report['base_unchanged'];save();print(json.dumps({k:v for k,v in report.items() if k not in ['reload','dialogue']}))
    except Exception as exc:report.update(status='failed',error=repr(exc));save();raise
    finally:
        if router is not None:router.close()
        if r is not None:r.close()


if __name__=='__main__':main()
