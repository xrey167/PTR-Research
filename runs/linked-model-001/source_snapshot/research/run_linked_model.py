"""Two real model stages, with persisted lineage between them, on a 16 GiB host.

Stage links: actual trained 0.5B Qwen LoRA and learned Dragonfly.
Stage capsules: frozen 3B Int8 decoder with a sibling capsule of the same fact.
This measures integration correctness, not simultaneous-serving latency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
from neural_pods.registry import Registry, InvalidState, verify_files
from neural_pods.dragonfly import DragonflyRouter
from neural_pods.symlink import NeuralSymlinks
from neural_pods.model import PodModel
from research.prefix_capsule import PrefixCapsules, weights_hash
from research.linked_capsule import LinkedCapsules, LinkPrediction
from research.cpu_int8 import quantize_in_place


def save(path, data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['links','capsules'])
    p.add_argument('--source',type=Path,default=Path('runs/embedded-symlink-002'))
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    run=args.run.resolve()
    started=time.perf_counter()
    if args.stage=='links':
        run.mkdir(parents=True,exist_ok=False)
        manifest=json.loads((args.source/'report.json').read_text(encoding='utf-8'))
        assert manifest['status']=='completed'
        source=sqlite3.connect((args.source/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
        destination=sqlite3.connect(run/'registry.sqlite3')
        source.backup(destination);source.close();destination.close()
        for name in ['qdrant','adapters']:shutil.copytree(args.source/name,run/name)
        original=json.loads((args.source/'symlink-report.json').read_text(encoding='utf-8'))
        manifest['questions']=list(dict.fromkeys(r['question'] for r in original['answers']))
        manifest['source_run']=str(args.source.resolve())
        manifest['reader_model']=str(Path('models/qwen3b').resolve())
        manifest['experiment']='staged-real-link-capsule:v1'
        save(run/'manifest.json',manifest)
    else:
        manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        assert json.loads((run/'links.json').read_text(encoding='utf-8'))['status']=='passed'
        if (run/'capsules.json').exists():raise FileExistsError('Preserve prior reader runs')
    report={'status':'running','stage':args.stage,'rows':[],'fictional_data':True}
    reg=Registry(run/'registry.sqlite3')
    router=None
    def checkpoint():
        report['elapsed_s']=time.perf_counter()-started
        save(run/(args.stage+'.json'),report)
    checkpoint()
    try:
        info=manifest['models']['encoder']
        encoder=SentenceTransformer(info['path'],device='cpu',local_files_only=True)
        router=DragonflyRouter(reg,encoder,run,encoder_id=info['repo']+'@'+info['revision'])
        router.load_weights()
        links=NeuralSymlinks(reg)
        if args.stage=='links':
            model=PodModel(manifest['models']['qwen']['path'])
            assert model.frozen_hash()==manifest['base_weights_sha256']
            for question in manifest['questions']:
                selection=router.select(question,principal='buyer')
                binding=links.active_binding(selection['knowledge_key'],'buyer')
                payload=reg.node(binding['adapter_key'])['payload']['payload']
                path=(run/'adapters'/payload['adapter']).resolve()
                assert path.parent==(run/'adapters').resolve()
                verify_files(path,payload['files'])
                assert payload['base_sha256']==manifest['base_weights_sha256']
                if payload['adapter'] not in model.model.peft_config:
                    model.model.load_adapter(path,adapter_name=payload['adapter'],is_trainable=False)
                snapshot=reg.snapshot([*selection['snapshot'].artifacts,binding['adapter_key'],binding['identity_key']],'buyer')
                prediction=model.generate(question,adapter=payload['adapter'],task='link')
                links.resolve(prediction['text'],expected_knowledge_key=selection['knowledge_key'],principal='buyer')
                proof=reg.artifact('answer',{'task':'link_prediction','question':question,'text':prediction['text'],
                      'adapter_key':binding['adapter_key'],'base_sha256':manifest['base_weights_sha256'],
                      'input_tokens':prediction['input_tokens'],'output_tokens':prediction['output_tokens']},snapshot.artifacts,'buyer')
                report['rows'].append({'question':question,**prediction,'proof_key':proof,
                        'knowledge_key':selection['knowledge_key'],'generation_key':selection['generation_key'],
                        'representation_key':selection['representation_key'],'adapter_key':binding['adapter_key']})
                checkpoint()
            report['base_unchanged']=model.frozen_hash()==manifest['base_weights_sha256']
            assert report['base_unchanged']
        else:
            proofs=json.loads((run/'links.json').read_text(encoding='utf-8'))['rows']
            torch.set_num_threads(4)
            model=AutoModelForCausalLM.from_pretrained(manifest['reader_model'],local_files_only=True,
                    dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False)
            report['quantization']=quantize_in_place(model,lambda x:print(json.dumps({'event':'quantize',**x}),flush=True))
            base=weights_hash(model)
            tok=AutoTokenizer.from_pretrained(manifest['reader_model'],local_files_only=True)
            eos=model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id)
            capsules=PrefixCapsules(reg,model,run/'capsule_states',base)
            joined=LinkedCapsules(router,capsules)
            prefixes={};tails={};artifacts={}
            for i,row in enumerate(proofs):
                generation=row['generation_key']
                if generation in artifacts:continue
                semantic=reg.node(generation)['payload']['semantic']
                aliases=semantic['retrieval']['trusted_aliases']
                fact=f"Supplier {aliases[0]} (aliases: {', '.join(aliases[1:])}) has a delivery lead time of {semantic['object']['value']} days for component {semantic['component']}."
                template=tok.apply_chat_template([
                    {'role':'system','content':'Use the supplied fact to answer the question. Follow the answer format requested in the question. If the fact is missing, answer UNKNOWN.'},
                    {'role':'user','content':'Fact: '+fact+'\nQuestion: QUESTION_BOUNDARY_9'}],tokenize=False,add_generation_prompt=True)
                prefix,tail=template.split('QUESTION_BOUNDARY_9')
                prefixes[generation]=tok.encode(prefix,add_special_tokens=False,return_tensors='pt')
                tails[generation]=tail
                artifact=capsules.compile(prefixes[generation],generation,'state_'+str(i))
                joined.bind_variant(generation,artifact)
                artifacts[generation]=artifact
            report['reader_base_sha256']=base
            report['capsule_artifacts']=artifacts
            raw={}
            for row in proofs:
                question=row['question'];t=time.perf_counter()
                prepared=joined.prepare(question,lambda q,a:LinkPrediction(row['text'],row['proof_key']))
                generation=prepared.generation_key
                suffix=tok.encode(question+'\nReply with only the integer number of days.'+tails[generation],add_special_tokens=False,return_tensors='pt')
                output=capsules.decode(suffix,prepared.cache,eos)
                text=tok.decode(output['tokens'],skip_special_tokens=True).strip()
                receipt=joined.commit(prepared,text)
                elapsed=time.perf_counter()-t
                reference=capsules.decode(suffix,capsules.prefill(prefixes[generation]),eos)
                equal=torch.equal(reference['logits'],output['logits']) and reference['tokens']==output['tokens']
                assert equal
                expected=str(reg.node(generation)['payload']['semantic']['object']['value'])
                item={'question':question,'text':text,'expected':expected,'correct':text==expected and output['eos'],
                      'tokens':output['tokens'],'eos':output['eos'],'proof_key':row['proof_key'],
                      'reader_s':elapsed,'receipt':receipt,'fresh_reference_equal':equal,
                      'knowledge_key':prepared.knowledge_key,'generation_key':generation,'artifact_key':prepared.artifact_key}
                assert {row['proof_key'],row['adapter_key'],row['representation_key'],prepared.artifact_key} <= set(receipt['dependencies'])
                raw[str(len(report['rows']))+'_cached']=output['logits']
                raw[str(len(report['rows']))+'_reference']=reference['logits']
                report['rows'].append(item)
                save_file(raw,run/'reader-logits.safetensors')
                checkpoint();print(json.dumps({'question_completed':len(report['rows']),'correct':item['correct']}),flush=True)
            # Clone-only revocation checks on real dependency graphs.
            controls={}
            for kind in ['proof','source','update']:
                clone=Registry(':memory:');reg.db.backup(clone.db)
                try:
                    sample=report['rows'][0]
                    snapshot=clone.snapshot(sample['receipt']['dependencies'],'buyer')
                    if kind=='proof':clone.revoke(sample['proof_key'])
                    elif kind=='source':
                        origin=clone.db.execute('SELECT parent FROM edges WHERE child=?',(sample['generation_key'],)).fetchone()[0]
                        clone.revoke(origin)
                    else:
                        original=clone.node(sample['generation_key'])['payload']
                        semantic=json.loads(json.dumps(original['semantic']));semantic['object']['value']=24
                        origin=clone.origin('fixture:integration','update','1',semantic,acl=['buyer'])
                        clone.publish(sample['knowledge_key'],semantic,[origin],'buyer',['buyer'])
                    try:clone.commit(snapshot,sample['text'])
                    except InvalidState:controls[kind+'_blocks_commit']=True
                    else:raise AssertionError(kind+' did not block')
                    if kind=='proof':
                        clone.snapshot([sample['artifact_key']],'buyer')
                        other=report['rows'][1]
                        clone.snapshot(other['receipt']['dependencies'],'buyer')
                        controls['proof_revocation_is_selective']=True
                finally:clone.close()
            report['controls']=controls
            report['base_unchanged']=weights_hash(model)==base
            assert report['base_unchanged']
            report['lookup_regression_passed']=all(x['correct'] for x in report['rows'])
            report['full_research_goal_complete']=False
        report['status']='passed' if args.stage=='links' else 'completed'
        checkpoint();print(json.dumps({'stage':args.stage,'status':report['status'],'rows':len(report['rows'])}))
    except Exception as exc:
        report.update(status='failed',error=repr(exc));checkpoint();raise
    finally:
        if router is not None:router.close()
        reg.close()


if __name__=='__main__':main()
