"""Staged real planner + real link LoRA + real capsule reader, one local DAG.

The imported planner lineage is a local snapshot, not distributed synchronization.
Dialogue history in these cases is explicitly supplied synthetic input evidence.
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
import psutil
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM,AutoTokenizer
from safetensors.torch import save_file
from neural_pods.registry import Registry,InvalidState,verify_files
from neural_pods.model import PodModel
from research.identity_dragonfly import IdentityDragonfly
from research.dialogue_access import DialogueAccess,PlannerDeferred,SYSTEM
from research.adopt_lineage import adopt_lineage
from research.linked_capsule import LinkedCapsules,LinkPrediction
from research.prefix_capsule import PrefixCapsules,weights_hash
from research.cpu_int8 import quantize_in_place


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['links','reader']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reader-input',choices=['json','plain'],default='json')
    p.add_argument('--reader-precision',choices=['int8','bf16'],default='int8')
    p.add_argument('--copy-from',type=Path,help='Clone a completed link stage for a new reader comparison')
    a=p.parse_args()
    run=a.run.resolve();source=Path('runs/identity-linked-001');training=Path('runs/planner-training-003')
    if a.copy_from:
        if a.stage!='reader':raise ValueError('Copy only for a reader comparison')
        old=json.loads((a.copy_from/'links.json').read_text(encoding='utf-8'));assert old['status']=='completed'
        run.mkdir(parents=True,exist_ok=False)
        src=sqlite3.connect((a.copy_from/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
        dst=sqlite3.connect(run/'registry.sqlite3');src.backup(dst);src.close();dst.close()
        for folder in ['adapters','qdrant','capsule_states']:shutil.copytree(a.copy_from/folder,run/folder)
        for name in ['manifest.json','cases.json','links.json']:shutil.copyfile(a.copy_from/name,run/name)
    if a.stage=='links':
        run.mkdir(parents=True,exist_ok=False)
        src=sqlite3.connect((source/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
        dst=sqlite3.connect(run/'registry.sqlite3');src.backup(dst);src.close();dst.close()
        for folder in ['adapters','qdrant','capsule_states']:shutil.copytree(source/folder,run/folder)
        shutil.copyfile(source/'manifest.json',run/'manifest.json')
        shutil.copyfile('research/internal-knowledge-cases.json',run/'cases.json')
    elif (run/'reader.json').exists():raise FileExistsError('Preserve prior reader attempt')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    cases=json.loads((run/'cases.json').read_text(encoding='utf-8'))['cases']
    report={'status':'running','stage':a.stage,'rows':[],'full_research_goal_complete':False,
            'reader_precision':a.reader_precision,
            'reader_input_format':a.reader_input,'copied_from':str(a.copy_from.resolve()) if a.copy_from else None}
    snapshot_dir=run/'source_snapshot'/a.stage
    snapshot_dir.mkdir(parents=True,exist_ok=False)
    sources={}
    for folder in ['research','neural_pods']:
        for path in sorted(Path(folder).glob('*.py')):
            target=snapshot_dir/path;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            sources[path.as_posix()]=hashlib.sha256(target.read_bytes()).hexdigest()
    (snapshot_dir/'manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    report['source_snapshot']=str(snapshot_dir.relative_to(run))
    report['source_manifest_sha256']=hashlib.sha256((snapshot_dir/'manifest.json').read_bytes()).hexdigest()
    proc=psutil.Process();report.update(phase='load',forward_calls=0,pid=proc.pid)
    start=time.perf_counter()
    def save():
        report['elapsed_s']=time.perf_counter()-start
        mem=proc.memory_info()
        report['resources']={'rss_bytes':mem.rss,'private_bytes':getattr(mem,'private',None),
            'available_ram_bytes':psutil.virtual_memory().available}
        target=run/(a.stage+'.json');temp=target.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(target)
    r=Registry(run/'registry.sqlite3');router=None;save()
    try:
        torch.set_num_threads(4)
        info=manifest['models']['encoder'];enc=SentenceTransformer(info['path'],device='cpu',local_files_only=True)
        router=IdentityDragonfly(r,enc,run,encoder_id=info['repo']+'@'+info['revision']);router.load_weights()
        if a.stage=='links':
            trained=json.loads((training/'report.json').read_text(encoding='utf-8'));assert trained['status']=='completed'
            planner=trained['planner_artifact'];src=Registry(training/'registry.sqlite3')
            try:adopt_lineage(src,r,planner)
            finally:src.close()
            pp=r.node(planner)['payload']['payload']
            shutil.copytree(training/'adapter'/pp['adapter'],run/'adapters'/pp['adapter'])
            verify_files(run/'adapters'/pp['adapter'],pp['files'])
            model=PodModel(manifest['models']['qwen']['path']);assert model.frozen_hash()==pp['base_sha256']
            model.model.load_adapter(run/'adapters'/pp['adapter'],adapter_name='planner',is_trainable=False)
            access=DialogueAccess(router)
            for case in cases:
                raw={}
                def predict(system,content):
                    assert system==SYSTEM
                    model.model.set_adapter('planner');model.model.eval()
                    prompt=model.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
                    inputs=model.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
                    with torch.inference_mode():tokens=model.model.generate(**inputs,max_new_tokens=12,do_sample=False,pad_token_id=model.tokenizer.eos_token_id)[0,inputs.input_ids.shape[1]:].tolist()
                    raw.update(text=model.tokenizer.decode(tokens,skip_special_tokens=True).strip(),tokens=tokens)
                    return raw['text']
                request=r.origin('fixture:dialogue-request',case['id'],'1',{'question':case['question'],'history':case['history']},acl=['buyer'])
                try:
                    plan=access.prepare(case['question'],case['history'],predict)
                    snapshot=plan['snapshot'];canonical=plan['canonical_lookup'];deferred=False
                except PlannerDeferred as decision:
                    snapshot=decision.snapshot;canonical=None;deferred=True
                proof=r.artifact('answer',{'task':'dialogue-plan','question':case['question'],'history':case['history'],
                    'text':raw['text'],'tokens':raw['tokens'],'canonical_lookup':canonical,'planner_key':planner},
                    [*snapshot.artifacts,planner,request],'buyer')
                row={'id':case['id'],'question':case['question'],'history':case['history'],'planner_proof':proof,
                     'planner_key':planner,'planner_text':raw['text'],'deferred':deferred,'canonical_lookup':canonical}
                if deferred:
                    row['receipt']=r.commit(r.snapshot([proof],'buyer'),'UNKNOWN')
                else:
                    selected=plan['selection'];binding=router.links.active_binding(selected['knowledge_key'],'buyer')
                    lp=r.node(binding['adapter_key'])['payload']['payload'];directory=(run/'adapters'/lp['adapter']).resolve()
                    assert directory.parent==(run/'adapters').resolve();verify_files(directory,lp['files'])
                    assert lp['base_sha256']==pp['base_sha256']
                    if lp['adapter'] not in model.model.peft_config:model.model.load_adapter(directory,adapter_name=lp['adapter'],is_trainable=False)
                    prediction=model.generate(canonical,adapter=lp['adapter'],task='link')
                    resolved=router.links.resolve(prediction['text'],expected_knowledge_key=selected['knowledge_key'],principal='buyer')
                    link_proof=r.artifact('answer',{'task':'link_prediction','question':canonical,'text':prediction['text'],
                        'adapter_key':binding['adapter_key']},[proof,*resolved['snapshot'].artifacts,*snapshot.artifacts],'buyer')
                    row.update(link_text=prediction['text'],link_proof=link_proof,knowledge_key=selected['knowledge_key'],
                        generation_key=selected['generation_key'],representation_key=selected['representation_key'],link_adapter=binding['adapter_key'])
                report['rows'].append(row);save()
            report['base_unchanged']=model.frozen_hash()==pp['base_sha256'];assert report['base_unchanged']
        else:
            linked=json.loads((run/'links.json').read_text(encoding='utf-8'));assert linked['status']=='completed'
            model=AutoModelForCausalLM.from_pretrained(manifest['reader_model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False)
            report['quantization']=quantize_in_place(model,lambda x:print(json.dumps(x),flush=True)) if a.reader_precision=='int8' else None
            base=weights_hash(model);tok=AutoTokenizer.from_pretrained(manifest['reader_model'],local_files_only=True)
            report['reader_model_sha256']=base
            def heartbeat(module,args,result):
                report['forward_calls']+=1;save()
            hook=model.register_forward_hook(heartbeat)
            eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id)
            capsules=PrefixCapsules(r,model,run/'capsule_states',base);joined=LinkedCapsules(router,capsules);raw={}
            compiled={}
            for row in linked['rows']:
                report.update(current_case=row['id'],phase='prepare');save()
                if row['deferred']:
                    r.snapshot(row['receipt']['dependencies'],'buyer')
                    report['rows'].append({**row,'text':'UNKNOWN','reader_invoked':False});save();continue
                if a.reader_input=='json':
                    question=json.dumps({'dialogue':[*row['history'],{'role':'user','content':row['question']}]},ensure_ascii=False)
                else:
                    history='\n'.join(m['role'].capitalize()+': '+m['content'] for m in row['history'])
                    question=('Earlier dialogue:\n'+history+'\nLatest user question: ' if history else '')+row['question']
                suffix_text=question+'\nAnswer the latest user question briefly. Use a number of days for a duration, and yes/no with a brief reason for a yes/no question.'
                r.snapshot([row['generation_key']],'buyer')
                s=r.node(row['generation_key'])['payload']['semantic'];aliases=s['retrieval']['trusted_aliases']
                fact=f"Supplier {aliases[0]} (aliases: {', '.join(aliases[1:])}) has a delivery lead time of {s['object']['value']} days for component {s['component']}."
                template=tok.apply_chat_template([{'role':'system','content':'Use the supplied fact to answer the question. Follow the answer format requested in the question. If the fact is missing, answer UNKNOWN.'},
                    {'role':'user','content':'Fact: '+fact+'\nQuestion: QUESTION_BOUNDARY_9'}],tokenize=False,add_generation_prompt=True)
                prefix,tail=template.split('QUESTION_BOUNDARY_9');ids=tok.encode(suffix_text+tail,add_special_tokens=False,return_tensors='pt')
                if a.reader_precision=='bf16' and row['generation_key'] not in compiled:
                    report['phase']='compile_capsule';save()
                    # Compile once per generation for this exact model; never reuse Int8 KV tensors.
                    key=capsules.compile(tok.encode(prefix,add_special_tokens=False,return_tensors='pt'),
                        row['generation_key'],'bf16_dialogue_'+str(len(compiled)))
                    joined.bind_variant(row['generation_key'],key)
                    compiled[row['generation_key']]=key
                    report['compiled_capsules']=dict(compiled);save()
                prepared=joined.prepare(row['canonical_lookup'],lambda q,a:LinkPrediction(row['link_text'],row['link_proof']))
                assert prepared.generation_key==row['generation_key']
                report['phase']='cached_decode';save()
                output=capsules.decode(ids,prepared.cache,eos,max_tokens=64)
                report['phase']='fresh_reference';save()
                reference=capsules.decode(ids,capsules.prefill(tok.encode(prefix,add_special_tokens=False,return_tensors='pt')),eos,max_tokens=64)
                assert torch.equal(output['logits'],reference['logits']) and output['tokens']==reference['tokens']
                text=tok.decode(output['tokens'],skip_special_tokens=True).strip();receipt=joined.commit(prepared,text)
                assert {row['planner_proof'],row['planner_key'],row['link_proof']} <= {n['id'] for n in r.ancestors(receipt['answer_id'])}
                raw[row['id']+'_cached']=output['logits'];raw[row['id']+'_reference']=reference['logits']
                report['rows'].append({**row,'text':text,'tokens':output['tokens'],'eos':output['eos'],'receipt':receipt,
                    'reader_invoked':True,'capsule_key':prepared.artifact_key,'fresh_equal':True})
                save_file(raw,run/'reader-logits.safetensors');save();print(json.dumps({'case':row['id'],'text':text}),flush=True)
            hook.remove();report['phase']='final_hash';save()
            report['base_unchanged']=weights_hash(model)==base;assert report['base_unchanged']
        report.update(status='completed',phase='done');save()
    except Exception as exc:report.update(status='failed',error=repr(exc));save();raise
    finally:
        if router is not None:router.close()
        r.close()


if __name__=='__main__':main()
