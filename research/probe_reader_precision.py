"""Reader-only BF16 control against the completed Int8 plain-dialogue run.

Exactly the same segmented prefix/suffix and 64-token greedy decoding. No
planner/encoder loaded, no Int8 cache reused. Saves each completed case.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psutil
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from neural_pods.registry import Registry
from research.prefix_capsule import PrefixCapsules,weights_hash
from research.progress_json import write_progress


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic-deadlines',action='store_true',help='Run the fixed 12-case language/unit diagnosis instead of the Int8 comparison')
    a=p.parse_args()
    source=Path('runs/dialogue-capsules-plain-001');out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    reference=json.loads((source/'reader.json').read_text(encoding='utf-8'));assert reference['status']=='completed'
    rows=[r for r in reference['rows'] if r['reader_invoked']]
    db=sqlite3.connect((source/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    try:semantics={r['generation_key']:json.loads(db.execute('SELECT payload FROM nodes WHERE id=?',(r['generation_key'],)).fetchone()[0])['semantic'] for r in rows}
    finally:db.close()
    diagnostic=None
    if a.diagnostic_deadlines:
        from research.reader_deadline_cases import build_cases
        seed=next(row for row in rows if row['id']=='direct')
        assert semantics[seed['generation_key']]['object']['value']==27
        diagnostic=build_cases()
        (out/'diagnostic-cases.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf-8')
        rows=[{'id':case['id'],**case['reader_input'],'generation_key':seed['generation_key'],
               'text':None,'tokens':None} for case in diagnostic]
    protocol={'model_path':manifest['reader_model'],'dtype':'bfloat16','source':str(source.resolve()),
        'reference_report_sha256':hashlib.sha256((source/'reader.json').read_bytes()).hexdigest(),
        'cases':[r['id'] for r in rows],'max_tokens':64,'scope':'Reader-only numerical control, no routing/lifecycle integration',
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if diagnostic is not None:
        protocol.update(scope='Reader-only language/unit diagnosis; no paired Int8 results, no training or held-out generalization claim',
            diagnostic_deadlines=True,diagnostic_generation=seed['generation_key'],
            diagnostic_cases_sha256=hashlib.sha256((out/'diagnostic-cases.json').read_bytes()).hexdigest())
    snapshot=out/'source_snapshot';snapshot.mkdir()
    sources={}
    for folder in ['research','neural_pods']:
        for path in sorted(Path(folder).glob('*.py')):
            target=snapshot/path;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);sources[path.as_posix()]=hashlib.sha256(target.read_bytes()).hexdigest()
    (snapshot/'manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    protocol['source_manifest_sha256']=hashlib.sha256((snapshot/'manifest.json').read_bytes()).hexdigest()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    report={'status':'running','phase':'load','rows':[],'forward_calls':0,'full_research_goal_complete':False}
    started=time.perf_counter();proc=psutil.Process()
    def save():
        mem=proc.memory_info();report.update(elapsed_s=time.perf_counter()-started,
            resources={'rss_bytes':mem.rss,'private_bytes':getattr(mem,'private',None),'page_faults':getattr(mem,'num_page_faults',None),
                       'available_ram_bytes':psutil.virtual_memory().available,'pid':proc.pid})
        write_progress(out/'report.json',report)
    save();r=Registry(':memory:')
    try:
        torch.set_num_threads(4)
        model=AutoModelForCausalLM.from_pretrained(protocol['model_path'],local_files_only=True,dtype=torch.bfloat16,
            attn_implementation='sdpa').eval().requires_grad_(False)
        report['phase']='initial_hash';save();base=weights_hash(model);report['base_sha256']=base
        tok=AutoTokenizer.from_pretrained(protocol['model_path'],local_files_only=True)
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id)
        capsules=PrefixCapsules(r,model,out/'unused_capsules',base)
        def heartbeat(module,args,result):
            report['forward_calls']+=1
            try:save()
            except PermissionError:
                report['progress_write_failures']=report.get('progress_write_failures',0)+1
        hook=model.register_forward_hook(heartbeat)
        raw={}
        for row in rows:
            report['current_case']=row['id'];report['phase']='prefill';save()
            s=semantics[row['generation_key']];aliases=s['retrieval']['trusted_aliases']
            fact=f"Supplier {aliases[0]} (aliases: {', '.join(aliases[1:])}) has a delivery lead time of {s['object']['value']} days for component {s['component']}."
            template=tok.apply_chat_template([{'role':'system','content':'Use the supplied fact to answer the question. Follow the answer format requested in the question. If the fact is missing, answer UNKNOWN.'},
                {'role':'user','content':'Fact: '+fact+'\nQuestion: QUESTION_BOUNDARY_9'}],tokenize=False,add_generation_prompt=True)
            prefix,tail=template.split('QUESTION_BOUNDARY_9')
            history='\n'.join(m['role'].capitalize()+': '+m['content'] for m in row['history'])
            question=('Earlier dialogue:\n'+history+'\nLatest user question: ' if history else '')+row['question']
            suffix=question+'\nAnswer the latest user question briefly. Use a number of days for a duration, and yes/no with a brief reason for a yes/no question.'+tail
            prefix_ids=tok.encode(prefix,add_special_tokens=False,return_tensors='pt');suffix_ids=tok.encode(suffix,add_special_tokens=False,return_tensors='pt')
            t=time.perf_counter();cache=capsules.prefill(prefix_ids);report['phase']='decode';save()
            output=capsules.decode(suffix_ids,cache,eos,max_tokens=64)
            text=tok.decode(output['tokens'],skip_special_tokens=True).strip()
            report['rows'].append({'id':row['id'],'question':row['question'],'history':row['history'],'text':text,
                'tokens':output['tokens'],'eos':output['eos'],'prefix_ids':prefix_ids[0].tolist(),'suffix_ids':suffix_ids[0].tolist(),
                'int8_text':row['text'],'int8_tokens':row['tokens'],'seconds':time.perf_counter()-t})
            raw[row['id']]=output['logits'];save_file(raw,out/'first-logits.safetensors');save()
            print(json.dumps({'case':row['id'],'text':text,'resources':report['resources']},ensure_ascii=False),flush=True)
            del cache,output
        hook.remove();report['phase']='final_hash';save()
        report['base_unchanged']=weights_hash(model)==base;assert report['base_unchanged']
        report.update(status='completed',phase='done');save()
    except Exception as exc:report.update(status='failed',error=repr(exc));save();raise
    finally:r.close()


if __name__=='__main__':main()
