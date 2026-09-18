"""Reader-only BF16 control against the completed Int8 plain-dialogue run.

Exactly the same segmented prefix/suffix and 64-token greedy decoding. No
planner/encoder loaded, no Int8 cache reused. Saves each completed case.
"""
import argparse
import hashlib
import json
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=Path('runs/dialogue-capsules-plain-001');out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    reference=json.loads((source/'reader.json').read_text(encoding='utf-8'));assert reference['status']=='completed'
    rows=[r for r in reference['rows'] if r['reader_invoked']]
    db=sqlite3.connect((source/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    try:semantics={r['generation_key']:json.loads(db.execute('SELECT payload FROM nodes WHERE id=?',(r['generation_key'],)).fetchone()[0])['semantic'] for r in rows}
    finally:db.close()
    protocol={'model_path':manifest['reader_model'],'dtype':'bfloat16','source':str(source.resolve()),
        'reference_report_sha256':hashlib.sha256((source/'reader.json').read_bytes()).hexdigest(),
        'cases':[r['id'] for r in rows],'max_tokens':64,'scope':'Reader-only numerical control, no routing/lifecycle integration',
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    report={'status':'running','phase':'load','rows':[],'forward_calls':0,'full_research_goal_complete':False}
    started=time.perf_counter();proc=psutil.Process()
    def save():
        mem=proc.memory_info();report.update(elapsed_s=time.perf_counter()-started,
            resources={'rss_bytes':mem.rss,'private_bytes':getattr(mem,'private',None),'page_faults':getattr(mem,'num_page_faults',None),
                       'available_ram_bytes':psutil.virtual_memory().available,'pid':proc.pid})
        temp=out/'report.json.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(out/'report.json')
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
            report['forward_calls']+=1;save()
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
