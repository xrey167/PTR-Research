"""Real frozen Qwen planner on unseen dialogue contract; address evaluation only."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from sentence_transformers import SentenceTransformer
from neural_pods.registry import Registry,InvalidState
from research.identity_dragonfly import IdentityDragonfly
from research.dialogue_access import DialogueAccess
from research.prefix_capsule import weights_hash
from research.cpu_int8 import quantize_in_place


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reader-3b',action='store_true')
    p.add_argument('--constrained',action='store_true',help='Constrain output syntax to catalogue addresses or UNKNOWN; does not establish semantic correctness')
    a=p.parse_args()
    source=Path('runs/identity-linked-001');out=a.output;out.mkdir(parents=True,exist_ok=False)
    src=sqlite3.connect((source/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    dst=sqlite3.connect(out/'registry.sqlite3');src.backup(dst);src.close();dst.close()
    shutil.copytree(source/'qdrant',out/'qdrant')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    cases=json.loads(Path('research/internal-knowledge-cases.json').read_text(encoding='utf-8'))
    (out/'protocol.json').write_text(json.dumps({'cases':cases,'models':manifest['models'],
        'planner_model':manifest['reader_model'] if a.reader_3b else manifest['models']['qwen']['path'],
        'numerics':'dynamic-int8' if a.reader_3b else 'float32',
        'constrained_output':a.constrained,
        'scope':'Zero-shot planning; no answer generation or link LoRA in this experiment'},ensure_ascii=False,indent=2),encoding='utf-8')
    report={'status':'running','rows':[],'full_internal_knowledge_gate_passed':False}
    start=time.perf_counter()
    def save():
        report['elapsed_s']=time.perf_counter()-start
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    r=Registry(out/'registry.sqlite3');router=None
    try:
        torch.set_num_threads(4)
        info=manifest['models']['encoder'];encoder=SentenceTransformer(info['path'],device='cpu',local_files_only=True)
        router=IdentityDragonfly(r,encoder,out,encoder_id=info['repo']+'@'+info['revision']);router.load_weights()
        access=DialogueAccess(router)
        path=manifest['reader_model'] if a.reader_3b else manifest['models']['qwen']['path']
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16 if a.reader_3b else torch.float32,
            attn_implementation='sdpa').eval().requires_grad_(False)
        if a.reader_3b:
            report['quantization']=quantize_in_place(model,lambda x:print(json.dumps(x),flush=True))
        base=weights_hash(model);report['base_sha256']=base
        for case in cases['cases']:
            raw={}
            def predict(system,payload):
                prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':payload}],tokenize=False,add_generation_prompt=True)
                inputs=tok(prompt,return_tensors='pt',add_special_tokens=False)
                kwargs={}
                if a.constrained:
                    catalogue=json.loads(payload)['catalogue']
                    allowed=[tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
                        for text in ['UNKNOWN',*[f"ADDRESS {entry['address']}" for entry in catalogue]]]
                    prompt_length=inputs.input_ids.shape[1]
                    def next_tokens(batch_id,sequence):
                        prefix=sequence[prompt_length:].tolist()
                        return sorted({candidate[len(prefix)] for candidate in allowed
                            if len(candidate)>len(prefix) and candidate[:len(prefix)]==prefix})
                    kwargs['prefix_allowed_tokens_fn']=next_tokens
                t=time.perf_counter()
                with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id,**kwargs)
                tokens=generated[0,inputs.input_ids.shape[1]:].tolist()
                text=tok.decode(tokens,skip_special_tokens=True).strip()
                raw.update(text=text,tokens=tokens,seconds=time.perf_counter()-t,input_tokens=inputs.input_ids.shape[1],planner_payload=json.loads(payload))
                return text
            try:
                proposed=access.prepare(case['question'],case['history'],predict)
                subject=proposed['selection']['query_semantics']['subject'];error=None
                dependencies=list(proposed['snapshot'].artifacts)
            except InvalidState as exc:subject=None;error=str(exc);dependencies=[]
            report['rows'].append({'id':case['id'],'expected_subject':case['expected_subject'],'subject':subject,
                'address_correct':subject==case['expected_subject'],'error':error,'model':raw,'dependencies':dependencies,
                'answer_evaluated':False})
            save();print(json.dumps({'case':case['id'],'correct':subject==case['expected_subject'],'text':raw.get('text')}),flush=True)
        report.update(status='completed',address_correct=sum(x['address_correct'] for x in report['rows']),
            base_unchanged=weights_hash(model)==base)
        assert report['base_unchanged'];save()
    except Exception as exc:report.update(status='failed',error=repr(exc));save();raise
    finally:
        if router is not None:router.close()
        r.close()


if __name__=='__main__':main()
