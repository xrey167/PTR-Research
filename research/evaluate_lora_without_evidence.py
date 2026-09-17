"""Probe whether a trained reader LoRA recalls held-out facts without evidence."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def run(model_path, adapter_path, cases_path, device='cuda', limit=6):
    cases=[r for r in json.loads(Path(cases_path).read_text(encoding='utf-8')) if r.get('task')=='lookup'][:limit]
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to(device).eval()
    model=PeftModel.from_pretrained(base,adapter_path,local_files_only=True).to(device).eval()
    rows=[]
    for r in cases:
        prompt=tok.apply_chat_template([{'role':'system','content':'Answer only the exact delivery duration in days.'},{'role':'user','content':r['question']}],tokenize=False,add_generation_prompt=True)
        ids=tok(prompt,return_tensors='pt').to(device)
        with torch.inference_mode(): out=model.generate(**ids,do_sample=False,max_new_tokens=12,pad_token_id=tok.eos_token_id)
        text=tok.decode(out[0,ids['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        rows.append({'id':r['id'],'question':r['question'],'text':text,'target':r['target'],'correct':text==r['target'],'fact_in_input':False})
    result={'rows':rows,'correct':sum(r['correct'] for r in rows),'total':len(rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),'scope':'held-out Qwen-3B reader LoRA without evidence text'}
    Path('runs/lora-no-evidence-control-001.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); return result
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--adapter',required=True); p.add_argument('--cases',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--limit',type=int,default=6); a=p.parse_args(); print(json.dumps(run(a.model,a.adapter,a.cases,a.device,a.limit),indent=2,ensure_ascii=False))
