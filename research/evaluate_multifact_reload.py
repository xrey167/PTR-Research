"""Reload a saved multi-fact Qwen LoRA and replay its no-evidence cases."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def run(model_path, adapter_path, report_path, device='cuda'):
    report=json.loads(Path(report_path).read_text(encoding='utf-8')); cases=report['test_rows']; tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to(device).eval(); model=PeftModel.from_pretrained(base,adapter_path,local_files_only=True).to(device).eval(); rows=[]
    for c in cases:
        prompt=tok.apply_chat_template([{'role':'system','content':'Answer the exact delivery duration in days. If the supplier is unknown, answer UNKNOWN.'},{'role':'user','content':c['question']}],tokenize=False,add_generation_prompt=True); ids=tok(prompt,return_tensors='pt').to(device)
        with torch.inference_mode(): out=model.generate(**ids,do_sample=False,max_new_tokens=8,pad_token_id=tok.eos_token_id)
        text=tok.decode(out[0,ids['input_ids'].shape[1]:],skip_special_tokens=True).strip(); rows.append({'question':c['question'],'text':text,'target':c['target'],'correct':text==c['target']})
    result={'rows':rows,'correct':sum(r['correct'] for r in rows),'total':len(rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),'scope':'fresh-process reload of multi-fact value-bearing LoRA; no evidence text'}; Path('runs/multifact-internal-lora-004/reload-report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); return result
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--adapter',required=True); p.add_argument('--report',required=True); p.add_argument('--device',default='cuda'); a=p.parse_args(); print(json.dumps(run(a.model,a.adapter,a.report,a.device),indent=2,ensure_ascii=False))
