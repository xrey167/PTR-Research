"""No-prefix control for the hidden Pod activation experiment."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from research.cpu_int8 import quantize_in_place
from research.prefix_capsule import PrefixCapsules, weights_hash

def run(run_dir, model_path, device="cpu"):
    run_dir=Path(run_dir); manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8')); questions=manifest['questions']
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False)
    quantize_in_place(model, lambda _: None); tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    eos=model.generation_config.eos_token_id; eos=set(eos if isinstance(eos,list) else [eos]); eos.add(tok.eos_token_id)
    cap=PrefixCapsules(None,model,run_dir/'_none',weights_hash(model))
    rows=[]
    for q in questions:
        text=q+'\nReply with only the integer number of days.\n<|im_start|>assistant\n'
        ids=tok.encode(text,add_special_tokens=False,return_tensors='pt')
        out=cap.decode(ids,None,eos,max_tokens=12); decoded=tok.decode(out['tokens'],skip_special_tokens=True).strip()
        rows.append({'question':q,'text':decoded,'expected':'18','correct':decoded=='18','fact_in_input':False})
    result={'rows':rows,'correct':sum(r['correct'] for r in rows),'total':len(rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),'scope':'Qwen-3B Int8 no-prefix control; no Pod KV activation'}
    (run_dir/'no-prefix-control.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); return result
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); p.add_argument('--model',required=True); p.add_argument('--device',default='cpu'); a=p.parse_args(); print(json.dumps(run(a.run,a.model,a.device),indent=2,ensure_ascii=False))
