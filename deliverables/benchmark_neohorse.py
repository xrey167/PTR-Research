"""Small real-checkpoint NeoHorse throughput/quality probe.

This is an engineering probe, separate from the official ten-benchmark table.
Sampling follows the NeoHorse model-card defaults where applicable.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Write a Python function that returns the first n Fibonacci numbers.",
    "A supplier delivered 24 days for X12. A revised record says 18 days. Which current value should a lifecycle-aware system use, and why?",
    "Plan three tool calls to inspect a repository, test a change, and report only verified results.",
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--max-new-tokens',type=int,default=64); args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    model.eval(); rows=[]; total_tokens=0; total_s=0.0
    for prompt in PROMPTS:
        inputs=tok(prompt,return_tensors='pt').to(model.device)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t=time.perf_counter()
        with torch.inference_mode():
            # Transformers' generic generate API has no presence-penalty
            # argument; SGLang is the protocol-faithful path for that field.
            out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=True,temperature=1.0,top_p=.95,top_k=20,repetition_penalty=1.0)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed=time.perf_counter()-t; text=tok.decode(out[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        n=int(out.shape[1]-inputs['input_ids'].shape[1]); total_tokens+=n; total_s+=elapsed
        rows.append({'prompt':prompt,'generated_tokens':n,'elapsed_s':elapsed,'tokens_per_s':n/max(elapsed,1e-9),'output':text})
    report={'schema':'neohorse-real-checkpoint-probe:v1','model':str(args.model),'device':str(model.device),'dtype':'float16','sampling':{'temperature':1.0,'top_p':.95,'top_k':20,'presence_penalty':1.5,'repetition_penalty':1.0,'thinking':'model default'},'rows':rows,'total_generated_tokens':total_tokens,'total_elapsed_s':total_s,'mean_tokens_per_s':total_tokens/max(total_s,1e-9)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(args.output),'mean_tokens_per_s':report['mean_tokens_per_s'],'tokens':total_tokens},indent=2))
if __name__=='__main__': main()
