"""Measure real Qwen3B LoRA generation throughput on a CUDA host."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',required=True); ap.add_argument('--adapter',required=True); ap.add_argument('--output',required=True); ap.add_argument('--runs',type=int,default=10); a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(a.base, local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation='eager', local_files_only=True).to('cuda').eval()
    model=PeftModel.from_pretrained(model,a.adapter, is_trainable=False).eval()
    prompt='Use the supplied typed Pod fact. Question: What is the supplier lead time? Answer briefly.'
    batch=[prompt]*4; enc=tok(batch,return_tensors='pt',padding=True).to('cuda')
    with torch.inference_mode():
        for _ in range(3): model.generate(**enc,max_new_tokens=16,do_sample=False,use_cache=True)
        torch.cuda.synchronize(); t0=time.perf_counter(); out=[]
        for _ in range(a.runs): out.append(model.generate(**enc,max_new_tokens=16,do_sample=False,use_cache=True))
        torch.cuda.synchronize(); elapsed=time.perf_counter()-t0
    new_tokens=sum(x.shape[1]-enc['input_ids'].shape[1] for x in out)
    report={'schema':'qwen-gpu-throughput:v1','device':torch.cuda.get_device_name(0),'batch_size':4,'runs':a.runs,'max_new_tokens':16,'elapsed_s':elapsed,'batch_latency_ms':elapsed/a.runs*1000,'generated_tokens':new_tokens,'tokens_per_s':new_tokens/elapsed,'sequences_per_s':(a.runs*4)/elapsed,'sample':tok.decode(out[0][0].tolist(),skip_special_tokens=True)}
    Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
