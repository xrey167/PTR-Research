"""Same-model latency comparison for compact Pod context vs longer RAG context."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def measure(model, tok, prompts, runs, max_new):
    enc=tok(prompts,return_tensors='pt',padding=True).to('cuda')
    with torch.inference_mode():
        for _ in range(3): model.generate(**enc,max_new_tokens=max_new,do_sample=False,use_cache=True)
        torch.cuda.synchronize(); t0=time.perf_counter(); outputs=[]
        for _ in range(runs): outputs.append(model.generate(**enc,max_new_tokens=max_new,do_sample=False,use_cache=True))
        torch.cuda.synchronize(); elapsed=time.perf_counter()-t0
    generated=sum(int(x.shape[1]-enc['input_ids'].shape[1]) for x in outputs)
    return {'input_tokens':int(enc['attention_mask'].sum(dim=1).float().mean()),'elapsed_s':elapsed,
            'batch_latency_ms':elapsed/runs*1000,'generated_tokens':generated,
            'tokens_per_s':generated/elapsed,'sequences_per_s':runs*len(prompts)/elapsed,
            'sample_output':tok.decode(outputs[-1][0].tolist(),skip_special_tokens=True)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',required=True); ap.add_argument('--adapter',required=True); ap.add_argument('--output',required=True); ap.add_argument('--runs',type=int,default=10); a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda').eval()
    model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False).eval()
    question='What is the supplier lead time for Müller GmbH X12? Answer briefly.'
    def chat(user):
        return tok.apply_chat_template([{'role':'system','content':'Answer only the exact delivery duration in days. Treat Pod IDs, generation IDs, and hashes as opaque identifiers; use only the explicit semantic value.'}, {'role':'user','content':user}], tokenize=False, add_generation_prompt=True)
    pod=[chat('Activated Pod type=context; knowledge_key=supplier:muller:x12:lead_time; generation=g8 (opaque). semantic_value=18 days. '+question)]*4
    evidence=' '.join(['Supplier Müller GmbH, component X12, historical procurement evidence: lead time is 18 days.']*12)
    rag=[chat(f'User question: {question}\nRetrieved documents:\n{evidence}\nSynthesize the answer briefly.')] * 4
    report={'schema':'qwen-pod-vs-rag-throughput:v1','device':torch.cuda.get_device_name(0),'batch_size':4,'runs':a.runs,'max_new_tokens':16,
            'pod':measure(model,tok,pod,a.runs,16),'rag':measure(model,tok,rag,a.runs,16),
            'interpretation':'same Qwen3B LoRA and GPU; Pod path uses compact typed activation metadata, RAG path uses repeated retrieved evidence; quality is not inferred from latency'}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__': main()


