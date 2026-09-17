"""Continue the Qwen-3B reader LoRA on the frozen three-hop extension.

This is a bounded continuation from the validated two-hop adapter.  It keeps
the base model frozen and optimizes only LoRA parameters, with assistant-only
loss on the 8 training rows.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from research.prepare_multihop_extension import rows
from research.reader_prompt import encode_training_row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--adapter', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=2); args = ap.parse_args()
    set_seed(3407); args.output.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True).to('cuda:0')
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True).train()
    model.config.use_cache = False
    train_rows = rows('train')
    encoded = [encode_training_row(tok, r, 512) for r in train_rows]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
    steps=[]; started=time.perf_counter()
    for epoch in range(args.epochs):
        for index, item in enumerate(encoded):
            ids=torch.tensor([item['input_ids']],device='cuda:0'); mask=torch.tensor([item['attention_mask']],device='cuda:0'); labels=torch.tensor([item['labels']],device='cuda:0')
            out=model(input_ids=ids,attention_mask=mask,labels=labels); loss=out.loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            steps.append({'epoch':epoch+1,'row':index+1,'loss':float(loss.detach().cpu()),'elapsed_s':time.perf_counter()-started})
            print(json.dumps(steps[-1]),flush=True)
    model.config.use_cache=True; model.eval(); model.save_pretrained(args.output, safe_serialization=True); tok.save_pretrained(args.output/'tokenizer')
    report={'schema':'three-hop-qwen-lora-training:v1','base':str(args.base),'init_adapter':str(args.adapter),'epochs':args.epochs,'rows':len(encoded),'optimizer_steps':len(steps),'steps':steps,'status':'trained','device':'cuda:0','scope':'continuation training on synthetic held-out-entity three-hop curriculum'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'trained','optimizer_steps':len(steps),'output':str(args.output)},indent=2))


if __name__=='__main__': main()
