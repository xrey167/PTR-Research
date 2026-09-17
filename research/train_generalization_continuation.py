"""Train a small Qwen3B LoRA continuation on held-out-like query forms."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
from peft import PeftModel
from research.prepare_generalization_curriculum import rows
from research.reader_prompt import encode_training_row

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--epochs',type=int,default=2); a=ap.parse_args()
 set_seed(3407); a.output.mkdir(parents=True,exist_ok=False); tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda:0')
 model=PeftModel.from_pretrained(model,str(a.adapter),is_trainable=True).train(); model.config.use_cache=False
 enc=[encode_training_row(tok,r,512) for r in rows()]; opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01); steps=[]; st=time.perf_counter()
 for ep in range(a.epochs):
  for i,x in enumerate(enc):
   ids=torch.tensor([x['input_ids']],device='cuda:0'); mask=torch.tensor([x['attention_mask']],device='cuda:0'); labels=torch.tensor([x['labels']],device='cuda:0'); loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); steps.append({'epoch':ep+1,'row':i+1,'loss':float(loss.detach().cpu())})
 model.config.use_cache=True; model.eval(); model.save_pretrained(a.output,safe_serialization=True); tok.save_pretrained(a.output/'tokenizer'); (a.output/'report.json').write_text(json.dumps({'schema':'generalization-continuation:v1','status':'trained','rows':len(enc),'optimizer_steps':len(steps),'epochs':a.epochs,'steps':steps},indent=2)+'\n'); print(json.dumps({'status':'trained','optimizer_steps':len(steps)}))
if __name__=='__main__': main()
