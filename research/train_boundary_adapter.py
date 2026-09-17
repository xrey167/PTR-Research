"""Continue a Qwen reader adapter on held-out-style arithmetic boundaries."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from research.reader_prompt import encode_training_row

def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True); p.add_argument('--adapter',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=2); a=p.parse_args()
    set_seed(3407); a.output.mkdir(parents=True,exist_ok=False); tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    rows=json.loads(a.data.read_text(encoding='utf-8')); model=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda:0'); model=PeftModel.from_pretrained(model,a.adapter,is_trainable=True).train(); model.config.use_cache=False
    enc=[encode_training_row(tok,r,512) for r in rows]; opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=5e-6,weight_decay=.01); steps=[]
    for ep in range(a.epochs):
      for i,x in enumerate(enc):
        ids=torch.tensor([x['input_ids']],device='cuda:0'); mask=torch.tensor([x['attention_mask']],device='cuda:0'); labels=torch.tensor([x['labels']],device='cuda:0'); loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); steps.append({'epoch':ep+1,'row':i+1,'loss':float(loss.detach().cpu())})
    model.config.use_cache=True; model.eval(); model.save_pretrained(a.output,safe_serialization=True); tok.save_pretrained(a.output/'tokenizer'); report={'schema':'boundary-qwen-lora-training:v1','rows':len(rows),'epochs':a.epochs,'optimizer_steps':len(steps),'learning_rate':5e-6,'device':'cuda:0','status':'trained'}; (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
