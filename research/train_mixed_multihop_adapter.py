"""Train a replay-protected Qwen reader adapter on two- and three-hop data."""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from research.prepare_multihop_extension import rows as three_rows
from research.reader_prompt import encode_training_row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--replay',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--epochs',type=int,default=2); a=ap.parse_args()
    set_seed(3407); a.output.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    # Replay the complete frozen two-hop reader curriculum.  Filtering to only
    # multi_hop_total rows causes forgetting of deadline and dialogue rules.
    old=json.loads(a.replay.read_text())
    new=three_rows('train'); all_rows=old+new
    model=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda:0')
    model=PeftModel.from_pretrained(model,a.adapter,is_trainable=True).train(); model.config.use_cache=False
    enc=[encode_training_row(tok,r,512) for r in all_rows]
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=.01); steps=[]; st=time.perf_counter()
    for ep in range(a.epochs):
      for i,x in enumerate(enc):
        ids=torch.tensor([x['input_ids']],device='cuda:0'); mask=torch.tensor([x['attention_mask']],device='cuda:0'); labels=torch.tensor([x['labels']],device='cuda:0')
        loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        steps.append({'epoch':ep+1,'row':i+1,'loss':float(loss.detach().cpu())}); print(json.dumps(steps[-1]),flush=True)
    model.config.use_cache=True; model.eval(); model.save_pretrained(a.output,safe_serialization=True); tok.save_pretrained(a.output/'tokenizer')
    report={'schema':'mixed-multihop-qwen-lora-training:v1','status':'trained','rows':len(enc),'replay_two_hop_rows':len(old),'new_three_hop_rows':len(new),'epochs':a.epochs,'optimizer_steps':len(steps),'device':'cuda:0','steps':steps}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'status':'trained','optimizer_steps':len(steps),'rows':len(enc)},indent=2))
if __name__=='__main__': main()
