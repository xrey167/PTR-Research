"""Train a projected Pod router on Vela encoder representations."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

ENTITIES = ["Müller", "Kern", "Nova", "Atlas", "Rhein", "Elbe"]
TRAIN = ["What is the delivery time for {name}?", "Find the procurement record for {name}.", "Which supplier memory contains {name}?", "Summarize the lead time associated with {name}."]
TEST = ["Retrieve transit duration associated with {name}.", "Give me the lead-time fact from {name}."]

class Router(torch.nn.Module):
    def __init__(self, hidden, width=128, classes=6):
        super().__init__(); self.proj=torch.nn.Sequential(torch.nn.LayerNorm(hidden),torch.nn.Linear(hidden,width),torch.nn.Tanh()); self.p=torch.nn.Parameter(torch.randn(classes,width)*.02); self.t=torch.nn.Parameter(torch.tensor(1.0))
    def forward(self,x): return torch.nn.functional.normalize(self.proj(x),dim=-1) @ torch.nn.functional.normalize(self.p,dim=-1).T * self.t.exp().clamp(max=20)

def run(model_id, device="cuda", pooling="mean"):
    tok=AutoTokenizer.from_pretrained(model_id); enc=AutoModel.from_pretrained(model_id).to(device).eval(); hidden=enc.config.hidden_size
    def encode(text):
        b=tok(text,return_tensors="pt").to(device)
        with torch.inference_mode(): o=enc(**b).last_hidden_state
        if pooling == "first": return o[0, 0].float().cpu()
        m=b["attention_mask"].unsqueeze(-1); return ((o*m).sum(1)/m.sum(1).clamp_min(1))[0].float().cpu()
    tr_x=[]; tr_y=[]; te_x=[]; te_y=[]
    for i,n in enumerate(ENTITIES):
        for q in TRAIN: tr_x.append(encode(q.format(name=n))); tr_y.append(i)
        for q in TEST: te_x.append(encode(q.format(name=n))); te_y.append(i)
    tr_x,tr_y,te_x,te_y=torch.stack(tr_x),torch.tensor(tr_y),torch.stack(te_x),torch.tensor(te_y)
    r=Router(hidden,classes=len(ENTITIES)); opt=torch.optim.AdamW(r.parameters(),lr=3e-3,weight_decay=.01)
    for _ in range(600):
        opt.zero_grad(); loss=torch.nn.functional.cross_entropy(r(tr_x),tr_y); loss.backward(); opt.step()
    with torch.inference_mode(): logits=r(te_x); probs=logits.softmax(-1); acc=float((logits.argmax(-1)==te_y).float().mean())
    neg=[float(v) for row,y in zip(probs,te_y) for i,v in enumerate(row) if i!=int(y)]
    out={"model":model_id,"pooling":pooling,"hidden":hidden,"train_questions":len(tr_y),"heldout_questions":len(te_y),"top1_accuracy":acc,"mean_negative_probability":sum(neg)/len(neg),"scope":"Vela encoder; trained projection/prototype router; synthetic alias-family pilot"}
    Path("runs/vela-dragonfly-router-001.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8"); return out

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default="llm-semantic-router/Vela-1.0-Encoder-307M"); p.add_argument("--device",default="cuda"); p.add_argument("--pooling",choices=["mean","first"],default="first"); a=p.parse_args(); print(json.dumps(run(a.model,a.device,a.pooling),indent=2))
