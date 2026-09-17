"""Train a multi-head Pod type/domain router on Vela representations."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

TYPES = ["context", "math", "model", "reasoning", "retrieval"]
ROWS = [
 ("Which supplier is linked to this component?", "context", "business"),
 ("How many days are in this delivery window?", "math", "business"),
 ("Which approval rule applies?", "model", "business"),
 ("Connect the invoice, supplier and delay across three records.", "reasoning", "business"),
 ("Find the most relevant evidence passage.", "retrieval", "general"),
 ("What is the legal meaning of this clause?", "context", "law"),
 ("Calculate the percentage change in price.", "math", "economics"),
 ("Should the policy be activated?", "model", "law"),
 ("Trace the multi-hop causal chain.", "reasoning", "science"),
 ("Search for matching documents and aliases.", "retrieval", "general"),
]
HELDOUT = [
 ("Which company context belongs to this part?", "context", "business"),
 ("Compute the number of transit days.", "math", "business"),
 ("What decision rule should govern this case?", "model", "business"),
 ("Follow the evidence chain across the related records.", "reasoning", "business"),
 ("Locate the supporting passage for this request.", "retrieval", "general"),
 ("Interpret the obligation in this agreement.", "context", "law"),
 ("Work out the rate of change.", "math", "economics"),
 ("Choose whether this policy should apply.", "model", "law"),
 ("Infer the cause from the linked observations.", "reasoning", "science"),
 ("Retrieve matching evidence and aliases.", "retrieval", "general"),
]

class Router(torch.nn.Module):
    def __init__(self, hidden, domain_count):
        super().__init__(); self.proj=torch.nn.Sequential(torch.nn.LayerNorm(hidden),torch.nn.Linear(hidden,128),torch.nn.Tanh()); self.type_head=torch.nn.Linear(128,len(TYPES)); self.domain_head=torch.nn.Linear(128,domain_count)
    def forward(self,x):
        z=self.proj(x); return self.type_head(z), self.domain_head(z)

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def run(model_id, device="cuda", data_path=None):
    torch.manual_seed(3407)
    tok=AutoTokenizer.from_pretrained(model_id); enc=AutoModel.from_pretrained(model_id).to(device).eval(); hidden=enc.config.hidden_size
    if data_path:
        all_rows=load_jsonl(data_path)
        train_rows=[r for r in all_rows if r["split"] == "train"]
        validation_rows=[r for r in all_rows if r["split"] == "validation"]
        test_rows=[r for r in all_rows if r["split"] == "test"]
    else:
        train_rows=[{"question":r[0],"pod_type":r[1],"domain":r[2]} for r in ROWS]
        validation_rows=[]
        test_rows=[{"question":r[0],"pod_type":r[1],"domain":r[2]} for r in HELDOUT]
    domains=sorted({r["domain"] for r in train_rows + validation_rows + test_rows}); dmap={d:i for i,d in enumerate(domains)}
    def encode(text):
        b=tok(text,return_tensors="pt").to(device)
        with torch.inference_mode(): o=enc(**b).last_hidden_state
        return o[0,0].float().cpu()
    x=torch.stack([encode(r["question"]) for r in train_rows]); y=torch.tensor([TYPES.index(r["pod_type"]) for r in train_rows]); yd=torch.tensor([dmap[r["domain"]] for r in train_rows])
    def tensors(rows):
        return (torch.stack([encode(r["question"]) for r in rows]),
                torch.tensor([TYPES.index(r["pod_type"]) for r in rows]),
                torch.tensor([dmap[r["domain"]] for r in rows]))
    vx,vy,vyd=tensors(validation_rows) if validation_rows else (None,None,None)
    hx,hy,hyd=tensors(test_rows)
    model=Router(hidden, len(domains)); opt=torch.optim.AdamW(model.parameters(),lr=3e-3)
    for _ in range(500):
        opt.zero_grad(); a,b=model(x); loss=torch.nn.functional.cross_entropy(a,y)+torch.nn.functional.cross_entropy(b,yd); loss.backward(); opt.step()
    with torch.inference_mode():
        a,b=model(x); ha,hb=model(hx)
        type_acc=float((a.argmax(-1)==y).float().mean()); domain_acc=float((b.argmax(-1)==yd).float().mean())
        test_type=float((ha.argmax(-1)==hy).float().mean()); test_domain=float((hb.argmax(-1)==hyd).float().mean())
        if vx is not None:
            va,vb=model(vx); validation_type=float((va.argmax(-1)==vy).float().mean()); validation_domain=float((vb.argmax(-1)==vyd).float().mean())
        else: validation_type=validation_domain=None
    out={"model":model_id,"train_rows":len(train_rows),"validation_rows":len(validation_rows),"test_rows":len(test_rows),"types":TYPES,"domains":domains,"type_train_accuracy":type_acc,"domain_train_accuracy":domain_acc,"type_validation_accuracy":validation_type,"domain_validation_accuracy":validation_domain,"type_test_accuracy":test_type,"domain_test_accuracy":test_domain,"scope":"balanced Pod taxonomy router; validation/test are held-out synthetic routing sets"}
    out["test_type_confusion"] = [[int(((ha.argmax(-1)==i)&(hy==j)).sum()) for j in range(len(TYPES))] for i in range(len(TYPES))]
    out["test_domain_confusion"] = [[int(((hb.argmax(-1)==i)&(hyd==j)).sum()) for j in range(len(domains))] for i in range(len(domains))]
    Path("runs/pod-router-training-002.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8"); return out

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default="llm-semantic-router/Vela-1.0-Encoder-307M"); p.add_argument("--device",default="cuda"); p.add_argument("--data",default=None); a=p.parse_args(); print(json.dumps(run(a.model,a.device,a.data),indent=2))
