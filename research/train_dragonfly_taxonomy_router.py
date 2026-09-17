"""Train Dragonfly-style Pod addresses jointly with type/domain heads."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

TYPES = ["context", "math", "model", "reasoning", "retrieval"]

class DragonflyTaxonomy(torch.nn.Module):
    def __init__(self, hidden, domains, roles, intents, tags):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden,128), torch.nn.Tanh())
        self.prototypes = torch.nn.Parameter(torch.randn(len(TYPES),128)*.02)
        self.type_head = torch.nn.Linear(128,len(TYPES)); self.domain_head = torch.nn.Linear(128,len(domains)); self.role_head=torch.nn.Linear(128,len(roles)); self.intent_head=torch.nn.Linear(128,len(intents)); self.tag_head=torch.nn.Linear(128,len(tags))
    def forward(self, x):
        z = torch.nn.functional.normalize(self.proj(x), dim=-1)
        p = torch.nn.functional.normalize(self.prototypes, dim=-1)
        return z @ p.T * 12.0, self.type_head(z), self.domain_head(z), self.role_head(z), self.intent_head(z), self.tag_head(z)

def run(model_id, data_path, device="cuda"):
    torch.manual_seed(3407)
    rows=[json.loads(x) for x in Path(data_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    domains=sorted({r["domain"] for r in rows}); dmap={d:i for i,d in enumerate(domains)}; roles=sorted({r["semantic_role"] for r in rows}); rmap={d:i for i,d in enumerate(roles)}; intents=sorted({r["intent"] for r in rows}); imap={d:i for i,d in enumerate(intents)}; tags=sorted({t for r in rows for t in r["tags"]}); tagmap={t:i for i,t in enumerate(tags)}
    tok=AutoTokenizer.from_pretrained(model_id); enc=AutoModel.from_pretrained(model_id).to(device).eval(); hidden=enc.config.hidden_size
    def encode(text):
        b=tok(text,return_tensors="pt").to(device)
        with torch.inference_mode(): return enc(**b).last_hidden_state[0,0].float().cpu()
    def tensors(split):
        rs=[r for r in rows if r["split"]==split]
        tag_y=torch.zeros((len(rs),len(tags)))
        for j,r in enumerate(rs):
            for tag in r["tags"]: tag_y[j,tagmap[tag]]=1
        return (torch.stack([encode(r["question"]) for r in rs]), torch.tensor([TYPES.index(r["pod_type"]) for r in rs]), torch.tensor([dmap[r["domain"]] for r in rs]), torch.tensor([rmap[r["semantic_role"]] for r in rs]), torch.tensor([imap[r["intent"]] for r in rs]), tag_y)
    x,y,yd,yr,yi,yt=tensors("train"); vx,vy,vyd,vyr,vyi,vyt=tensors("validation"); tx,ty,tyd,tyr,tyi,tyt=tensors("test")
    model=DragonflyTaxonomy(hidden,domains,roles,intents,tags); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=.01)
    for _ in range(700):
        opt.zero_grad(); address, typ, dom, role, intent, tag=model(x)
        loss=sum(torch.nn.functional.cross_entropy(a,b) for a,b in ((address,y),(typ,y),(dom,yd),(role,yr),(intent,yi))) + torch.nn.functional.binary_cross_entropy_with_logits(tag,yt)
        loss.backward(); opt.step()
    with torch.inference_mode():
        a,t,d,r,i,tag=model(tx); va,vt,vd,vr,vi,vtag=model(vx)
        tag_acc=float(((tag.sigmoid() >= .5)==tyt.bool()).float().mean())
        val_tag_acc=float(((vtag.sigmoid() >= .5)==vyt.bool()).float().mean())
        out={"model":model_id,"data":data_path,"train_rows":len(y),"validation_rows":len(vy),"test_rows":len(ty),"domains":domains,
             "dragonfly_test_top1":float((a.argmax(-1)==ty).float().mean()),"type_test_accuracy":float((t.argmax(-1)==ty).float().mean()),"domain_test_accuracy":float((d.argmax(-1)==tyd).float().mean()),"role_test_accuracy":float((r.argmax(-1)==tyr).float().mean()),"intent_test_accuracy":float((i.argmax(-1)==tyi).float().mean()),"tag_accuracy":tag_acc,"tags":tags,
             "dragonfly_validation_top1":float((va.argmax(-1)==vy).float().mean()),"type_validation_accuracy":float((vt.argmax(-1)==vy).float().mean()),"domain_validation_accuracy":float((vd.argmax(-1)==vyd).float().mean()),"role_validation_accuracy":float((vr.argmax(-1)==vyr).float().mean()),"intent_validation_accuracy":float((vi.argmax(-1)==vyi).float().mean()),"tag_validation_accuracy":val_tag_acc,
             "scope":"joint Dragonfly prototype-address and Pod taxonomy training on held-out templates/entities"}
    artifact="runs/dragonfly-taxonomy-router-001.pt"; torch.save({"state_dict":model.state_dict(),"hidden":hidden,"domains":domains,"roles":roles,"intents":intents,"tags":tags,"seed":3407},artifact); out["artifact"]=artifact
    reloaded=DragonflyTaxonomy(hidden,domains,roles,intents,tags); reloaded.load_state_dict(torch.load(artifact,map_location="cpu",weights_only=True)["state_dict"]); reloaded.eval()
    with torch.inference_mode():
        ra,rt,rd,rr,ri,rg=reloaded(tx)
        out["reload_dragonfly_test_top1"]=float((ra.argmax(-1)==ty).float().mean()); out["reload_type_test_accuracy"]=float((rt.argmax(-1)==ty).float().mean()); out["reload_domain_test_accuracy"]=float((rd.argmax(-1)==tyd).float().mean()); out["reload_role_test_accuracy"]=float((rr.argmax(-1)==tyr).float().mean()); out["reload_intent_test_accuracy"]=float((ri.argmax(-1)==tyi).float().mean()); out["reload_tag_accuracy"]=float(((rg.sigmoid() >= .5)==tyt.bool()).float().mean())
    Path("runs/dragonfly-taxonomy-training-001.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8"); return out

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default="llm-semantic-router/Vela-1.0-Encoder-307M"); p.add_argument("--data",required=True); p.add_argument("--device",default="cuda"); a=p.parse_args(); print(json.dumps(run(a.model,a.data,a.device),indent=2))
