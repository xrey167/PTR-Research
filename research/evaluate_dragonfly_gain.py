"""Compare raw Vela centroid routing with the learned Dragonfly checkpoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
from train_dragonfly_taxonomy_router import DragonflyTaxonomy, TYPES

def run(model_id, data_path, artifact, device="cuda"):
    rows=[json.loads(x) for x in Path(data_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    tr=[r for r in rows if r["split"]=="train"]; te=[r for r in rows if r["split"]=="test"]
    tok=AutoTokenizer.from_pretrained(model_id); enc=AutoModel.from_pretrained(model_id).to(device).eval()
    def e(text):
        b=tok(text,return_tensors="pt").to(device)
        with torch.inference_mode(): return torch.nn.functional.normalize(enc(**b).last_hidden_state[0,0].float().cpu(),dim=0)
    train_x=torch.stack([e(r["question"]) for r in tr]); test_x=torch.stack([e(r["question"]) for r in te]); y=torch.tensor([TYPES.index(r["pod_type"]) for r in tr]); ty=torch.tensor([TYPES.index(r["pod_type"]) for r in te])
    centroids=torch.stack([torch.nn.functional.normalize(train_x[y==i].mean(0),dim=0) for i in range(len(TYPES))]); baseline=(test_x@centroids.T).argmax(-1)
    ck=torch.load(artifact,map_location="cpu",weights_only=True); model=DragonflyTaxonomy(enc.config.hidden_size,ck["domains"],ck["roles"],ck["intents"],ck["tags"]); model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.inference_mode(): learned=model(test_x)[0].argmax(-1)
    out={"baseline_raw_centroid_accuracy":float((baseline==ty).float().mean()),"dragonfly_accuracy":float((learned==ty).float().mean()),"delta":float(((learned==ty).float().mean()-(baseline==ty).float().mean())),"test_rows":len(te),"scope":"same held-out questions; raw Vela centroid versus learned Dragonfly address"}
    Path("runs/dragonfly-gain-comparison-001.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8"); return out
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default="llm-semantic-router/Vela-1.0-Encoder-307M"); p.add_argument("--data",required=True); p.add_argument("--artifact",required=True); p.add_argument("--device",default="cuda"); a=p.parse_args(); print(json.dumps(run(a.model,a.data,a.artifact,a.device),indent=2))
