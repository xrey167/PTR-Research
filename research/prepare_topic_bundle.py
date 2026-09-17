"""Prepare a real Qwen LoRA bundle by extending the frozen reader curriculum."""
from __future__ import annotations
import hashlib, json, math, shutil
from pathlib import Path

def topics(split):
    items = [
      ("What isolates a Pod cache?", "Namespace, branch, Pod type and tags isolate the cache."),
      ("What does Pod identity remain stable across?", "Pod identity remains stable across immutable artifact branches."),
      ("Which field selects the current revision?", "The lifecycle boundary resolves the current artifact revision."),
      ("What is a retrieval Pod used for?", "A retrieval Pod combines ANN, BM25, filters and rank fusion."),
      ("What is a reasoning Pod used for?", "A reasoning Pod performs multi-hop inference over evidence."),
      ("What is the MoE router separate from?", "The MoE expert router is separate from the semantic Pod router."),
      ("What does prewarm do?", "Prewarm builds a namespace snapshot before a latency-sensitive burst."),
      ("When must a Pod be retrained?", "Retraining is needed when semantic mapping or aliases change."),
    ]
    rows=[]
    for i,(q,a) in enumerate(items):
      rows.append({'id':f'{split}:topic:{i}','task':'typed_topic','pod_type':'context' if i<3 else ('retrieval' if i in (3,6) else 'reasoning'),'language':'en','evidence':None,'question':q,'history':[],'target':a,'assessment':{'topic':True}})
    return rows

def main(src, out):
    src, out = Path(src), Path(out); shutil.copytree(src,out,dirs_exist_ok=False)
    for name, split in [('train.json','train'),('dev.json','dev'),('test.json','test')]:
      p=out/'inputs'/name; rows=json.loads(p.read_text()); rows.extend(topics(split)); p.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    protocol=json.loads((out/'protocol.json').read_text()); inp=out/'inputs'; protocol['rows']={k:len(json.loads((inp/f'{k}.json').read_text())) for k in ('train','dev','test')}; cfg=json.loads((inp/'config.json').read_text()); t=cfg['training']; protocol['optimizer_updates_per_epoch']=math.ceil(protocol['rows']['train']/t['gradient_accumulation_steps']); protocol['optimizer_updates']=protocol['optimizer_updates_per_epoch']*t['epochs']; protocol['warmup_updates']=math.ceil(protocol['optimizer_updates']*t['warmup_ratio']); protocol['last_window_microbatches']=protocol['rows']['train']%t['gradient_accumulation_steps'] or t['gradient_accumulation_steps']; protocol['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inp.iterdir()}; protocol['status']='prepared_not_trained'; protocol['purpose']='Typed Pod topics added to the Qwen reader curriculum'; (out/'protocol.json').write_text(json.dumps(protocol,indent=2)); print(json.dumps({'rows':protocol['rows'],'updates':protocol['optimizer_updates']}))
if __name__=='__main__':
 import sys; main(sys.argv[1],sys.argv[2])
