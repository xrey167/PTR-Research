import json,shutil,hashlib,math,sys
from pathlib import Path
src,out=Path(sys.argv[1]),Path(sys.argv[2]);shutil.copytree(src,out)
for name in ('train','dev','test'):
 p=out/'inputs'/f'{name}.json'; rows=json.loads(p.read_text()); base=[r for r in rows if ':toolchoice:' in r.get('id','')]
 for rep in range(8):
  for r in base:
   x=dict(r); x['id']=f'{name}:toolfocus:{rep}:{r["id"].split(":")[-1]}'; x['assessment']={'tool_contract_v1':True,'focused_repeat':rep+1,'negative':r.get('assessment',{}).get('negative',False)}; rows.append(x)
 p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
protocol=json.loads((out/'protocol.json').read_text());cfg=json.loads((out/'inputs/config.json').read_text());t=cfg['training'];protocol['rows']={k:len(json.loads((out/'inputs'/f'{k}.json').read_text())) for k in ('train','dev','test')};protocol['optimizer_updates_per_epoch']=math.ceil(protocol['rows']['train']/t['effective_batch_size']);protocol['optimizer_updates']=protocol['optimizer_updates_per_epoch']*t['epochs'];protocol['warmup_updates']=math.ceil(protocol['optimizer_updates']*t['warmup_ratio']);protocol['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'inputs').iterdir()};protocol['status']='prepared_not_trained_toolfocus';(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');print(protocol['rows'],protocol['optimizer_updates'])
