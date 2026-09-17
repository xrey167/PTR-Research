import json, shutil, hashlib, math, sys
from pathlib import Path
src,out=Path(sys.argv[1]),Path(sys.argv[2]); shutil.copytree(src,out)
for name in ('train','dev','test'):
 p=out/'inputs'/f'{name}.json'; rows=json.loads(p.read_text()); hard=[r for r in rows if ':generation2:' in r.get('id','') and not r.get('id','').startswith(f'{name}:ngu')]
 for repeat in range(3):
  for r in hard:
   x=dict(r); x['id']=f"{name}:ngu:{repeat}:{r['id'].split(':')[-1]}"; x['assessment']={'generation3':True,'ngu_retry':repeat+1}; rows.append(x)
 p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
cfgp=out/'inputs/config.json'; cfg=json.loads(cfgp.read_text()); t=cfg['training']; t['micro_batch_size']=2; t['gradient_accumulation_steps']=8; t['effective_batch_size']=16; t['epochs']=2; cfgp.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
protocol=json.loads((out/'protocol.json').read_text()); protocol['rows']={k:len(json.loads((out/'inputs'/f'{k}.json').read_text())) for k in ('train','dev','test')}; protocol['optimizer_updates_per_epoch']=math.ceil(protocol['rows']['train']/t['effective_batch_size']); protocol['optimizer_updates']=protocol['optimizer_updates_per_epoch']*t['epochs']; protocol['warmup_updates']=math.ceil(protocol['optimizer_updates']*t['warmup_ratio']); protocol['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'inputs').iterdir()}; protocol['status']='prepared_not_trained_generation3_full_load'; protocol['purpose']='NGU oversampling with microbatch 2 for higher GPU occupancy'; (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8'); print(protocol['rows'],protocol['optimizer_updates'])
