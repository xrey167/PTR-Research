"""Verify selective revocation blocks a replay answer receipt."""
from __future__ import annotations
import argparse, json, shutil, sqlite3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.registry import Registry, InvalidState

def run(source):
    source=Path(source); target=source.parent/(source.name+'-revocation-check')
    if target.exists(): shutil.rmtree(target)
    target.mkdir(); shutil.copy2(source/'registry.sqlite3',target/'registry.sqlite3')
    capsules=json.loads((source/'capsules.json').read_text(encoding='utf-8')); row=capsules['rows'][0]; answer_id=row['receipt']['answer_id']; generation=row['generation_key']
    reg=Registry(target/'registry.sqlite3'); node=reg.node(generation); origin=node['payload']['lifecycle']['source']; revoked=set(reg.revoke(origin))
    blocked=False
    try: reg.snapshot([answer_id],'buyer')
    except InvalidState: blocked=True
    result={'origin':origin,'revoked_count':len(revoked),'answer_id':answer_id,'receipt_blocked':blocked,'scope':'copied replay registry; selective origin revocation'}
    (source/'revocation-control.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); return result
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run'); a=p.parse_args(); print(json.dumps(run(a.run),indent=2))
