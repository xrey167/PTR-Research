"""Verify generation supersession invalidates old reader materializations."""
from __future__ import annotations
import argparse, json, shutil
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.registry import Registry, InvalidState

def run(source):
    source=Path(source); target=source.parent/(source.name+'-generation-check')
    if target.exists(): shutil.rmtree(target)
    target.mkdir(); shutil.copy2(source/'registry.sqlite3',target/'registry.sqlite3')
    row=json.loads((source/'capsules.json').read_text(encoding='utf-8'))['rows'][0]; old=row['generation_key']; answer=row['receipt']['answer_id']
    reg=Registry(target/'registry.sqlite3'); old_node=reg.node(old); sem=deepcopy(old_node['payload']['semantic']); sem['object']['value']=19
    origin=reg.origin('replay-generation-control','supplier-muller-x12','next',{'value':19})
    new=reg.publish(old_node['payload']['knowledge_key'],sem,[origin],'buyer',acl=('buyer',),lifecycle={'owner':'demo-procurement','source':origin,'valid_from':'2026-09-01T00:00:00+00:00'})
    blocked=False
    try: reg.snapshot([answer],'buyer')
    except InvalidState: blocked=True
    result={'old_generation':old,'new_generation':new,'old_answer_id':answer,'old_receipt_blocked':blocked,'new_value':19,'scope':'copied replay registry; generation supersession control'}
    (source/'generation-control.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); return result
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run'); a=p.parse_args(); print(json.dumps(run(a.run),indent=2))
