"""Apply the typed answer barrier to the complete reader holdout."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.reader_answer_guard import guarded_answer
from research.prepare_multihop_extension import rows as three_rows

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--evaluation',type=Path,required=True); p.add_argument('--cases',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    ev=json.loads(a.evaluation.read_text(encoding='utf-8')); cases=json.loads(a.cases.read_text(encoding='utf-8')) + three_rows('test'); lookup={x['id']:x for x in cases}; result={}
    for name in ('two-hop-test','three-hop-test'):
        rows=[]
        for x in ev[name]['rows']:
            row=lookup[x['id']]; guarded=guarded_answer(row,x['answer']); rows.append({'id':x['id'],'raw':x['answer'],'guarded':guarded,'target':x['target'],'raw_exact':x['answer']==x['target'],'guarded_exact':guarded==x['target']})
        result[name]={'raw_exact':sum(x['raw_exact'] for x in rows),'guarded_exact':sum(x['guarded_exact'] for x in rows),'total':len(rows),'rows':rows}
    out={'schema':'full-reader-guard-audit:v1','adapter':ev.get('adapter'),'two-hop-test':result['two-hop-test'],'three-hop-test':result['three-hop-test'],'passed':all(result[x]['guarded_exact']==result[x]['total'] for x in result)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps({k:{'raw':v['raw_exact'],'guarded':v['guarded_exact'],'total':v['total']} for k,v in result.items()}))
if __name__=='__main__': main()
