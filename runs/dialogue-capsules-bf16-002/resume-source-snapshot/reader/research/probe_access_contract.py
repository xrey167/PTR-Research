"""Measure current deterministic parser coverage, not full model/dialogue quality."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neural_pods.semantics import resolve_query
from neural_pods.registry import InvalidState


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    contract=json.loads(Path('research/internal-knowledge-cases.json').read_text(encoding='utf-8'))
    db=sqlite3.connect((a.run/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        semantics=[json.loads(row[0])['semantic'] for row in db.execute('''SELECT n.payload FROM nodes n
            JOIN heads h ON h.node_id=n.id JOIN semantic_bindings b ON b.generation_key=n.id WHERE n.revoked=0''')]
    finally: db.close()
    rows=[]
    for case in contract['cases']:
        try:
            parsed=resolve_query(case['question'],semantics)
            subject=parsed['subject'];error=None
        except InvalidState as exc:
            subject=None;error=str(exc)
        rows.append({'id':case['id'],'subject':subject,'expected_subject':case['expected_subject'],
            'parser_target_matches':subject==case['expected_subject'],'error':error,
            'history_required':bool(case['history']),'history_supported':False,
            'answer_evaluated':False})
    result={'scope':'Parser only; current parser has no dialogue history interface; no LLM, ANN, ACL or answer check',
            'rows':rows,'parser_target_matches':sum(r['parser_target_matches'] for r in rows),
            'full_internal_knowledge_gate_passed':False}
    target=a.run/'access-parser-diagnostic.json'
    if target.exists():raise FileExistsError('Preserve previous diagnostic')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
