"""Audit stored planner tokens and distinguish model decisions from guard rejection."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
from tokenizers import Tokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    protocol=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'))
    assert report['status']=='completed'
    path=Path(protocol.get('planner_model',protocol['models']['qwen']['path']))
    tok=Tokenizer.from_file(str(path/'tokenizer.json'))
    config=json.loads((path/'generation_config.json').read_text())
    eos=config['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
    db=sqlite3.connect((a.run/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        semantics=[json.loads(row[0])['semantic'] for row in db.execute('SELECT n.payload FROM nodes n JOIN heads h ON h.node_id=n.id')]
    finally: db.close()
    cases=protocol['cases']['cases'];rows=report['rows']
    assert [r['id'] for r in rows]==[c['id'] for c in cases]
    raw_correct=0;syntax_valid=0;terminated=0
    for row,case in zip(rows,cases):
        raw=row['model'];tokens=raw['tokens']
        assert all(type(t) is int and 0<=t<tok.get_vocab_size() for t in tokens)
        assert tok.decode(tokens,skip_special_tokens=True).strip()==raw['text']
        assert not any(t in eos for t in tokens[:-1])
        ended=bool(tokens and tokens[-1] in eos);terminated+=ended
        match=re.fullmatch(r'ADDRESS ([1-9][0-9]*)',raw['text'])
        if case['expected_subject'] is None:
            correct=raw['text']=='UNKNOWN' and ended
        else:
            labels={s['subject_label'] for s in semantics if s.get('subject')==case['expected_subject'] and 'subject_label' in s}
            entry=next((c for c in raw['planner_payload']['catalogue'] if match and c['address']==int(match[1])),None)
            correct=bool(entry and entry['subject'] in labels and ended)
        syntax_valid+=bool(ended and (raw['text']=='UNKNOWN' or match))
        raw_correct+=correct
        assert row['address_correct']==(row['subject']==case['expected_subject'])
    guarded=sum(r['address_correct'] for r in rows)
    assert guarded==report['address_correct']
    result={'audit_passed':True,'cases':len(rows),'model_decisions_correct':raw_correct,
            'guarded_outcomes_correct':guarded,'syntactically_valid':syntax_valid,'eos_terminated':terminated,
            'full_internal_knowledge_gate_passed':False,
            'limits':'Stored tokenizer/model-output consistency; no inference replay, answer evaluation, or independent catalogue construction replay.'}
    (a.run/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
