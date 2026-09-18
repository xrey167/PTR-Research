"""Audit BF16 reader outputs and exact input reconstruction against the Int8 run."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
from research.audit_prefix_probe import validate_continuation


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    report=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    protocol=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'))
    assert report['status']=='completed'
    assert report['phase']=='done' and report['base_unchanged'] is True
    assert protocol['dtype']=='bfloat16' and protocol['max_tokens']==64
    runner=a.run/'source_snapshot'/'research'/'probe_reader_precision.py'
    assert hashlib.sha256(runner.read_bytes()).hexdigest()==protocol['runner_sha256']
    source=Path(protocol['source']);data=json.loads((source/'reader.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((source/'reader.json').read_bytes()).hexdigest()==protocol['reference_report_sha256']
    reference=[r for r in data['rows'] if r['reader_invoked']]
    assert [r['id'] for r in report['rows']]==[r['id'] for r in reference]==protocol['cases']
    tok=AutoTokenizer.from_pretrained(protocol['model_path'],local_files_only=True)
    config=json.loads((Path(protocol['model_path'])/'generation_config.json').read_text())
    eos=config['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id)
    logits=load_file(a.run/'first-logits.safetensors');assert set(logits)==set(protocol['cases'])
    db=sqlite3.connect((source/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        changed=[]
        for row,old in zip(report['rows'],reference):
            assert row['question']==old['question'] and row['history']==old['history']
            assert row['int8_text']==old['text'] and row['int8_tokens']==old['tokens']
            s=json.loads(db.execute('SELECT payload FROM nodes WHERE id=?',(old['generation_key'],)).fetchone()[0])['semantic']
            aliases=s['retrieval']['trusted_aliases']
            fact=f"Supplier {aliases[0]} (aliases: {', '.join(aliases[1:])}) has a delivery lead time of {s['object']['value']} days for component {s['component']}."
            template=tok.apply_chat_template([{'role':'system','content':'Use the supplied fact to answer the question. Follow the answer format requested in the question. If the fact is missing, answer UNKNOWN.'},
                {'role':'user','content':'Fact: '+fact+'\nQuestion: QUESTION_BOUNDARY_9'}],tokenize=False,add_generation_prompt=True)
            prefix,tail=template.split('QUESTION_BOUNDARY_9')
            history='\n'.join(m['role'].capitalize()+': '+m['content'] for m in old['history'])
            question=('Earlier dialogue:\n'+history+'\nLatest user question: ' if history else '')+old['question']
            suffix=question+'\nAnswer the latest user question briefly. Use a number of days for a duration, and yes/no with a brief reason for a yes/no question.'+tail
            assert row['prefix_ids']==tok.encode(prefix,add_special_tokens=False)
            assert row['suffix_ids']==tok.encode(suffix,add_special_tokens=False)
            raw=logits[row['id']];assert torch.isfinite(raw).all()
            validate_continuation(row,raw,tok,eos)
            assert len(row['tokens'])<=protocol['max_tokens']
            assert row['eos'] or len(row['tokens'])==protocol['max_tokens']
            if row['tokens']!=old['tokens']:changed.append(row['id'])
    finally:db.close()
    result={'audit_passed':True,'sequences':len(reference),'input_sequences_verified':len(reference),
        'report_sha256':hashlib.sha256((a.run/'report.json').read_bytes()).hexdigest(),
        'logits_sha256':hashlib.sha256((a.run/'first-logits.safetensors').read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256((a.run/'protocol.json').read_bytes()).hexdigest(),
        'runner_snapshot_verified':True,'base_unchanged_reported':True,
        'changed_from_int8':changed,'answer_quality_evaluated':False,'full_research_goal_complete':False,
        'limits':'Saved token/greedy/EOS/input consistency. BF16 versus layerwise Int8 variant, not independent model replay; no routing or lifecycle integration in this control.'}
    (a.run/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
