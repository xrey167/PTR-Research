"""Audit saved planner experiment; token/target counts, split and artifact hashes."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
from neural_pods.registry import digest,verify_files


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    read=lambda name:json.loads((a.run/name).read_text(encoding='utf-8'))
    report,protocol,data=map(read,['report.json','protocol.json','dataset.json'])
    assert report['status']=='completed'
    assert hashlib.sha256((a.run/'dataset.json').read_bytes()).hexdigest()==protocol['dataset_sha256']
    assert len(report['losses'])==protocol['steps']==len(data['train'])*protocol['epochs']
    assert all(math.isfinite(x) for x in report['losses'])
    for field in ['subject','component']:
        collect=lambda split:{c[field] for row in data[split] for c in row['input']['catalogue']}
        assert collect('train').isdisjoint(collect('test'))
    assert {r['input']['dialogue'][-1]['content'] for r in data['train']}.isdisjoint(r['input']['dialogue'][-1]['content'] for r in data['test'])
    tok=Tokenizer.from_file(str(Path(protocol['model']['path'])/'tokenizer.json'))
    config=json.loads((Path(protocol['model']['path'])/'generation_config.json').read_text())
    eos=config['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
    scores={};families={}
    for mode in ['baseline','trained']:
        rows=report[mode];assert [r['id'] for r in rows]==[r['id'] for r in data['test']]
        counts=defaultdict(lambda:{'correct':0,'total':0})
        for row,example in zip(rows,data['test']):
            tokens=row['tokens']
            assert tokens and all(type(t) is int and 0<=t<tok.get_vocab_size() for t in tokens)
            assert not any(t in eos for t in tokens[:-1])
            ended=tokens[-1] in eos
            assert ended==row['eos']
            text=tok.decode(tokens,skip_special_tokens=True).strip()
            assert text==row['text'] and row['target']==example['target']
            correct=text==example['target'] and ended
            assert correct==row['correct']
            counts[example['family']]['correct']+=correct;counts[example['family']]['total']+=1
        scores[mode]=sum(c['correct'] for c in counts.values());families[mode]=dict(counts)
        assert scores[mode]==report[mode+'_correct']
    assert report['generalization_gate_passed']==(scores['trained']==len(data['test']))
    db=sqlite3.connect((a.run/'registry.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        nodes={key:(kind,json.loads(payload)) for key,kind,payload in db.execute('SELECT id,kind,payload FROM nodes')}
        for key,(kind,payload) in nodes.items():
            parents=sorted(r[0] for r in db.execute('SELECT parent FROM edges WHERE child=?',(key,)))
            assert key==kind+':'+digest({'kind':kind,'payload':payload,'parents':parents})
        payload=nodes[report['planner_artifact']][1]['payload']
        path=(a.run/'adapter'/payload['adapter']).resolve()
        assert path.parent==(a.run/'adapter').resolve()
        verify_files(path,payload['files'])
        assert payload['base_sha256']==protocol['expected_base_sha256']
        assert payload['protocol_sha256']==hashlib.sha256((a.run/'protocol.json').read_bytes()).hexdigest()
    finally:db.close()
    result={'audit_passed':True,'scores':scores,'families':families,'sequences_checked':2*len(data['test']),
        'generalization_gate_passed':report['generalization_gate_passed'],'full_research_goal_complete':False,
        'limits':'Saved token/target/split/hash consistency. No independent training or inference replay; synthetic small English task, not end-to-end knowledge use.'}
    (a.run/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
