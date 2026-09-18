import json
from research.planner_training_data import build_data, FAMILIES


def test_split_identity_and_template_separation_and_balanced_tasks():
    data=build_data()
    assert len(data['train'])==100 and len(data['test'])==50
    assert set(FAMILIES['train'].values()).isdisjoint(FAMILIES['test'].values())
    for field in ['subject','component']:
        collect=lambda split:{c[field] for row in data[split] for c in row['input']['catalogue']}
        assert collect('train').isdisjoint(collect('test'))
    for rows in data.values():
        assert sum(r['target']=='UNKNOWN' for r in rows)==len(rows)*2//5
        for size in [2,3]:
            from collections import Counter
            counts=Counter(r['selected_address'] for r in rows if len(r['input']['catalogue'])==size)
            assert set(counts)==set(range(1,size+1)) and len(set(counts.values()))==1
        for family in ['unknown_part','unknown_name']:
            absent=[r for r in rows if r['family']==family]
            present=[r for r in rows if r['family']==family+'_present']
            assert len(absent)==len(present) and all(r['target']!='UNKNOWN' for r in present)
        for row in rows:
            assert 'object' not in json.dumps(row['input'])
            if row['target']!='UNKNOWN':
                number=int(row['target'].split()[1])
                assert number in [c['address'] for c in row['input']['catalogue']]
    regression=json.load(open('research/internal-knowledge-cases.json',encoding='utf-8'))
    train_questions={r['input']['dialogue'][-1]['content'] for r in data['train']}
    assert train_questions.isdisjoint(c['question'] for c in regression['cases'])
