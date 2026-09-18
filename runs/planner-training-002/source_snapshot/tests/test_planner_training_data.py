import json
from research.planner_training_data import build_data, FAMILIES


def test_split_identity_and_template_separation_and_balanced_tasks():
    data=build_data()
    assert len(data['train'])==96 and len(data['test'])==32
    assert set(FAMILIES['train'].values()).isdisjoint(FAMILIES['test'].values())
    for field in ['subject','component']:
        collect=lambda split:{c[field] for row in data[split] for c in row['input']['catalogue']}
        assert collect('train').isdisjoint(collect('test'))
    for rows in data.values():
        assert sum(r['target']=='UNKNOWN' for r in rows)==len(rows)//2
        for row in rows:
            assert 'object' not in json.dumps(row['input'])
            if row['target']!='UNKNOWN':
                number=int(row['target'].split()[1])
                assert number in [c['address'] for c in row['input']['catalogue']]
    regression=json.load(open('research/internal-knowledge-cases.json',encoding='utf-8'))
    train_questions={r['input']['dialogue'][-1]['content'] for r in data['train']}
    assert train_questions.isdisjoint(c['question'] for c in regression['cases'])
