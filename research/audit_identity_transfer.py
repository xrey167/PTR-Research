"""Recount stored transfer links and verify graph hashes; no inference replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.registry import digest, verify_files


def audit(run):
    report = json.loads((run/'report.json').read_text(encoding='utf-8'))
    assert report['status']=='completed'
    db = sqlite3.connect((run/'registry.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        nodes = {key: (kind,json.loads(payload)) for key,kind,payload in db.execute('SELECT id,kind,payload FROM nodes')}
        parents = {key:[] for key in nodes}
        for child,parent in db.execute('SELECT child,parent FROM edges'): parents[child].append(parent)
        for key,(kind,payload) in nodes.items():
            assert key==kind+':'+digest({'kind':kind,'payload':payload,'parents':sorted(parents[key])})
        def closure(key):
            seen, todo = set(), [key]
            while todo:
                key = todo.pop()
                if key in seen: continue
                seen.add(key); todo.extend(parents[key])
            return seen
        rows = report['predictions']
        assert Counter(r['stage'] for r in rows)=={'before':6,'after':6}
        assert [r['question'] for r in rows[:6]]==[r['question'] for r in rows[6:]]
        assert len(set(r['representation_key'] for r in rows))==1
        assert len(set(r['adapter_key'] for r in rows))==1
        generations = {r['generation_key'] for r in rows}
        assert len(generations)==2
        rep = rows[0]['representation_key']
        assert not generations & closure(rep)
        payload = nodes[rep][1]['payload']
        assert payload['schema']=='dragonfly-identity-address:research-v1'
        identity = payload['identity_key']
        assert identity in closure(rep)
        assert 'object' not in nodes[identity][1]['semantic']
        adapter = rows[0]['adapter_key']; adapter_payload = nodes[adapter][1]['payload']
        assert identity in closure(adapter)
        path = (run/'adapters'/adapter_payload['adapter']).resolve()
        assert path.parent==(run/'adapters').resolve()
        verify_files(path,adapter_payload['files'])
        target = f"LINK {adapter_payload['link_slot']} CLUSTER {adapter_payload['cluster_slot']}"
        for row in rows:
            assert row['text']==target
            receipt=row['receipt']; key=receipt['answer_id']
            assert nodes[key][1]['payload']['text']==row['text']==receipt['text']
            assert sorted(receipt['dependencies'])==sorted(parents[key])
            assert {rep,adapter,identity,row['generation_key']} <= closure(key)
            if row['stage']=='after': assert row['generation_key']==report['new_generation']
        assert nodes[report['new_generation']][1]['semantic']['object']['value']==27
        result={'audit_passed':True,'links_before':6,'links_after':6,'graph_nodes_verified':len(nodes),
                'one_identity_representation':True,'one_link_adapter':True,'fact_generations_not_in_address_lineage':True,
                'factual_answer_generation_tested':False,'full_research_goal_complete':False,
                'limitations':'Stored text/graph/file consistency; no raw token audit or independent model/gradient replay. Fixture is existing lookup questions.'}
        (run/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    finally: db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    print(json.dumps(audit(parser.parse_args().run)))
