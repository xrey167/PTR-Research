"""Audit persisted staged integration evidence without rerunning either model.

This verifies consistency and lineage, not independent attestation of inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from neural_pods.registry import digest, verify_files
from research.audit_prefix_probe import validate_continuation


def audit(run):
    read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
    manifest, links, reader = map(read, ['manifest.json', 'links.json', 'capsules.json'])
    assert links['status'] == 'passed' and reader['status'] == 'completed'
    questions = manifest['questions']
    assert len(questions) == 6 and len(set(questions)) == 6
    assert [r['question'] for r in links['rows']] == questions
    assert [r['question'] for r in reader['rows']] == questions
    db = sqlite3.connect((run / 'registry.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        nodes = {key: {'kind': kind, 'payload': json.loads(payload), 'revoked': revoked}
                 for key, kind, payload, revoked in db.execute('SELECT id,kind,payload,revoked FROM nodes')}
        parents = {key: [] for key in nodes}
        for child, parent in db.execute('SELECT child,parent FROM edges'):
            parents[child].append(parent)
        heads = dict(db.execute('SELECT knowledge_key,node_id FROM heads'))
        for key, node in nodes.items():
            assert key == node['kind']+':'+digest({'kind': node['kind'], 'payload': node['payload'], 'parents': sorted(parents[key])})
        def closure(key):
            seen, todo = set(), [key]
            while todo:
                item = todo.pop()
                if item in seen: continue
                seen.add(item); todo.extend(parents[item])
            return seen
        model = Path(manifest['reader_model'])
        tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
        config = json.loads((model/'generation_config.json').read_text())
        eos = config['eos_token_id']
        eos = set(eos if isinstance(eos, list) else [eos]); eos.add(tok.eos_token_id)
        raw = load_file(run/'reader-logits.safetensors')
        assert set(raw) == {f'{i}_{mode}' for i in range(6) for mode in ['cached','reference']}
        correct = 0
        for i, (link, row) in enumerate(zip(links['rows'], reader['rows'])):
            cached, reference = raw[f'{i}_cached'], raw[f'{i}_reference']
            assert torch.isfinite(cached).all() and torch.equal(cached, reference)
            validate_continuation(row, cached, tok, eos)
            assert row['proof_key'] == link['proof_key']
            proof = nodes[row['proof_key']]['payload']['payload']
            assert proof['task'] == 'link_prediction'
            for field in ['question', 'text', 'adapter_key', 'input_tokens', 'output_tokens']:
                assert proof[field] == link[field]
            assert link['adapter_key'] in closure(row['proof_key'])
            receipt = row['receipt']; answer = nodes[receipt['answer_id']]
            assert answer['kind'] == 'answer' and answer['payload']['payload']['text'] == row['text'] == receipt['text']
            assert sorted(receipt['dependencies']) == sorted(parents[receipt['answer_id']])
            assert {link['adapter_key'], link['representation_key'], row['proof_key'], row['artifact_key']} <= set(receipt['dependencies'])
            lineage = closure(receipt['answer_id'])
            assert row['generation_key'] == link['generation_key'] == heads[row['knowledge_key']]
            assert row['generation_key'] in lineage
            for key in lineage:
                node = nodes[key]
                assert not node['revoked']
                acl = node['payload'].get('acl', ['*'])
                assert '*' in acl or 'buyer' in acl
                if node['kind'] == 'knowledge':
                    assert heads[node['payload']['knowledge_key']] == key
            capsule = nodes[row['artifact_key']]['payload']['payload']
            assert row['generation_key'] in parents[row['artifact_key']]
            assert capsule['model_sha256'] == reader['reader_base_sha256']
            directory = (run/'capsule_states'/capsule['directory']).resolve()
            assert directory.parent == (run/'capsule_states').resolve()
            verify_files(directory, capsule['files'])
            adapter = nodes[link['adapter_key']]['payload']['payload']
            directory = (run/'adapters'/adapter['adapter']).resolve()
            assert directory.parent == (run/'adapters').resolve()
            verify_files(directory, adapter['files'])
            expected = str(nodes[row['generation_key']]['payload']['semantic']['object']['value'])
            assert row['expected'] == expected
            actual_correct = row['text'] == expected and row['eos']
            assert row['correct'] == actual_correct
            correct += actual_correct
        assert reader['lookup_regression_passed'] == (correct == 6)
        result = {'audit_passed': True, 'rows': 6, 'correct': correct,
                  'full_first_logits_equal': 6, 'graph_nodes_hash_verified': len(nodes),
                  'receipt_lineage_verified': 6, 'full_research_goal_complete': False,
                  'evidence_sha256': {name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                      for name in ['links.json','capsules.json','reader-logits.safetensors']},
                  'limits': 'Persisted evidence audit; no independent inference replay, no independent replay of runner lifecycle controls; only six existing lookup queries.'}
        (run/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    print(json.dumps(audit(parser.parse_args().run)))
