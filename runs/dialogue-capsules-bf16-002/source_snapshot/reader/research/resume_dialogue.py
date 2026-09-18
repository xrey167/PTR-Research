"""Verify a completed prefix of an interrupted dialogue reader run."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from neural_pods.registry import digest, verify_files
from research.audit_prefix_probe import validate_continuation


def verify_prefix(directory, report, linked, registry, tokenizer, eos_ids):
    directory = Path(directory)
    assert report['status'] == 'failed'
    rows = report['rows']
    assert rows and len(rows) < len(linked['rows'])
    assert [r['id'] for r in rows] == [r['id'] for r in linked['rows'][:len(rows)]]
    sources = directory / report['source_snapshot']
    source_hashes = json.loads((sources / 'manifest.json').read_text(encoding='utf-8'))
    verify_files(sources, {**source_hashes, 'manifest.json': report['source_manifest_sha256']})
    raw = load_file(directory / 'reader-logits.safetensors')
    expected_logits = set()
    checked = set()
    for row, original in zip(rows, linked['rows']):
        assert all(row.get(key) == value for key, value in original.items())
        receipt = row['receipt']
        registry.snapshot(receipt['dependencies'], 'buyer')
        answer = registry.node(receipt['answer_id'])
        assert answer['payload']['payload']['text'] == receipt['text'] == row['text']
        assert sorted(receipt['dependencies']) == answer['payload']['parent_artifact_keys']
        for node in registry.ancestors(receipt['answer_id']):
            if node['id'] in checked:
                continue
            parents = sorted(x[0] for x in registry.db.execute('SELECT parent FROM edges WHERE child=?', (node['id'],)))
            assert node['id'] == node['kind'] + ':' + digest({'kind': node['kind'], 'payload': node['payload'], 'parents': parents})
            checked.add(node['id'])
        if not row['reader_invoked']:
            assert row['deferred'] and row['text'] == 'UNKNOWN'
            continue
        cap = registry.node(row['capsule_key'])
        payload = cap['payload']['payload']
        assert payload['model_sha256'] == report['reader_model_sha256']
        assert row['generation_key'] in cap['payload']['parent_artifact_keys']
        verify_files(directory / 'capsule_states' / payload['directory'], payload['files'])
        cached, reference = row['id'] + '_cached', row['id'] + '_reference'
        expected_logits.update((cached, reference))
        assert torch.isfinite(raw[cached]).all() and torch.equal(raw[cached], raw[reference])
        validate_continuation(row, raw[cached], tokenizer, eos_ids)
    assert set(raw) == expected_logits
    return raw
