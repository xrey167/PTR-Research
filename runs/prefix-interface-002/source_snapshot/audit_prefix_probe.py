"""Recount stored outputs and compare complete saved logits without inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import load_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    data = json.loads((a.run / 'results.json').read_text(encoding='utf-8'))
    assert data['status'] == 'completed'
    protocol = json.loads((a.run / 'protocol.json').read_text(encoding='utf-8'))
    logits = load_file(a.run / 'first_logits.safetensors')
    groups = {}
    scores = {mode: 0 for mode in protocol['modes']}
    for i, row in enumerate(data['rows']):
        raw = logits[str(i)]
        assert torch.isfinite(raw).all()
        assert hashlib.sha256(raw.numpy().tobytes()).hexdigest() == row['first_logits_sha256']
        key = (row['entity'], row['value'], row['operator'])
        assert row['mode'] not in groups.setdefault(key, {})
        groups[key][row['mode']] = (row, raw)
        value = row['value']
        if row['mode'] == 'wrong_value_prefix':
            values = protocol['values']
            value = values[(values.index(value) + 1) % len(values)]
        target = {'lookup':str(value), 'plus_two':str(value + 2), 'above_25':'yes' if value > 25 else 'no'}[row['operator']]
        if row['mode'] == 'no_fact': target = 'UNKNOWN'
        correct = row['text'].lower() == target.lower() and row['eos']
        assert row['expected'] == target and row['correct'] == correct
        scores[row['mode']] += int(correct)
    assert len(groups) == 24 and len(data['rows']) == 96
    changed = 0
    for group in groups.values():
        assert set(group) == set(protocol['modes'])
        (fresh, fl), (saved, sl) = group['fresh_text_prefix'], group['saved_prefix']
        assert torch.equal(fl, sl) and fresh['tokens'] == saved['tokens']
        changed += saved['tokens'] != group['wrong_value_prefix'][0]['tokens']
    for mode, count in scores.items(): assert count == data['scores'][mode]['correct']
    result = {'audit_passed':True, 'sequences':96, 'exact_logits_and_tokens':24,
              'correct':scores, 'wrong_state_changes':changed,
              'scientific_gate_passed':data['scientific_gate_passed'],
              'limitation':'Recounts saved evidence; lifecycle and weight hashes are runner assertions, not independently replayed here.'}
    (a.run / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__': main()
