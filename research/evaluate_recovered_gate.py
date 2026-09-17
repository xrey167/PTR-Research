"""Run reviewed recovered pure-data gate against synthetic controls.

The missing historical reducer is not replaced or mocked. Its comparative
claims remain unverified; only the recovered strict gate is exercised here.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    root = Path(__file__).resolve().parent
    fragments = root / 'recovered-fragments-002'
    gate_path = fragments / 'entry_8532_0.py'
    fixture_path = fragments / 'entry_8574_1.py'
    gate = module(gate_path, 'recovered_strict_gate')
    fixtures = module(fixture_path, 'recovered_fixtures')
    cases, base = fixtures.fixture()
    rows = []
    def check(name, records, expected):
        verdict = gate.evaluate(records, cases, 16, {15})
        rows.append({'name':name,'expected':expected,'verdict':verdict})
        assert verdict['cqta2_e0_gate'] is expected, (name, verdict)
    check('valid_fixture', base, True)
    for name, records in fixtures.mutations().items():
        check(name, records, False)
    shuffled = copy.deepcopy(base)
    random.Random(42).shuffle(shuffled)
    check('order_independent_pairing', shuffled, True)
    for name, field, value in [('declared_fact_leak','knowledge_text_tokens_student',1),
                                ('broken_prompt_concatenation','student_input_ids',[5]),
                                ('greedy_mismatch','tokens',[2,15])]:
        changed = copy.deepcopy(base)
        changed[2][field] = value
        check(name, changed, False)
    changed = copy.deepcopy(base)
    changed[2].update(tokens=[1,14], eos_seen=False)
    check('missing_eos', changed, False)
    changed = copy.deepcopy(base)
    changed[2]['first_logits'][0] = -0.0
    check('signed_zero_mismatch', changed, False)
    report = {'status':'passed','checks':rows,'synthetic_only':True,'new_model_sequences':0,
              'missing_historical_reducer_comparison':'not_run',
              'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [gate_path,fixture_path]}}
    destination = root / 'recovered-gate-evaluation.json'
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','checks':len(rows),'new_model_sequences':0}))


if __name__ == '__main__': main()
