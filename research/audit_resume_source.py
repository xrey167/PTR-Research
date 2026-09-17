"""Check actual saved prefix and reject modified copies before continuation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from neural_pods.registry import Registry
from research.resume_dialogue import verify_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    read = lambda name: json.loads((args.run/name).read_text(encoding='utf-8'))
    report, linked, manifest = read('failure.json'), read('links.json'), read('manifest.json')
    tokenizer = AutoTokenizer.from_pretrained(manifest['reader_model'], local_files_only=True)
    config = json.loads((Path(manifest['reader_model'])/'generation_config.json').read_text())
    eos = config['eos_token_id']; eos = set(eos if isinstance(eos, list) else [eos]); eos.add(tokenizer.eos_token_id)
    registry = Registry(args.run/'registry.sqlite3')
    try:
        raw = verify_prefix(args.run, report, linked, registry, tokenizer, eos)
        controls = []
        for case in ('order', 'text', 'token', 'model'):
            altered = copy.deepcopy(report)
            if case == 'order': altered['rows'][:2] = reversed(altered['rows'][:2])
            elif case == 'text': altered['rows'][0]['text'] = 'forged answer'
            elif case == 'token': altered['rows'][0]['tokens'][0] += 1
            elif case == 'model': altered['reader_model_sha256'] = 'wrong-model'
            try: verify_prefix(args.run, altered, linked, registry, tokenizer, eos)
            except (AssertionError, ValueError): controls.append(case)
            else: raise AssertionError(f'Altered {case} was accepted')
        result = {'saved_rows_verified': len(report['rows']), 'logit_tensors_verified': len(raw),
                  'tampering_rejected': controls, 'source_status': report['status'],
                  'failure_sha256': hashlib.sha256((args.run/'failure.json').read_bytes()).hexdigest(),
                  'full_run_completed': False, 'independent_inference_replay': False}
        (args.run/'resume-validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    finally: registry.close()


if __name__ == '__main__': main()
