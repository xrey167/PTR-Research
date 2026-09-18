"""Build DPO preference pairs from the Gen-4 adapter with a verifier reward.

For each training prompt the Gen-4 model samples N continuations at
moderate temperature. The canonical target is the chosen answer; the first
sample that deviates verbatim from the target becomes the rejected answer.
This teaches the policy to prefer the exact canonical formulation over its
own paraphrase drift — the exact weakness measured on the typed split
(74 -> 70 raw exact matches in Gen-4).

Output: frozen pairs file with hashes, so the DPO run is reproducible.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.train_reader import file_sha, load_bundle
from research.reader_prompt import render_segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--adapter-run', type=Path, required=True,
                        help='Policy the DPO run will start from (must match inputs protocol)')
    parser.add_argument('--policy-adapter-run', type=Path,
                        help='Sampling policy for rejected answers (defaults to adapter-run). '
                             'May be an earlier generation; its protocol may differ because '
                             'only the verifier (exact target match) judges the samples.')
    parser.add_argument('--allow-policy-protocol-mismatch', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--samples-per-prompt', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--max-prompts', type=int, default=0,
                        help='0 = all training rows')
    parser.add_argument('--id-substrings', default='',
                        help='comma-separated substrings; keep only rows whose id matches one')
    args = parser.parse_args()

    protocol, config, _ = load_bundle(args.inputs)
    rows = json.loads((args.inputs / 'inputs' / 'train.json').read_text(encoding='utf-8'))
    training_report = json.loads((args.adapter_run / 'report.json').read_text(encoding='utf-8'))
    if training_report['protocol_sha256'] != file_sha(args.inputs / 'protocol.json'):
        raise ValueError('Adapter run does not match the inputs protocol')
    policy_run = args.policy_adapter_run or args.adapter_run
    policy_report = json.loads((policy_run / 'report.json').read_text(encoding='utf-8'))
    if policy_run != args.adapter_run:
        if not args.allow_policy_protocol_mismatch:
            raise ValueError('Pass --allow-policy-protocol-mismatch when the sampling policy '
                             'was trained on a different generation of inputs')
        # Cross-generation policies are legitimate: chosen is always the
        # canonical target and the verifier judges every sample, so a weaker
        # policy only contributes more informative rejected answers.
    for name, expected in policy_report['adapter_files'].items():
        if file_sha(policy_run / 'adapter' / name) != expected:
            raise ValueError('Adapter file mismatch')

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in ('build_dpo_pairs.py', 'train_reader.py', 'reader_prompt.py',
                 'reader_answer_guard.py', 'progress_json.py'):
        shutil.copyfile(Path(__file__).parent / name, output / name)
        sources[name] = file_sha(output / name)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.set_num_threads(4)
    model_path = Path(protocol['model_path']).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                 dtype=torch.bfloat16, attn_implementation='eager')
    model = PeftModel.from_pretrained(model, policy_run / 'adapter')
    model.to(args.device)
    model.eval().requires_grad_(False)
    model.config.use_cache = True
    if args.id_substrings:
        needles = [x.strip() for x in args.id_substrings.split(',') if x.strip()]
        rows = [r for r in rows if any(n in r['id'] for n in needles)]
    if args.max_prompts:
        rows = rows[:args.max_prompts]

    started = time.perf_counter()
    pairs = []
    stats = {'prompts': 0, 'with_rejected': 0, 'identical_all': 0}
    for row in rows:
        prefix, suffix = render_segments(tokenizer, row)
        prompt_ids = (tokenizer.encode(prefix, add_special_tokens=False)
                      + tokenizer.encode(suffix, add_special_tokens=False))
        input_ids = torch.tensor([prompt_ids], device=args.device)
        generated = model.generate(
            input_ids, do_sample=True, temperature=args.temperature, top_p=0.95,
            max_new_tokens=48, min_new_tokens=2, pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=args.samples_per_prompt)
        samples = [tokenizer.decode(g[len(prompt_ids):], skip_special_tokens=True).strip()
                   for g in generated]
        target = row['target'].strip()
        rejected = next((s for s in samples if s and s != target), None)
        stats['prompts'] += 1
        if rejected is None:
            stats['identical_all'] += 1
            continue
        stats['with_rejected'] += 1
        pairs.append({'id': row['id'], 'prompt_ids': prompt_ids,
                      'chosen': target, 'rejected': rejected})
        if stats['prompts'] % 50 == 0:
            print(json.dumps({'progress': stats['prompts'],
                              'elapsed_s': round(time.perf_counter() - started, 1)}), flush=True)

    pairs_path = output / 'pairs.json'
    pairs_path.write_text(json.dumps(pairs, ensure_ascii=False), encoding='utf-8')
    report = {
        'status': 'pairs_completed',
        'inputs': str(args.inputs), 'adapter_run': str(args.adapter_run),
        'policy_adapter_run': str(policy_run),
        'protocol_sha256': file_sha(args.inputs / 'protocol.json'),
        'adapter_report_sha256': file_sha(args.adapter_run / 'report.json'),
        'sources': sources,
        'samples_per_prompt': args.samples_per_prompt, 'temperature': args.temperature,
        'pairs': len(pairs), 'stats': stats,
        'pairs_sha256': file_sha(pairs_path),
        'elapsed_s': round(time.perf_counter() - started, 1),
        'scope': ('Chosen is the canonical training target, rejected is the policy '
                  'sample deviating verbatim; deterministic verifier = exact match.'),
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['stats'] | {'pairs': len(pairs)}))


if __name__ == '__main__':
    main()
