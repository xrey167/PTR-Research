"""Real, unconstrained generation on a frozen reader development or test split.

Run base and adapter separately to avoid two 3B copies on this host. This is a
reader-only evaluation, not proof of runtime provenance or compact neural state.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.train_reader import file_sha, load_bundle
from research.reader_prompt import render_segments
from research.reader_answer_guard import guarded_answer
from research.progress_json import write_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--adapter-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--ids-file', type=Path,
                        help='Optional newline-delimited case IDs for a bounded diagnostic evaluation')
    args = parser.parse_args()
    protocol, config, _ = load_bundle(args.inputs)
    rows_path = args.inputs / 'inputs' / (args.split + '.json')
    rows = json.loads(rows_path.read_text(encoding='utf-8'))
    if args.ids_file:
        ids = {line.strip().lstrip('\ufeff') for line in args.ids_file.read_text(encoding='utf-8').splitlines() if line.strip()}
        rows = [row for row in rows if row['id'] in ids]
        if len(rows) != len(ids):
            raise ValueError('ids-file contains an unknown case ID')
    training_report = None
    if args.adapter_run:
        training_report = json.loads((args.adapter_run / 'report.json').read_text(encoding='utf-8'))
        if (training_report['status'] != 'trained_not_evaluated'
                or training_report['protocol_sha256'] != file_sha(args.inputs / 'protocol.json')):
            raise ValueError('Adapter run does not match the frozen training protocol')
        for name, expected in training_report['adapter_files'].items():
            target = (args.adapter_run / 'adapter' / name).resolve()
            if not target.is_relative_to((args.adapter_run / 'adapter').resolve()) or file_sha(target) != expected:
                raise ValueError('Adapter file mismatch')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    sources = {}
    for name in ('evaluate_reader.py', 'train_reader.py', 'reader_prompt.py', 'reader_answer_guard.py',
                 'reader_identity.py', 'prefix_capsule.py', 'progress_json.py'):
        shutil.copyfile(Path(__file__).parent / name, output / name)
        sources[name] = file_sha(output / name)
    state = {'status': 'running', 'phase': 'verify_model', 'rows': [],
             'split': args.split, 'variant': 'adapter' if training_report else 'base',
             'protocol_sha256': file_sha(args.inputs / 'protocol.json'),
             'cases_sha256': file_sha(output / 'cases.json'), 'sources': sources,
             'source_adapter_report_sha256': file_sha(args.adapter_run / 'report.json') if training_report else None,
             'max_new_tokens': protocol['evaluation_max_new_tokens'],
             'attention_implementation': 'eager', 'precision': 'bfloat16', 'device': args.device,
             'full_research_goal_complete': False,
             'scope': 'Reader-only supplied-fact generation; no routing/lifecycle or no-runtime-text claim',
             'scoring': 'Exact target match is a format metric only; semantic quality needs separate review'}
    started = time.perf_counter()
    def save():
        state['elapsed_s'] = time.perf_counter() - started
        write_progress(output / 'report.json', state)
    save()
    registry = None
    try:
        model_path = Path(protocol['model_path']).resolve()
        manifest = json.loads((args.inputs / 'inputs/model-manifest.json').read_text(encoding='utf-8'))
        for name, metadata in manifest['files'].items():
            target = (model_path / name).resolve()
            if not target.is_relative_to(model_path) or file_sha(target) != metadata['sha256']:
                raise ValueError('Base model file mismatch')
        import psutil
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from neural_pods.registry import Registry
        from research.reader_identity import reader_identity
        from research.prefix_capsule import PrefixCapsules
        torch.set_num_threads(4)
        process = psutil.Process()
        state.update(phase='load', pid=process.pid, model_files_verified=True)
        save()
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation='eager')
        if training_report:
            model = PeftModel.from_pretrained(model, args.adapter_run / 'adapter')
        model.to(args.device)
        model.eval().requires_grad_(False)
        model.config.use_cache = True
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        state['phase'] = 'initial_hash'
        save()
        identity = reader_identity(model)
        if training_report and identity != training_report['reader_identity']:
            raise ValueError('Reloaded reader identity differs from the training artifact')
        state['reader_identity'] = identity
        eos_ids = model.generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        eos_ids.add(tokenizer.eos_token_id)
        registry = Registry(':memory:')
        capsules = PrefixCapsules(registry, model, output / 'unused_capsules', identity['sha256'])
        state['forward_calls'] = 0
        last_heartbeat = time.perf_counter()
        def heartbeat(_module, _inputs, _output):
            nonlocal last_heartbeat
            state['forward_calls'] += 1
            if time.perf_counter() - last_heartbeat >= 30:
                state['rss_bytes'] = process.memory_info().rss
                try:
                    save()
                except PermissionError:
                    state['progress_write_failures'] = state.get('progress_write_failures', 0) + 1
                last_heartbeat = time.perf_counter()
        hook = model.register_forward_hook(heartbeat)
        raw = {}
        for row in rows:
            state.update(current_case=row['id'], phase='prefill')
            save()
            # render_segments reads only evidence/history/question, never target or assessment.
            prefix, suffix = render_segments(tokenizer, row)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').to(args.device)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False, return_tensors='pt').to(args.device)
            if prefix_ids.numel() == 0 or suffix_ids.numel() == 0:
                raise ValueError('Tokenizer produced an empty reader prompt segment')
            case_started = time.perf_counter()
            cache = capsules.prefill(prefix_ids)
            state['phase'] = 'decode'
            save()
            result = capsules.decode(suffix_ids, cache, eos_ids, max_tokens=state['max_new_tokens'])
            if not torch.isfinite(result['logits']).all():
                raise ArithmeticError('Nonfinite first-token reader logits')
            answer = tokenizer.decode(result['tokens'], skip_special_tokens=True).strip()
            guarded = guarded_answer(row, answer)
            state['rows'].append({'id': row['id'], 'text': answer, 'guarded_text': guarded,
                'guarded_exact_target_match': guarded == row['target'], 'tokens': result['tokens'],
                'eos': result['eos'], 'exact_target_match': answer == row['target'],
                'prefix_ids': prefix_ids[0].tolist(), 'suffix_ids': suffix_ids[0].tolist(),
                'seconds': time.perf_counter() - case_started})
            raw[row['id']] = result['logits']
            save_file(raw, output / 'first-logits.safetensors')
            save()
            print(json.dumps({'case': row['id'], 'text': answer}, ensure_ascii=False), flush=True)
            del cache, result
        hook.remove()
        state['phase'] = 'final_hash'
        save()
        if reader_identity(model) != identity:
            raise AssertionError('Reader changed during evaluation')
        state.update(status='completed', phase='done', reader_unchanged=True,
                     exact_target_matches=sum(r['exact_target_match'] for r in state['rows']),
                     guarded_exact_target_matches=sum(r['guarded_exact_target_match'] for r in state['rows']),
                     total=len(rows), first_logits_sha256=file_sha(output / 'first-logits.safetensors'))
        save()
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        save()
        raise
    finally:
        if registry is not None:
            registry.close()


if __name__ == '__main__':
    main()
