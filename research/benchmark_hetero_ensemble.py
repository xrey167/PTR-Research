"""Heterogeneous ensemble benchmark: Gen-5 (qwen3b, GPU0) as primary and
Gen-6 (NeoHorse-1-4B, GPU1) as fallback — two different base models behind
the router. Union metrics measure complementary correction on the frozen
test split.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.vllm_router import VllmReplica, VllmReplicaRouter
from research.reader_prompt import render_segments
from research.reader_answer_guard import guarded_answer
from research.train_reader import file_sha, load_bundle


def ask(router, model, prompt, max_tokens):
    result = router.completion({"model": model, "prompt": prompt,
                                "max_tokens": max_tokens, "temperature": 0.0})
    return result["choices"][0]["text"].strip()


#: The modules these numbers are evidence ABOUT.
SUBJECT = ["neural_pods/vllm_router.py"]


def summarise(observations, *, errors: int, elapsed_s: float,
              primary: str, fallback: str, split: str = 'test') -> dict:
    """Score the recorded answers of the heterogeneous ensemble. Pure.

    `fallback_used` counts the cases where the primary was wrong and the
    fallback was therefore asked — including the ones where asking it
    failed, which is why `fallback_error` is carried. Previously a failed
    fallback call skipped the case entirely AFTER incrementing
    `fallback_used`, so that counter could exceed `n` and the two numbers
    described different sets of cases.
    """
    stats = {'n': len(observations), 'errors': errors,
             'primary_raw': 0, 'primary_guarded': 0, 'fallback_raw': 0,
             'fallback_used': 0, 'fallback_errors': 0,
             'union_raw': 0, 'union_guarded': 0}
    for observation in observations:
        target = observation['target']
        p_raw = observation['primary_answer'] == target
        p_guarded = observation['primary_guarded'] == target
        asked_fallback = not p_raw
        f_raw = ('fallback_answer' in observation
                 and observation['fallback_answer'] == target)
        stats['fallback_used'] += asked_fallback
        stats['fallback_errors'] += bool(observation.get('fallback_error'))
        stats['primary_raw'] += p_raw
        stats['primary_guarded'] += p_guarded
        stats['fallback_raw'] += f_raw
        stats['union_raw'] += (p_raw or f_raw)
        stats['union_guarded'] += (p_guarded or f_raw)
    stats['elapsed_s'] = round(elapsed_s, 1)
    return {'status': 'completed', 'primary': primary, 'fallback': fallback,
            'split': split, 'metrics': stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('runs/reader-training-inputs-generation6'))
    parser.add_argument('--primary', default='reader-gen5')
    parser.add_argument('--fallback', default='reader-gen6')
    parser.add_argument('--output', type=Path, default=Path('research/runs/ensemble-hetero-20260919.json'))
    args = parser.parse_args()

    primary_router = VllmReplicaRouter([VllmReplica('gpu0', 'http://127.0.0.1:18000')], timeout_s=120)
    fallback_router = VllmReplicaRouter([VllmReplica('gpu1', 'http://127.0.0.1:18001')], timeout_s=120)
    protocol, _config, _ = load_bundle(args.inputs)
    rows = json.loads((args.inputs / 'inputs' / 'test.json').read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    qwen_protocol = json.loads(Path('runs/reader-training-inputs-generation5/protocol.json').read_text(encoding='utf-8'))
    tokenizer_qwen = AutoTokenizer.from_pretrained(Path(qwen_protocol['model_path']).resolve(), local_files_only=True)
    tokenizer_neo = AutoTokenizer.from_pretrained(Path(protocol['model_path']).resolve(), local_files_only=True)

    started = time.perf_counter()
    observations = []
    errors = 0
    for row in rows:
        prefix, suffix = render_segments(tokenizer_qwen, row)
        try:
            primary = ask(primary_router, args.primary, prefix + suffix, protocol['evaluation_max_new_tokens'])
        except RuntimeError:
            errors += 1
            continue
        observation = {'id': row['id'], 'target': row['target'],
                       'primary_answer': primary,
                       'primary_guarded': guarded_answer(row, primary)}
        if primary != row['target']:
            try:
                prefix_n, suffix_n = render_segments(tokenizer_neo, row)
                observation['fallback_answer'] = ask(
                    fallback_router, args.fallback, prefix_n + suffix_n,
                    protocol['evaluation_max_new_tokens'])
            except RuntimeError:
                # The case is still recorded: the primary DID answer it, and
                # dropping it made `fallback_used` count cases that `n` did
                # not, so the two numbers described different runs.
                errors += 1
                observation['fallback_error'] = True
        observations.append(observation)

    result = summarise(observations, errors=errors,
                       elapsed_s=time.perf_counter() - started,
                       primary=args.primary, fallback=args.fallback)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
