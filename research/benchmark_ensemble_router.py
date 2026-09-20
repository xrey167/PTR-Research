"""Ensemble serving benchmark: frozen test split through the vLLM multi-LoRA router.

Primary pod is the promoted reader-gen5 adapter; reader-gen3 serves as the
fallback pod for cases the primary misses (verbatim raw match). Reports
per-adapter and union metrics plus router request counts. Prompts are
byte-identical to evaluate_reader.py (prefix+suffix via completions API).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.vllm_router import VllmReplica, VllmReplicaRouter
from research.reader_prompt import render_segments
from research.evidence import write as write_evidence
from research.reader_answer_guard import guarded_answer
from research.train_reader import file_sha, load_bundle


def ask(router, model, prompt, max_tokens):
    result = router.completion({"model": model, "prompt": prompt,
                                "max_tokens": max_tokens, "temperature": 0.0})
    return result["choices"][0]["text"].strip()


#: The modules these numbers are evidence ABOUT.
SUBJECT = ["neural_pods/vllm_router.py"]


def summarise(observations, *, errors: int, elapsed_s: float,
              router_metrics: dict, primary: str, fallback: str,
              split: str = 'test') -> dict:
    """Score the recorded answers. Pure.

    One observation per case: the target, the primary pod's answer, its
    guarded form, and — only when the primary was wrong — the fallback's
    answer. `union_*` is "at least one of the two got it right".

    The guarded union used to read

        g5_guarded or (g3_raw and row['target'] == row['target'])

    whose right-hand comparison is a value against itself and is therefore
    always true. To be precise about what that did and did not cost: the
    expression reduces to `g5_guarded or g3_raw`, so the recorded NUMBERS
    were never wrong. What was wrong is that a reader of a promotion metric
    could not tell which condition was intended, and a clause that cannot be
    false is indistinguishable from one that was meant to be a real guard
    and silently stopped being one. It now says what it computes.
    """
    stats = {'n': len(observations), 'errors': errors,
             'gen5_raw': 0, 'gen5_guarded': 0, 'gen3_raw': 0,
             'fallback_used': 0, 'fallback_errors': 0,
             'union_raw': 0, 'union_guarded': 0}
    for observation in observations:
        target = observation['target']
        g5_raw = observation['primary_answer'] == target
        g5_guarded = observation['primary_guarded'] == target
        # Asked, not answered: a fallback call that raised still counts as
        # used, or `fallback_used` and `n` would describe different sets.
        asked_fallback = not g5_raw
        g3_raw = ('fallback_answer' in observation
                  and observation['fallback_answer'] == target)
        stats['fallback_used'] += asked_fallback
        stats['fallback_errors'] += bool(observation.get('fallback_error'))
        stats['gen5_raw'] += g5_raw
        stats['gen5_guarded'] += g5_guarded
        stats['gen3_raw'] += g3_raw
        stats['union_raw'] += (g5_raw or g3_raw)
        stats['union_guarded'] += (g5_guarded or g3_raw)
    stats['elapsed_s'] = round(elapsed_s, 1)
    stats['router_metrics'] = router_metrics
    return {'status': 'completed', 'primary': primary, 'fallback': fallback,
            'split': split, 'metrics': stats}


def run(inputs, output, port0=18000, port1=18001):
    router = VllmReplicaRouter([VllmReplica("gpu0", f"http://127.0.0.1:{port0}"),
                                VllmReplica("gpu1", f"http://127.0.0.1:{port1}")])
    protocol, _config, _ = load_bundle(inputs)
    rows = json.loads((inputs / 'inputs' / 'test.json').read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    model_path = Path(protocol['model_path']).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    started = time.perf_counter()
    observations = []
    errors = 0
    for row in rows:
        prefix, suffix = render_segments(tokenizer, row)
        prompt = prefix + suffix
        try:
            primary = ask(router, 'reader-gen5', prompt, protocol['evaluation_max_new_tokens'])
        except RuntimeError:
            errors += 1
            continue
        observation = {'id': row['id'], 'target': row['target'],
                       'primary_answer': primary,
                       'primary_guarded': guarded_answer(row, primary)}
        if primary != row['target']:
            # Guarded exactly like the primary call above. VllmReplicaRouter
            # raises RuntimeError once every replica has failed, and this call
            # was bare: one fallback failure aborted run(), discarding every
            # observation collected so far and writing no evidence file at
            # all. benchmark_hetero_ensemble.py already handled it; this did
            # not, in the same commit.
            try:
                observation['fallback_answer'] = ask(
                    router, 'reader-gen3', prompt,
                    protocol['evaluation_max_new_tokens'])
            except RuntimeError:
                errors += 1
                observation['fallback_error'] = True
        observations.append(observation)

    result = summarise(
        observations, errors=errors,
        elapsed_s=time.perf_counter() - started,
        router_metrics={'requests': router.metrics.requests,
                        'successes': router.metrics.successes,
                        'failovers': router.metrics.failovers},
        primary='reader-gen5', fallback='reader-gen3')
    write_evidence(result, Path(output), __file__, subject=SUBJECT)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('runs/reader-training-inputs-generation5'))
    parser.add_argument('--output', type=Path, default=Path('research/runs/ensemble-20260919.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.output), indent=2))
