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
from research.reader_answer_guard import guarded_answer
from research.train_reader import file_sha, load_bundle


def ask(router, model, prompt, max_tokens):
    result = router.completion({"model": model, "prompt": prompt,
                                "max_tokens": max_tokens, "temperature": 0.0})
    return result["choices"][0]["text"].strip()


def run(inputs, output, port0=18000, port1=18001):
    router = VllmReplicaRouter([VllmReplica("gpu0", f"http://127.0.0.1:{port0}"),
                                VllmReplica("gpu1", f"http://127.0.0.1:{port1}")])
    protocol, _config, _ = load_bundle(inputs)
    rows = json.loads((inputs / 'inputs' / 'test.json').read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    model_path = Path(protocol['model_path']).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    started = time.perf_counter()
    stats = {'n': 0, 'errors': 0, 'gen5_raw': 0, 'gen5_guarded': 0,
             'gen3_raw': 0, 'fallback_used': 0, 'union_raw': 0, 'union_guarded': 0}
    for row in rows:
        prefix, suffix = render_segments(tokenizer, row)
        prompt = prefix + suffix
        try:
            primary = ask(router, 'reader-gen5', prompt, protocol['evaluation_max_new_tokens'])
        except RuntimeError:
            stats['errors'] += 1
            continue
        g5_raw = primary == row['target']
        g5_guarded = guarded_answer(row, primary) == row['target']
        secondary = None
        if not g5_raw:
            stats['fallback_used'] += 1
            secondary = ask(router, 'reader-gen3', prompt, protocol['evaluation_max_new_tokens'])
        g3_raw = (secondary == row['target']) if secondary is not None else False
        stats['n'] += 1
        stats['gen5_raw'] += g5_raw
        stats['gen5_guarded'] += g5_guarded
        stats['gen3_raw'] += g3_raw
        stats['union_raw'] += (g5_raw or g3_raw)
        stats['union_guarded'] += (g5_guarded or (g3_raw and row['target'] == row['target']))
    stats['elapsed_s'] = round(time.perf_counter() - started, 1)
    stats['router_metrics'] = {'requests': router.metrics.requests,
                               'successes': router.metrics.successes,
                               'failovers': router.metrics.failovers}
    result = {'status': 'completed', 'primary': 'reader-gen5', 'fallback': 'reader-gen3',
              'split': 'test', 'metrics': stats}
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('runs/reader-training-inputs-generation5'))
    parser.add_argument('--output', type=Path, default=Path('research/runs/ensemble-20260919.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.output), indent=2))
