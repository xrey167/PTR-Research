"""P1 reflex dispatch benchmark: the main model (Gen-6 on NeoHorse, GPU1)
emits a pod address signal itself; the ReflexChannel resolves it via the
TemporalPortPlane and dispatches. Compares against the static primary/
fallback baseline (ensemble-hetero-20260919.json) on the frozen test split.

WHAT THE 2026-09-19 RECORDING ACTUALLY SHOWED, and why this script now
records more. Its channel stats were reflex_hits 0, reflex_misses 132,
failovers 132: not one address signal resolved, and all 132 answers came
from the default pod. The quality numbers (union_raw 126) therefore describe
the FAILOVER path, not the reflex — and the gate passed it, because the
check looked at neither the hit count nor the real error counter.

Nothing in the recording said which step failed, so it could not be
diagnosed after the fact. The raw selector output, the alias it was mapped
to, and the per-reason miss counts are now part of the evidence. `errors`,
a key that was initialised to zero and never incremented, is gone: the
channel keeps that counter itself and it is reported from there.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.registry import Registry
from neural_pods.symlink import TemporalPortPlane
from neural_pods.reflex import ReflexChannel
from neural_pods.vllm_router import VllmReplica, VllmReplicaRouter
from research.reader_prompt import render_segments
from research.reader_answer_guard import guarded_answer
from research.evidence import write as write_evidence
from research.train_reader import file_sha, load_bundle

#: The modules these numbers are evidence ABOUT. research/evidence.py
#: hashes them into the report, and the gate refuses the file once any
#: of them changes: a measurement of code that no longer exists is not
#: evidence, however carefully it was recorded.
SUBJECT = [
    "neural_pods/reflex.py",
    "neural_pods/symlink.py",
    "neural_pods/registry.py",
    "neural_pods/vllm_router.py",
]

ALIASES = ["reader-gen5", "reader-gen6"]
SELECTOR_SYSTEM = (
    "You are the controller of a pod system. Available pods: "
    + ", ".join(ALIASES)
    + ". Read the question and decide which pod should answer it. "
      "Reply with exactly one pod name and nothing else."
)


def ask_raw(router, model, prompt, max_tokens):
    result = router.completion({"model": model, "prompt": prompt,
                                "max_tokens": max_tokens, "temperature": 0.0})
    return result["choices"][0]["text"].strip()


def main():
    inputs = Path('runs/reader-training-inputs-generation6')
    primary_router = VllmReplicaRouter([VllmReplica('gpu0', 'http://127.0.0.1:18000')], timeout_s=120)
    fallback_router = VllmReplicaRouter([VllmReplica('gpu1', 'http://127.0.0.1:18001')], timeout_s=120)
    protocol, _config, _ = load_bundle(inputs)
    rows = json.loads((inputs / 'inputs' / 'test.json').read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    tok_qwen = AutoTokenizer.from_pretrained(
        Path(json.loads(Path('runs/reader-training-inputs-generation5/protocol.json').read_text())['model_path']),
        local_files_only=True)
    tok_neo = AutoTokenizer.from_pretrained(Path(protocol['model_path']).resolve(), local_files_only=True)

    # Registry bindings: the pods are published knowledge; aliases point to
    # the dispatch handles (pod keys resolved by the ReflexChannel).
    registry = Registry(':memory:')
    plane = TemporalPortPlane(registry)
    for alias in ALIASES:
        origin = registry.origin('reader_adapters', alias, 1, {'kind': 'reader_lora_adapter'})
        registry.publish(f'adapter:{alias}', {'pod_type': 'lora'}, parents=[origin])
        plane.bind(f'adapter:{alias}', [alias], value_handle=f'pod:{alias}')

    def dispatch(pod_key, request):
        prompt = request['prompt']
        if isinstance(prompt, dict):
            prompt = prompt[pod_key]
        if pod_key == 'pod:reader-gen5':
            return ask_raw(primary_router, 'reader-gen5', prompt, request['max_tokens'])
        return ask_raw(fallback_router, 'reader-gen6', prompt, request['max_tokens'])

    channel = ReflexChannel(plane, dispatch, default_pod='pod:reader-gen5')

    started = time.perf_counter()
    stats = {'n': 0, 'selected_gen5': 0, 'selected_gen6': 0,
             'reflex_raw': 0, 'reflex_guarded': 0, 'union_raw': 0,
             'other_pod_correct': 0}
    # Diagnosable next time: what the model emitted, what it was mapped to,
    # and how often the mapping failed outright.
    signal_samples: list[dict] = []
    unmapped_signals = 0
    latencies = []
    for row in rows:
        # The main model emits its own address signal (in-band, no JSON).
        selector_prompt = (row['question'] + '\n\nWhich pod should answer this '
                           'question? Reply with exactly one name.')
        chat = fallback_router.chat({
            'model': 'neohorse-base',
            'messages': [{'role': 'system', 'content': SELECTOR_SYSTEM},
                         {'role': 'user', 'content': selector_prompt}],
            'max_tokens': 8, 'temperature': 0.0})
        raw_signal = chat['choices'][0]['message']['content'].strip()
        signal = raw_signal.split()[0] if raw_signal.split() else ''
        alias = next((a for a in ALIASES if a in signal), 'unknown-alias')
        if alias == 'unknown-alias':
            unmapped_signals += 1
        if len(signal_samples) < 20:
            signal_samples.append({'raw': raw_signal[:120], 'alias': alias})

        prefix_q, suffix_q = render_segments(tok_qwen, row)
        prefix_n, suffix_n = render_segments(tok_neo, row)
        prompt_by_pod = {'pod:reader-gen5': prefix_q + suffix_q,
                         'pod:reader-gen6': prefix_n + suffix_n}
        case_started = time.perf_counter()
        result = channel.invoke(alias, {'prompt': prompt_by_pod,
                                        'max_tokens': protocol['evaluation_max_new_tokens']})
        latencies.append((time.perf_counter() - case_started) * 1000)
        stats['n'] += 1
        if result['pod'] == 'pod:reader-gen5':
            stats['selected_gen5'] += 1
        else:
            stats['selected_gen6'] += 1
        answer = result['result']
        raw = answer == row['target']
        guarded = guarded_answer(row, answer) == row['target']
        stats['reflex_raw'] += raw
        stats['reflex_guarded'] += guarded
        if not raw:
            other = 'pod:reader-gen6' if result['pod'] == 'pod:reader-gen5' else 'pod:reader-gen5'
            other_answer = dispatch(other, {'prompt': prompt_by_pod[other],
                                            'max_tokens': protocol['evaluation_max_new_tokens']})
            other_raw = other_answer == row['target']
            stats['other_pod_correct'] += other_raw
            stats['union_raw'] += (raw or other_raw)
            stats['reflex_guarded'] += 0
        else:
            stats['union_raw'] += 1
    latencies.sort()
    stats['reflex_p50_ms'] = round(latencies[len(latencies) // 2], 2)
    stats['reflex_p99_ms'] = round(latencies[int(len(latencies) * 0.99)], 2)
    stats['channel_stats'] = channel.stats()
    stats['unmapped_signals'] = unmapped_signals
    stats['signal_samples'] = signal_samples
    stats['answers_from_reflex'] = stats['channel_stats']['reflex_hits']
    stats['answers_from_failover'] = stats['channel_stats']['failovers']
    stats['elapsed_s'] = round(time.perf_counter() - started, 1)
    result = {'status': 'completed', 'split': 'test',
              'baseline_union_raw': 126, 'metrics': stats}
    write_evidence(result,
                   Path('research/runs/reflex-dispatch-20260919.json'),
                   __file__, subject=SUBJECT)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
