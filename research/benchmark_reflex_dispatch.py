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

STRUCTURE. `collect()` needs two vLLM replicas on two GPUs; `summarise()`
needs nothing. The scoring — raw, guarded, union against the static
baseline — is comparison of strings that were already produced, and it
decides whether a generation may be promoted. It used to be interleaved
with the model calls, so it could only ever run on the server.
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


def percentile(values: list[float], p: float) -> float | None:
    """None for an empty sample: "never measured" must not read as "fast"."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * p))], 2)


def summarise(observations: list[dict], *, channel_stats: dict,
              baseline_union_raw: int, elapsed_s: float,
              split: str = 'test') -> dict:
    """Score the recorded per-case observations. Pure.

    One observation per case, each carrying what the selector emitted, which
    pod answered, that pod's answer, the guarded form of it, the target, and
    — when the answer was wrong — what the OTHER pod said. Everything below
    is comparison; nothing here can call a model.
    """
    stats = {
        'n': len(observations),
        'selected_gen5': sum(1 for o in observations
                             if o['pod'] == 'pod:reader-gen5'),
        'selected_gen6': sum(1 for o in observations
                             if o['pod'] == 'pod:reader-gen6'),
        'reflex_raw': 0, 'reflex_guarded': 0, 'union_raw': 0,
        'other_pod_correct': 0,
    }
    for observation in observations:
        raw = observation['answer'] == observation['target']
        guarded = observation['guarded_answer'] == observation['target']
        stats['reflex_raw'] += raw
        stats['reflex_guarded'] += guarded
        if raw:
            stats['union_raw'] += 1
            continue
        # Only a wrong answer is worth asking the other pod about, so
        # `other_answer` is absent for the cases that were already right.
        other_raw = (observation.get('other_answer') is not None
                     and observation['other_answer'] == observation['target'])
        stats['other_pod_correct'] += other_raw
        stats['union_raw'] += other_raw

    latencies = [o['latency_ms'] for o in observations if 'latency_ms' in o]
    stats['reflex_p50_ms'] = percentile(latencies, 0.5)
    stats['reflex_p99_ms'] = percentile(latencies, 0.99)
    stats['channel_stats'] = channel_stats
    stats['unmapped_signals'] = sum(1 for o in observations
                                    if o['alias'] == 'unknown-alias')
    stats['signal_samples'] = [
        {'raw': o.get('raw_signal', '')[:120], 'alias': o['alias']}
        for o in observations[:20]]
    # Named so the split between the two paths cannot be overlooked: the
    # 2026-09-19 run answered all 132 cases from the failover pod.
    stats['answers_from_reflex'] = channel_stats.get('reflex_hits', 0)
    stats['answers_from_failover'] = channel_stats.get('failovers', 0)
    stats['elapsed_s'] = round(elapsed_s, 1)
    return {'status': 'completed', 'split': split,
            'baseline_union_raw': baseline_union_raw, 'metrics': stats}


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
    # One observation per case. Nothing is scored here: the comparison of
    # answers against targets is research-grade arithmetic that decides
    # whether a generation may be promoted, and it belongs somewhere a test
    # can reach it. See summarise().
    observations: list[dict] = []
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

        prefix_q, suffix_q = render_segments(tok_qwen, row)
        prefix_n, suffix_n = render_segments(tok_neo, row)
        prompt_by_pod = {'pod:reader-gen5': prefix_q + suffix_q,
                         'pod:reader-gen6': prefix_n + suffix_n}
        case_started = time.perf_counter()
        result = channel.invoke(alias, {'prompt': prompt_by_pod,
                                        'max_tokens': protocol['evaluation_max_new_tokens']})
        latency_ms = (time.perf_counter() - case_started) * 1000
        answer = result['result']

        observation = {
            'id': row['id'],
            'raw_signal': raw_signal,
            'alias': alias,
            'pod': result['pod'],
            'reflex': result['reflex'],
            'answer': answer,
            'guarded_answer': guarded_answer(row, answer),
            'target': row['target'],
            'latency_ms': latency_ms,
        }
        if answer != row['target']:
            # Only a wrong answer is worth asking the other pod about.
            other = ('pod:reader-gen6' if result['pod'] == 'pod:reader-gen5'
                     else 'pod:reader-gen5')
            observation['other_pod'] = other
            observation['other_answer'] = dispatch(
                other, {'prompt': prompt_by_pod[other],
                        'max_tokens': protocol['evaluation_max_new_tokens']})
        observations.append(observation)

    result = summarise(observations, channel_stats=channel.stats(),
                       baseline_union_raw=126,
                       elapsed_s=time.perf_counter() - started)
    write_evidence(result,
                   Path('research/runs/reflex-dispatch-20260919.json'),
                   __file__, subject=SUBJECT)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
