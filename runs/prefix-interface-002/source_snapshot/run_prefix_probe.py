"""No-gradient real-model interface probe with counterfactual and lifecycle controls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
from neural_pods.registry import Registry, InvalidState
from research.prefix_capsule import PrefixCapsules, weights_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('models/qwen'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--typed-format', action='store_true', help='Exploratory prompt-only follow-up with per-question output type')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = {'schema': 'local-prefix-interface:v1', 'entities': ['Neral', 'Vost'],
                'values': [18, 24, 31, 42], 'operators': ['lookup', 'plus_two', 'above_25'],
                'modes': ['fresh_text_prefix', 'saved_prefix', 'wrong_value_prefix', 'no_fact'],
                'max_tokens': 12, 'training_steps': 0, 'reconstruction_not_original_cqp1': True,
                'typed_format': a.typed_format,
                'gate': 'All full first logits and complete tokens equal for fresh/saved; all answers including EOS correct; wrong states follow wrong values.',
                'token_boundary': 'Prefix and question tokenized separately; reference uses the same boundary and cache execution schedule.'}
    (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    start = time.perf_counter()
    result = {'status': 'running', 'rows': [], 'lifecycle': {}, 'fictional_data': True}
    def save():
        result['elapsed_s'] = time.perf_counter() - start
        (a.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    save()
    reg = Registry(a.output / 'registry.sqlite3')
    try:
        torch.set_num_threads(4)
        torch.manual_seed(20260915)
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, trust_remote_code=False,
                     dtype=torch.float32, attn_implementation='sdpa').eval()
        model.requires_grad_(False)
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
        base_hash = weights_hash(model)
        capsules = PrefixCapsules(reg, model, a.output / 'capsules', base_hash)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        eos.add(tok.eos_token_id)
        system = 'Use the supplied fictional facts. Answer with only the requested number or yes/no. If no fact is provided, answer UNKNOWN.'
        if a.typed_format:
            system = 'Use the supplied fact to answer the question. Follow the answer format requested in the question. If the fact is missing, answer UNKNOWN.'
        def prompt_parts(entity, value, question):
            marker = 'QUESTION_BOUNDARY_9'
            content = f'Fact: Supplier {entity} has a delivery lead time of {value} days.\nQuestion: {marker}'
            prompt = tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            prefix, end = prompt.split(marker)
            return tok.encode(prefix, add_special_tokens=False, return_tensors='pt'), tok.encode(question + end, add_special_tokens=False, return_tensors='pt')
        ids, artifacts, generations, origins = {}, {}, {}, {}
        for entity in protocol['entities']:
            for value in protocol['values']:
                key = f'{entity}_{value}'
                origin = reg.origin('fictional:prefix-probe', key, '1', {'entity': entity, 'value': value}, acl=['buyer'])
                generation = reg.publish('probe:' + key, {'entity':entity, 'value':value}, [origin], 'buyer', ['buyer'])
                prefix, _ = prompt_parts(entity, value, '')
                ids[key] = prefix
                artifacts[key] = capsules.compile(prefix, generation, key)
                generations[key], origins[key] = generation, origin
        result['base_sha256'] = base_hash
        result['capsule_tensor_bytes'] = {key: reg.node(art)['payload']['payload']['tensor_bytes'] for key, art in artifacts.items()}
        raw_logits = {}
        for entity in protocol['entities']:
            for vi, value in enumerate(protocol['values']):
                wrong = protocol['values'][(vi + 1) % len(protocol['values'])]
                key, wrong_key = f'{entity}_{value}', f'{entity}_{wrong}'
                for op in protocol['operators']:
                    question = {'lookup':f'What is the delivery lead time of {entity}?',
                                'plus_two':f'What is the delivery lead time of {entity} plus 2 days?',
                                'above_25':f'Is the delivery lead time of {entity} greater than 25 days?'}[op]
                    if a.typed_format:
                        question += '\nReply with only yes or no.' if op == 'above_25' else '\nReply with only the integer number of days.'
                    def expected(v):
                        return str(v) if op == 'lookup' else str(v + 2) if op == 'plus_two' else 'yes' if v > 25 else 'no'
                    _, suffix = prompt_parts(entity, value, question)
                    outputs = {}
                    for mode in protocol['modes']:
                        t = time.perf_counter()
                        snapshot = None
                        if mode == 'fresh_text_prefix':
                            cache = capsules.prefill(ids[key])
                            model_input = suffix
                        elif mode in ['saved_prefix', 'wrong_value_prefix']:
                            cache, snapshot = capsules.load(artifacts[key if mode == 'saved_prefix' else wrong_key])
                            model_input = suffix
                        else:
                            cache = None
                            model_input = tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':question}], tokenize=True,add_generation_prompt=True,return_tensors='pt')
                            if not isinstance(model_input, torch.Tensor):
                                model_input = model_input['input_ids']
                        generated = capsules.decode(model_input, cache, eos, protocol['max_tokens'])
                        text = tok.decode(generated['tokens'], skip_special_tokens=True).strip()
                        target = 'UNKNOWN' if mode == 'no_fact' else expected(wrong if mode == 'wrong_value_prefix' else value)
                        row = {'entity':entity,'value':value,'operator':op,'mode':mode,'question':question,
                               'expected':target,'text':text,'correct':text.lower() == target.lower() and generated['eos'],
                               'tokens':generated['tokens'],'eos':generated['eos'],'elapsed_s':time.perf_counter()-t,
                               'new_input_tokens':model_input.shape[1]}
                        if mode == 'saved_prefix':
                            row['receipt'] = reg.commit(snapshot, text)
                        index = str(len(result['rows']))
                        raw_logits[index] = generated['logits']
                        row['first_logits_sha256'] = hashlib.sha256(generated['logits'].numpy().tobytes()).hexdigest()
                        outputs[mode] = generated
                        result['rows'].append(row)
                    assert torch.equal(outputs['fresh_text_prefix']['logits'], outputs['saved_prefix']['logits']), 'Serialized cache logits differ'
                    assert outputs['fresh_text_prefix']['tokens'] == outputs['saved_prefix']['tokens'], 'Serialized cache continuation differs'
                    save()
                print(json.dumps({'entity':entity,'value':value,'completed_sequences':len(result['rows'])}), flush=True)
        save_file(raw_logits, a.output / 'first_logits.safetensors')
        # Update one identity and compile a new state without gradients.
        key = 'Neral_18'
        _, pending = capsules.load(artifacts[key])
        new_origin = reg.origin('fictional:prefix-probe', key, '2', {'entity':'Neral','value':24}, acl=['buyer'])
        new_generation = reg.publish('probe:' + key, {'entity':'Neral','value':24}, [new_origin], 'buyer', ['buyer'])
        updated = capsules.compile(ids['Neral_24'], new_generation, 'Neral_updated')
        def blocked(label, action):
            try: action()
            except InvalidState: result['lifecycle'][label] = True
            else: raise AssertionError(label + ' was not blocked')
        blocked('stale_load', lambda: capsules.load(artifacts[key]))
        blocked('stale_commit', lambda: reg.commit(pending, '18'))
        cache, snap = capsules.load(updated)
        update_question = 'What is the delivery lead time of Neral?'
        if a.typed_format: update_question += '\nReply with only the integer number of days.'
        _, suffix = prompt_parts('Neral', 24, update_question)
        answer = capsules.decode(suffix, cache, eos)
        result['update_answer'] = tok.decode(answer['tokens'], skip_special_tokens=True).strip()
        result['update_correct'] = result['update_answer'] == '24' and answer['eos']
        reg.commit(snap, result['update_answer'])
        reg.revoke(new_origin)
        blocked('revoked_load', lambda: capsules.load(updated))
        blocked('revoked_commit', lambda: reg.commit(snap, '24'))
        blocked('acl', lambda: capsules.load(artifacts['Vost_24'], 'outsider'))
        capsules.load(artifacts['Vost_24'])
        result['lifecycle']['unrelated_still_active'] = True
        result['base_unchanged'] = weights_hash(model) == base_hash
        assert result['base_unchanged']
        result['scores'] = {mode: {'correct':sum(r['correct'] for r in result['rows'] if r['mode']==mode),
                                 'total':sum(r['mode']==mode for r in result['rows']),
                                 'median_s':statistics.median(r['elapsed_s'] for r in result['rows'] if r['mode']==mode)} for mode in protocol['modes']}
        result['exact_serialization_cases'] = 24
        result['scientific_gate_passed'] = all(r['correct'] for r in result['rows'] if r['mode'] != 'no_fact') and result['update_correct']
        result['status'] = 'completed'
        print(json.dumps({'status':result['status'],'scores':result['scores'],'scientific_gate_passed':result['scientific_gate_passed']}))
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        save()
        reg.close()


if __name__ == '__main__':
    main()
