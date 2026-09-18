"""Prompt/masking interface for the proposed reader trainer; no model imports."""

SYSTEM = ('Use the supplied fact to answer the question. Follow the answer format '
          'requested in the question. If the fact is missing, answer UNKNOWN. '
          'For a deadline, convert every week to 7 days, add any extra days, '
          'and answer Yes only when the resulting deadline is at least the '
          'lead time; otherwise answer No. For a buffer question, add the '
          'requested buffer days to the lead time. Pods have explicit types: '
          'context Pods provide prose facts, math Pods require deterministic '
          'unit arithmetic, and model Pods describe an executable adapter. '
          'Never use a model Pod as if it were a fact.')
INSTRUCTION = ('\nAnswer the latest user question briefly. Use a number of days for a '
               'duration, and yes/no with a brief reason for a yes/no question.')
BOUNDARY = 'QUESTION_BOUNDARY_9'


def render_segments(tokenizer, row):
    evidence = row['evidence']
    if evidence is None:
        fact = f"Pod type: {row.get('pod_type', 'context')}. No fact is available."
    elif row.get('pod_type') == 'model':
        fact = (f"Pod type: model. Registered model adapter {evidence.get('model_name', 'unknown')} "
                f"with adapter identity {evidence.get('adapter_sha256', 'unknown')}.")
    else:
        fact = (f"Pod type: {row.get('pod_type', 'context')}. Supplier {evidence['supplier']} has a delivery lead time of "
                f"{evidence['lead_time_days']} days for component {evidence['component']}.")
    if BOUNDARY in fact:
        raise ValueError('Fact conflicts with the prompt boundary')
    template = tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': 'Fact: ' + fact + '\nQuestion: ' + BOUNDARY},
    ], tokenize=False, add_generation_prompt=True)
    prefix, tail = template.split(BOUNDARY)
    history = row['history']
    if any(m['role'] not in ('user', 'assistant') for m in history):
        raise ValueError('History supports user and assistant messages only')
    dialogue = '\n'.join(m['role'].capitalize() + ': ' + m['content'] for m in history)
    question = ('Earlier dialogue:\n' + dialogue + '\nLatest user question: ' if history else '') + row['question']
    return prefix, question + INSTRUCTION + tail


def encode_training_row(tokenizer, row, max_length):
    prefix, suffix = render_segments(tokenizer, row)
    prompt_ids = (tokenizer.encode(prefix, add_special_tokens=False)
                  + tokenizer.encode(suffix, add_special_tokens=False))
    if not prompt_ids or tokenizer.eos_token_id is None:
        raise ValueError('A nonempty prompt and EOS token are required')
    if not row['target'].strip():
        raise ValueError('Empty training answer')
    answer_ids = tokenizer.encode(row['target'], add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prompt_ids + answer_ids
    if len(ids) > max_length:
        raise ValueError('Training row exceeds length limit; truncation is not allowed')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prompt_ids) + answer_ids}
