"""Training primitives for the proposed reader LoRA, single process only.

No model is loaded by importing this module. This is not yet a full trainer.
"""
from itertools import islice


def accumulation_windows(items, size):
    if type(size) is not int or size < 1:
        raise ValueError('Accumulation size must be a positive integer')
    iterator = iter(items)
    while window := list(islice(iterator, size)):
        yield window


def token_normalized_backward(model, batches):
    """Accumulate one window's gradients, normalized by supervised next tokens.

    Caller zeroes gradients before this function, then clips and steps once.
    Each batch has input_ids, labels and optionally attention_mask. Labels use
    -100 for prompt/padding. No DDP, loss scaler or optimizer policy is implied.
    """
    import torch
    import torch.nn.functional as functional
    batches = list(batches)
    counts = [int(batch['labels'][..., 1:].ne(-100).sum()) for batch in batches]
    total = sum(counts)
    if total == 0:
        raise ValueError('Window has no supervised next tokens')
    loss_sum = 0.0
    for batch, count in zip(batches, counts):
        if count == 0:
            continue
        output = model(input_ids=batch['input_ids'],
                       attention_mask=batch.get('attention_mask'), use_cache=False)
        logits = output.logits[..., :-1, :]
        # Match standard causal-LM loss computation in at least FP32.
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        summed = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         batch['labels'][..., 1:].reshape(-1),
                                         ignore_index=-100, reduction='sum')
        if not torch.isfinite(summed):
            raise ArithmeticError('Nonfinite reader training loss')
        (summed / total).backward()
        loss_sum += float(summed.detach())
        # Do not retain the previous vocabulary-sized logits during the next forward.
        del output, logits, summed
    return {'loss': loss_sum / total, 'supervised_tokens': total,
            'microbatches': len(batches)}
