"""DPO fine-tuning of the reader LoRA on verifier-judged preference pairs.

Policy starts from a frozen Gen-4 adapter; the reference model is that same
adapter without gradients. Loss: -logsigmoid(beta * ((pol_c - ref_c) - (pol_r
- ref_r))) over the completion tokens (prompt-masked). A short preflight run
bounds CUDA memory before the full run, mirroring the SFT pipeline contract.

Run base and adapter evaluations separately afterwards (evaluate_reader.py);
this script does not evaluate.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.train_reader import file_sha
from research.progress_json import write_progress


def seq_logprobs(model, input_ids, attention_mask, labels):
    """Sum log p(labels) for masked positions; returns per-sequence sums."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    gathered = torch.gather(logprobs, 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    # Per-token mean (length-normalized): summing log-probs inflates the DPO
    # margin for long completions and collapses the policy.
    return (gathered * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--adapter-run', type=Path, required=True,
                        help='frozen adapter the policy and reference start from')
    parser.add_argument('--inputs', type=Path, required=True,
                        help='inputs dir matching the adapter protocol (identity checks)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--mode', choices=['preflight', 'train'], required=True)
    parser.add_argument('--max-pairs', type=int, default=0, help='0 = all (train mode)')
    args = parser.parse_args()

    pairs = json.loads((args.pairs / 'pairs.json').read_text(encoding='utf-8'))
    adapter_report = json.loads((args.adapter_run / 'report.json').read_text(encoding='utf-8'))
    if adapter_report['protocol_sha256'] != file_sha(args.inputs / 'protocol.json'):
        raise ValueError('Adapter run does not match the inputs protocol')
    for name, expected in adapter_report['adapter_files'].items():
        if file_sha(args.adapter_run / 'adapter' / name) != expected:
            raise ValueError('Adapter file mismatch')

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in ('train_reader_dpo.py', 'train_reader.py', 'reader_prompt.py',
                 'progress_json.py'):
        shutil.copyfile(Path(__file__).parent / name, output / name)
        sources[name] = file_sha(output / name)

    state = {'status': 'running', 'phase': 'load', 'mode': args.mode,
             'protocol_sha256': adapter_report['protocol_sha256'],
             'source_adapter_report_sha256': file_sha(args.adapter_run / 'report.json'),
             'pairs_sha256': file_sha(args.pairs / 'pairs.json'),
             'pairs': len(pairs), 'beta': args.beta, 'epochs': args.epochs,
             'learning_rate': args.learning_rate, 'sources': sources,
             'scope': 'DPO on verifier-judged pairs; evaluation happens separately'}
    started = time.perf_counter()

    def save():
        state['elapsed_s'] = time.perf_counter() - started
        write_progress(output / 'report.json', state)

    save()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.set_num_threads(4)
    protocol = json.loads((args.inputs / 'protocol.json').read_text(encoding='utf-8'))
    model_path = Path(protocol['model_path']).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                dtype=torch.bfloat16, attn_implementation='eager')
    policy = PeftModel.from_pretrained(base, args.adapter_run / 'adapter', is_trainable=True)
    policy.to(args.device)
    policy.train()
    save()
    # Reference: second instance on the same device, no gradients.
    ref_base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                    dtype=torch.bfloat16, attn_implementation='eager')
    reference = PeftModel.from_pretrained(ref_base, args.adapter_run / 'adapter')
    reference.to(args.device)
    reference.eval().requires_grad_(False)

    def encode(pair):
        prompt = pair['prompt_ids']
        def ids(answer):
            return prompt + tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        c, r = ids(pair['chosen']), ids(pair['rejected'])
        def tensor(seq):
            labels = [-100] * len(prompt) + seq[len(prompt):]
            return (torch.tensor([seq], device=args.device),
                    torch.tensor([labels], device=args.device))
        return (*tensor(c), *tensor(r))

    rows = pairs if not args.max_pairs else pairs[:args.max_pairs]
    state['phase'] = 'dpo' if args.mode == 'train' else 'preflight'
    state['rows'] = []
    save()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                  lr=args.learning_rate)
    step = 0
    for epoch in range(args.epochs):
        for pair in rows:
            step += 1
            c_ids, c_labels, r_ids, r_labels = encode(pair)
            mask = torch.ones_like(c_ids)
            pol_c = seq_logprobs(policy, c_ids, mask, c_labels)
            pol_r = seq_logprobs(policy, r_ids, mask, r_labels)
            with torch.no_grad():
                ref_c = seq_logprobs(reference, c_ids, mask, c_labels)
                ref_r = seq_logprobs(reference, r_ids, mask, r_labels)
            loss = -torch.nn.functional.logsigmoid(
                args.beta * ((pol_c - ref_c) - (pol_r - ref_r))).mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            if step % 5 == 0 or args.mode == 'preflight':
                state['rows'].append({'step': step, 'loss': float(loss.detach()),
                                      'margin': float((pol_c - ref_c - pol_r + ref_r).detach()),
                                      'grad_norm': float(grad_norm)})
                save()
            if args.mode == 'preflight':
                break
        if args.mode == 'preflight':
            break

    state['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if args.device == 'cuda' else None
    if args.mode == 'preflight':
        state.update(status='preflight_completed', phase='done',
                     note='No quality or runtime lineage claim.')
        save()
        return

    state['phase'] = 'save'
    save()
    save()
    adapter_dir = output / 'adapter'
    policy.save_pretrained(adapter_dir)
    # Identity of the saved artifact via a fresh default reload, computed
    # exactly like evaluate_reader.py so its check matches deterministically.
    del reference
    torch.cuda.empty_cache()
    from research.reader_identity import reader_identity
    check_base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                      dtype=torch.bfloat16, attn_implementation='eager')
    reloaded = PeftModel.from_pretrained(check_base, adapter_dir)
    reloaded.to(args.device)
    reloaded.eval()
    state['reader_identity'] = reader_identity(reloaded)
    save()
    adapter_files = {p.name: file_sha(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()}
    state.update(status='trained_not_evaluated', phase='done',
                 optimizer_updates=step, adapter_files=adapter_files,
                 adapter_dir=str(adapter_dir))
    save()


if __name__ == '__main__':
    main()
