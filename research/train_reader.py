"""Local BF16 reader LoRA training. Preflight and training are separate runs.

This trains the reader interface, not the planner, links or mutable Pod facts.
An adapter produced here still requires held-out and capsule integration tests.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.reader_prompt import encode_training_row
from research.reader_training import accumulation_windows, token_normalized_backward
from research.progress_json import write_progress


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(path):
    path = Path(path).resolve()
    protocol = json.loads((path / 'protocol.json').read_text(encoding='utf-8'))
    for folder, mapping in [('inputs', 'input_hashes'), ('source_snapshot', 'source_hashes')]:
        for name, expected in protocol[mapping].items():
            target = (path / folder / name).resolve()
            if not target.is_relative_to(path / folder) or file_sha(target) != expected:
                raise ValueError(f'Frozen input mismatch: {folder}/{name}')
    # These three implementations define the actual training inputs and loss.
    for name in ('reader_prompt.py', 'reader_training.py', 'reader_identity.py'):
        relative = 'research/' + name
        if file_sha(Path(__file__).parent / name) != protocol['source_hashes'][relative]:
            raise ValueError(f'Implementation changed since input freeze: {name}')
    config = json.loads((path / 'inputs/config.json').read_text(encoding='utf-8'))
    training = config['training']
    if training['world_size'] != 1 or training['micro_batch_size'] < 1:
        raise ValueError('Only single-process training is supported')
    if (training['scheduler'] != 'linear' or not training['assistant_only_loss']
            or config['base_precision'] != 'bfloat16'):
        raise ValueError('Unsupported training configuration')
    if training['effective_batch_size'] != training['micro_batch_size'] * training['gradient_accumulation_steps']:
        raise ValueError('Effective batch mismatch')
    rows = json.loads((path / 'inputs/train.json').read_text(encoding='utf-8'))
    steps = math.ceil(len(rows) / training['effective_batch_size']) * training['epochs']
    if steps != protocol['optimizer_updates'] or len(rows) != protocol['rows']['train']:
        raise ValueError('Frozen training schedule mismatch')
    return protocol, config, rows


def parameter_hash(model, trainable):
    import torch
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if tensor.requires_grad != trainable:
            continue
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast('B'))
    return digest.hexdigest()


def train_windows(model, encoded, training, report_step, preflight=False,
                  preflight_microbatches=1):
    """Actual optimizer path shared by the full run and memory probe."""
    import torch
    from transformers import get_linear_schedule_with_warmup
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('No trainable parameters')
    optimizer = torch.optim.AdamW(params, lr=training['learning_rate'],
                                  weight_decay=training['weight_decay'])
    micro_batch_size = training.get('micro_batch_size', 1)
    gradient_accumulation_steps = training['gradient_accumulation_steps']
    effective_batch_size = training.get('effective_batch_size', micro_batch_size * gradient_accumulation_steps)
    total = math.ceil(len(encoded) / effective_batch_size) * training['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=math.ceil(total * training['warmup_ratio']), num_training_steps=total)
    device = next(model.parameters()).device
    model.train()
    step = 0
    for epoch in range(training['epochs']):
        indices = list(range(len(encoded)))
        random.Random(training['seed'] + epoch).shuffle(indices)
        if preflight:
            if type(preflight_microbatches) is not int or preflight_microbatches < 1:
                raise ValueError('Preflight microbatches must be positive')
            # Exercise the largest sequence through one bounded optimizer step,
            # allocating gradients and Adam state without committing to a full
            # 16-microbatch CPU experiment.
            longest = max(indices, key=lambda i: len(encoded[i]['input_ids']))
            indices = [longest] * preflight_microbatches
        window_size = micro_batch_size * gradient_accumulation_steps
        for window in accumulation_windows(indices, window_size):
            batches = []
            for start in range(0, len(window), micro_batch_size):
                group = window[start:start + micro_batch_size]
                max_len = max(len(encoded[i]['input_ids']) for i in group)
                batch = {}
                for key in encoded[group[0]]:
                    values = [encoded[i][key] + [0] * (max_len - len(encoded[i][key])) for i in group]
                    batch[key] = torch.tensor(values, dtype=torch.long, device=device)
                batches.append(batch)
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            metrics = token_normalized_backward(model, batches)
            norm = torch.nn.utils.clip_grad_norm_(params, training['max_grad_norm'], error_if_nonfinite=True)
            lr_used = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            step += 1
            report_step(dict(metrics, step=step, epoch=epoch + 1,
                             learning_rate=lr_used, gradient_norm=float(norm),
                             elapsed_s=time.perf_counter() - started))
        if preflight:
            break
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'train'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--init-adapter', type=Path)
    parser.add_argument('--probe-microbatches', type=int, default=1)
    args = parser.parse_args()
    protocol, config, rows = load_bundle(args.inputs)
    protocol_sha = file_sha(args.inputs / 'protocol.json')
    runner_sha = file_sha(__file__)
    if args.mode == 'train':
        if args.preflight is None:
            parser.error('Training requires --preflight PATH to a completed probe directory')
        probe = json.loads((args.preflight / 'report.json').read_text(encoding='utf-8'))
        if (probe['status'] != 'preflight_completed' or probe['protocol_sha256'] != protocol_sha
                or probe['device'] != args.device or probe['runner_sha256'] != runner_sha):
            raise ValueError('Preflight does not match this training run')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / 'train_reader.py')
    shutil.copyfile(args.inputs / 'protocol.json', output / 'input-protocol.json')
    state = {'status': 'running', 'phase': 'verify_model', 'mode': args.mode,
             'pid': os.getpid(), 'device': args.device, 'protocol_sha256': protocol_sha,
             'runner_sha256': runner_sha, 'steps': [], 'full_research_goal_complete': False,
             'held_out_evaluation_completed': False, 'capsule_integration_completed': False}
    started = time.perf_counter()
    monitor_stop = threading.Event()
    monitor_thread = None
    def save():
        state['elapsed_s'] = time.perf_counter() - started
        write_progress(output / 'report.json', state)
    save()
    try:
        model_path = Path(protocol['model_path'])
        manifest = json.loads((args.inputs / 'inputs/model-manifest.json').read_text(encoding='utf-8'))
        for name, metadata in manifest['files'].items():
            target = (model_path / name).resolve()
            if not target.is_relative_to(model_path.resolve()) or file_sha(target) != metadata['sha256']:
                raise ValueError(f'Base model file mismatch: {name}')
        state['model_files_verified'] = True
        state['phase'] = 'load'
        save()
        import psutil
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from research.reader_identity import reader_identity
        torch.set_num_threads(4)
        set_seed(config['training']['seed'])
        if args.device == 'cuda' and not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 unsupported on selected GPU')
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        encoded = [encode_training_row(tokenizer, row, protocol['max_sequence_tokens']) for row in rows]
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                    attn_implementation='eager', local_files_only=True).to(args.device)
        model = get_peft_model(model, LoraConfig(**config['lora']))
        if args.init_adapter is not None:
            from safetensors.torch import load_file
            init_state = load_file(str(args.init_adapter / 'adapter_model.safetensors'), device='cpu')
            # PEFT 0.20 names adapter parameters with an explicit `.default` slot;
            # older checkpoints omit that slot. Normalize both formats so a prior
            # Pod adapter can initialize the next generation without changing the
            # model or adapter identity.
            normalized_state = {}
            for key, value in init_state.items():
                if '.lora_A.weight' in key:
                    key = key.replace('.lora_A.weight', '.lora_A.default.weight')
                elif '.lora_B.weight' in key:
                    key = key.replace('.lora_B.weight', '.lora_B.default.weight')
                normalized_state[key] = value
            missing, unexpected = model.load_state_dict(normalized_state, strict=False)
            adapter_missing = [key for key in missing if '.lora_' in key]
            if unexpected or adapter_missing:
                raise ValueError(f'Adapter state mismatch: unexpected={unexpected[:3]}, missing={adapter_missing[:3]}')
            state['initialized_from_adapter'] = str(args.init_adapter)
            state['initialized_adapter_keys'] = len(init_state)
        if any(p.requires_grad and 'lora_' not in name for name, p in model.named_parameters()):
            raise ValueError('Unexpected non-LoRA trainable weights')
        if config['training']['gradient_checkpointing']:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        state['frozen_parameters_before'] = parameter_hash(model, False)
        state['adapter_parameters_before'] = parameter_hash(model, True)
        state['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state['maximum_training_tokens'] = max(len(r['input_ids']) for r in encoded)
        state['phase'] = args.mode
        save()
        process = psutil.Process()
        def monitor():
            while not monitor_stop.wait(30):
                state['elapsed_s'] = time.perf_counter() - started
                state['monitor_rss_bytes'] = process.memory_info().rss
                state['monitor_available_system_bytes'] = psutil.virtual_memory().available
                try:
                    save()
                except PermissionError:
                    state['progress_write_failures'] = state.get('progress_write_failures', 0) + 1
        monitor_thread = threading.Thread(target=monitor, name='reader-progress', daemon=True)
        monitor_thread.start()
        last_heartbeat = time.perf_counter()
        state['forward_calls'] = 0
        state['progress_write_failures'] = 0
        def heartbeat(_module, _inputs, _output):
            nonlocal last_heartbeat
            state['forward_calls'] += 1
            if time.perf_counter() - last_heartbeat >= 30:
                state['current_rss_bytes'] = process.memory_info().rss
                try:
                    save()
                except PermissionError:
                    state['progress_write_failures'] += 1
                last_heartbeat = time.perf_counter()
        hook = model.register_forward_hook(heartbeat)
        def report_step(metrics):
            memory = process.memory_info()
            metrics['rss_bytes'] = memory.rss
            metrics['peak_working_set_bytes'] = getattr(memory, 'peak_wset', None)
            metrics['private_bytes'] = getattr(memory, 'private', None)
            metrics['available_system_bytes'] = psutil.virtual_memory().available
            if args.device == 'cuda':
                metrics['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            state['steps'].append(metrics)
            save()
            print(json.dumps(metrics), flush=True)
        try:
            train_windows(model, encoded, config['training'], report_step, args.mode == 'preflight',
                          args.probe_microbatches)
        finally:
            hook.remove()
        state['phase'] = 'verify_weights'
        save()
        state['frozen_parameters_after'] = parameter_hash(model, False)
        state['adapter_parameters_after'] = parameter_hash(model, True)
        if state['frozen_parameters_before'] != state['frozen_parameters_after']:
            raise AssertionError('Frozen base parameters changed')
        if args.mode == 'train' and state['adapter_parameters_before'] == state['adapter_parameters_after']:
            raise AssertionError('Adapter weights did not change')
        if args.mode == 'train':
            if len(state['steps']) != protocol['optimizer_updates']:
                raise AssertionError('Training did not complete the frozen schedule')
            # Inference configuration participates in the capsule reader identity.
            model.config.use_cache = True
            model.gradient_checkpointing_disable()
            for adapter_config in model.peft_config.values():
                adapter_config.inference_mode = True
            model.save_pretrained(output / 'adapter', safe_serialization=True)
            tokenizer.save_pretrained(output / 'tokenizer')
            state['reader_identity'] = reader_identity(model)
            state['adapter_files'] = {p.name: file_sha(p) for p in (output / 'adapter').iterdir() if p.is_file()}
            state['status'] = 'trained_not_evaluated'
        else:
            state['status'] = 'preflight_completed'
            state['probe_adapter_discarded'] = True
            state['probe_microbatches'] = args.probe_microbatches
            state['limitations'] = 'One bounded optimizer step on the longest training row by default; available memory and paging can change. No quality or runtime lineage claim.'
        state['phase'] = 'done'
        save()
    except BaseException as error:
        state['status'] = 'failed'
        state['error'] = repr(error)
        save()
        raise
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2)


if __name__ == '__main__':
    main()
