"""Experimental full-prefix KV capsules; reconstructed locally, not original CQP1.

Trusted local, CPU, full-attention decoder only. Each call gets a fresh cache.
"""
from pathlib import Path
import hashlib

import torch
from safetensors.torch import load_file, save_file
from transformers import DynamicCache
from neural_pods.registry import InvalidState, hash_files, verify_files


def weights_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.named_parameters():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class PrefixCapsules:
    def __init__(self, registry, model, directory, model_sha256):
        self.registry, self.model = registry, model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model_sha256 = model_sha256
        if getattr(model.config, 'use_sliding_window', False):
            raise ValueError('This experiment supports full attention only')

    @torch.inference_mode()
    def prefill(self, ids):
        return self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          past_key_values=DynamicCache(config=self.model.config), use_cache=True).past_key_values

    def compile(self, ids, generation, name, principal='buyer'):
        self.registry.snapshot([generation], principal)
        if not name.isidentifier():
            raise ValueError('Use an identifier as capsule name')
        target = self.directory / name
        target.mkdir(exist_ok=False)
        cache = self.prefill(ids)
        tensors = {}
        for i, (k, v, sliding) in enumerate(cache):
            if sliding is not None:
                raise ValueError('Sliding cache cannot be serialized by this experiment')
            tensors[f'k_{i}'], tensors[f'v_{i}'] = k.contiguous(), v.contiguous()
        save_file(tensors, target / 'state.safetensors')
        payload = {'schema': 'research-full-prefix:v1', 'directory': name,
                   'model_sha256': self.model_sha256, 'token_count': ids.shape[1],
                   'layers': len(cache.layers), 'files': hash_files(target),
                   'tensor_bytes': sum(t.numel() * t.element_size() for t in tensors.values())}
        return self.registry.artifact('cache', payload, [generation], principal)

    def load(self, artifact, principal='buyer'):
        snapshot = self.registry.snapshot([artifact], principal)
        node = self.registry.node(artifact)
        p = node['payload']['payload']
        if node['kind'] != 'cache' or p.get('schema') != 'research-full-prefix:v1':
            raise InvalidState('Not a prefix capsule')
        if p['model_sha256'] != self.model_sha256:
            raise InvalidState('Capsule model differs')
        target = (self.directory / p['directory']).resolve()
        if target.parent != self.directory.resolve():
            raise InvalidState('Invalid capsule path')
        verify_files(target, p['files'])
        tensors = load_file(target / 'state.safetensors')
        cache = DynamicCache([(tensors[f'k_{i}'], tensors[f'v_{i}']) for i in range(p['layers'])], config=self.model.config)
        if cache.get_seq_length() != p['token_count']:
            raise InvalidState('Capsule length differs')
        return cache, snapshot

    @torch.inference_mode()
    def decode(self, suffix_ids, cache, eos_ids, max_tokens=12):
        """Return every generated token including EOS and full first-token logits."""
        prefix_length = cache.get_seq_length() if cache is not None else 0
        out = self.model(input_ids=suffix_ids,
                         attention_mask=torch.ones((1, prefix_length + suffix_ids.shape[1]), dtype=torch.long),
                         past_key_values=cache, use_cache=True)
        # BF16 values widen exactly to FP32; preserve all values for portable audits.
        first = out.logits[:, -1, :].detach().to(torch.float32).clone()
        if not torch.isfinite(first).all():
            raise ArithmeticError('Nonfinite logits')
        tokens = []
        for _ in range(max_tokens):
            token = int(out.logits[0, -1].argmax())
            tokens.append(token)
            if token in eos_ids:
                break
            length = out.past_key_values.get_seq_length() + 1
            out = self.model(input_ids=torch.tensor([[token]]),
                             attention_mask=torch.ones((1, length), dtype=torch.long),
                             past_key_values=out.past_key_values, use_cache=True)
        return {'tokens': tokens, 'eos': bool(tokens and tokens[-1] in eos_ids), 'logits': first}
