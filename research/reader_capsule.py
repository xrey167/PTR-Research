"""Reader-identity and provenance bound full-prefix capsules.

Research correctness path: recomputes the full reader identity at use boundaries.
This is expensive on large models; it makes no low-overhead performance claim.
Local trusted-process assumptions of PrefixCapsules and Registry still apply.
"""
import torch
from safetensors.torch import save_file
from neural_pods.registry import InvalidState, hash_files
from research.prefix_capsule import PrefixCapsules
from research.reader_identity import reader_identity
from research.reader_answer_guard import guarded_answer


class ReaderCapsules(PrefixCapsules):
    def __init__(self, registry, model, directory, reader_key, principal='buyer'):
        self.reader_key = reader_key
        self.principal = principal
        registry.snapshot([reader_key], principal)
        node = registry.node(reader_key)
        payload = node['payload'].get('payload', {})
        if node['kind'] != 'lora' or payload.get('schema') != 'research-reader:v1':
            raise InvalidState('Reader requires a registered model artifact')
        self.identity = reader_identity(model)
        if payload.get('reader_identity') != self.identity:
            raise InvalidState('Registered reader identity differs from actual model')
        super().__init__(registry, model, directory, self.identity['sha256'])

    def validate_reader(self, principal=None):
        self.registry.snapshot([self.reader_key], principal or self.principal)
        if reader_identity(self.model) != self.identity:
            raise InvalidState('Reader weights or effective configuration changed')

    @torch.inference_mode()
    def compile(self, ids, generation, name, principal='buyer'):
        self.validate_reader(principal)
        self.registry.snapshot([generation, self.reader_key], principal)
        if not name.isidentifier():
            raise ValueError('Use an identifier as capsule name')
        target = self.directory / name
        target.mkdir(exist_ok=False)
        cache = self.prefill(ids)
        self.validate_reader(principal)
        tensors = {}
        for i, (key, value, sliding) in enumerate(cache):
            if sliding is not None:
                raise ValueError('Sliding cache cannot be serialized by this experiment')
            tensors[f'k_{i}'], tensors[f'v_{i}'] = key.contiguous(), value.contiguous()
        save_file(tensors, target / 'state.safetensors')
        payload = {'schema': 'research-full-prefix:v1', 'directory': name,
                   'model_sha256': self.model_sha256, 'reader_key': self.reader_key,
                   'token_count': ids.shape[1], 'layers': len(cache.layers),
                   'files': hash_files(target),
                   'tensor_bytes': sum(t.numel() * t.element_size() for t in tensors.values())}
        return self.registry.artifact('cache', payload, [generation, self.reader_key], principal)

    def load(self, artifact, principal='buyer'):
        self.validate_reader(principal)
        node = self.registry.node(artifact)
        if (self.reader_key not in node['payload'].get('parent_artifact_keys', [])
                or node['payload'].get('payload', {}).get('reader_key') != self.reader_key):
            raise InvalidState('Capsule lacks this reader ancestry')
        return super().load(artifact, principal)

    def decode(self, suffix_ids, cache, eos_ids, max_tokens=12):
        self.validate_reader()
        result = super().decode(suffix_ids, cache, eos_ids, max_tokens)
        self.validate_reader()
        return result

    def guard_answer(self, row, raw_text):
        """Apply the typed Pod answer barrier after reader decoding."""
        self.validate_reader()
        return guarded_answer(row, raw_text)
