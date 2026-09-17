"""Apply generation-bound lifecycle writes to an actual Qwen KV cache."""
from __future__ import annotations

from copy import deepcopy
import torch

from neural_pods.lifecycle_transport import TransportedLifecycleToken, apply_token, validate_registry_binding


class QwenCacheWriteAdapter:
    """Feature-space write adapter for Qwen DynamicCache tensors.

    This is a real decoder-cache operation: every key/value head receives the
    same orthogonal feature transform. It is intentionally separate from the
    model's learned weights and remains generation-bound by the caller.
    """

    def __init__(self, model):
        self.model = model

    @staticmethod
    def clone(cache):
        return deepcopy(cache)

    @staticmethod
    def cast(cache, dtype):
        """Copy a DynamicCache to a storage dtype for mixed-precision policy."""
        result = deepcopy(cache)
        for layer in result.layers:
            if layer.keys is not None:
                layer.keys = layer.keys.to(dtype)
                layer.values = layer.values.to(dtype)
        return result

    @classmethod
    def fp32_master(cls, cache):
        """Keep lifecycle writes in FP32 and materialize only at model boundaries.

        Repeated BF16 writes round every intermediate state.  A FP32 master
        cache avoids that accumulation while still allowing the decoder to
        consume a BF16 materialization.
        """
        return cls.cast(cache, torch.float32)

    def write(self, cache, matrix: torch.Tensor):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("write must be a square matrix")
        for layer in cache.layers:
            if layer.keys is None:
                continue
            if layer.keys.shape[-1] != matrix.shape[0]:
                raise ValueError("write dimension differs from Qwen head dimension")
            layer.keys = layer.keys @ matrix.to(layer.keys)
            layer.values = layer.values @ matrix.to(layer.values)
        return cache

    def transported_delete(self, cache, token: TransportedLifecycleToken, *,
                           registry=None, principal="buyer"):
        if registry is not None:
            validate_registry_binding(token, registry, principal=principal)
        return self.write(cache, apply_token(torch.eye(token.matrix.shape[0], dtype=token.matrix.dtype), token,
                                             identity_key=token.identity_key,
                                             generation_key=token.generation_key,
                                             snapshot_key=token.snapshot_key))

    @staticmethod
    def max_cache_error(left, right):
        errors = []
        for a, b in zip(left.layers, right.layers):
            if a.keys is not None:
                errors.extend([(a.keys - b.keys).abs().max(), (a.values - b.values).abs().max()])
        return max(float(error) for error in errors) if errors else 0.0

    @staticmethod
    def relative_cache_error(left, right):
        numerator, denominator = [], []
        for a, b in zip(left.layers, right.layers):
            if a.keys is not None:
                numerator.extend([(a.keys - b.keys).abs().max(), (a.values - b.values).abs().max()])
                denominator.extend([b.keys.abs().max(), b.values.abs().max()])
        if not numerator:
            return 0.0
        ratios = [float(n / d) for n, d in zip(numerator, denominator) if float(d)]
        return max(ratios) if ratios else 0.0
