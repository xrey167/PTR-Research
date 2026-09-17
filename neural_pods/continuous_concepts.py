"""Lifecycle-safe continuous concept injection for Pod-aware decoders.

This is the small model-side primitive suggested by CoCoMix: a concept vector
is predicted/loaded separately and mixed into token hidden states.  Retrieval,
ACL and generation validation remain outside this module; callers must pass
only concepts that have already crossed those barriers.
"""
from __future__ import annotations

import torch
from torch import nn


class ContinuousConceptMixer(nn.Module):
    """Inject a sequence of concept vectors without changing token length."""

    def __init__(self, hidden_size: int, concept_size: int | None = None,
                 initial_gate: float = 0.0):
        super().__init__()
        concept_size = concept_size or hidden_size
        self.project = nn.Identity() if concept_size == hidden_size else nn.Linear(concept_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))

    def forward(self, hidden: torch.Tensor, concepts: torch.Tensor,
                *, concept_mask: torch.Tensor | None = None) -> torch.Tensor:
        if hidden.ndim != 3 or concepts.ndim != 3:
            raise ValueError("hidden and concepts must be [batch, sequence, width]")
        if hidden.shape[0] != concepts.shape[0]:
            raise ValueError("hidden/concepts batch dimensions differ")
        injected = self.project(concepts.to(dtype=hidden.dtype, device=hidden.device))
        if injected.shape[1] != hidden.shape[1]:
            # Align a variable number of Pod concepts to token positions by
            # deterministic nearest-neighbour expansion.
            index = torch.linspace(0, injected.shape[1] - 1, hidden.shape[1], device=hidden.device).round().long()
            injected = injected.index_select(1, index)
        if concept_mask is not None:
            if concept_mask.shape != injected.shape[:2]:
                raise ValueError("concept_mask must be [batch, sequence]")
            injected = injected * concept_mask.to(injected.dtype).unsqueeze(-1)
        return hidden + torch.sigmoid(self.gate).to(hidden.dtype) * injected


class DecoderConceptInjector:
    """Attach a mixer to one decoder block without modifying model weights."""

    def __init__(self, model: nn.Module, mixer: ContinuousConceptMixer,
                 layer_index: int = 0):
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or not 0 <= layer_index < len(layers):
            raise ValueError("model does not expose the requested decoder layer")
        self.mixer, self.layer = mixer, layers[layer_index]
        self._handle = None
        self._concepts = None
        self._mask = None

    def set_concepts(self, concepts: torch.Tensor | None,
                     concept_mask: torch.Tensor | None = None):
        self._concepts, self._mask = concepts, concept_mask

    def _hook(self, _module, _inputs, output):
        if self._concepts is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        mixed = self.mixer(hidden, self._concepts, concept_mask=self._mask)
        return (mixed, *output[1:]) if isinstance(output, tuple) else mixed

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
