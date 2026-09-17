"""Small local rank-profile primitives inspired by Vespa's phased ranking.

The profile is deliberately dependency-free.  A future ONNX/Tensor backend can
implement the same callable contract without moving lifecycle decisions into
the search index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .local_search import SearchHit


TensorScorer = Callable[[tuple[float, ...], Mapping[str, Any]], float]


@dataclass(frozen=True)
class TensorRankProfile:
    """Bounded second-phase scorer over search features.

    ``tensor_scorer`` may wrap an ONNX/Tensor model.  It receives the dense
    feature vector and hard-filtered metadata, but it cannot bypass ACL,
    revocation or generation checks performed before ranking.
    """

    vector_weight: float = 0.65
    lexical_weight: float = 0.35
    regex_weight: float = 0.15
    metadata_weights: Mapping[str, float] = field(default_factory=dict)
    tensor_scorer: TensorScorer | None = None

    def score(self, hit: SearchHit) -> float:
        base = (self.vector_weight * hit.vector_score
                + self.lexical_weight * hit.lexical_score
                + self.regex_weight * hit.regex_score)
        for key, weight in self.metadata_weights.items():
            try:
                base += float(weight) * float(hit.metadata.get(key, 0.0))
            except (TypeError, ValueError):
                continue
        if self.tensor_scorer is not None:
            features = (hit.vector_score, hit.lexical_score, hit.regex_score)
            base += float(self.tensor_scorer(features, hit.metadata))
        return base


def onnx_ranker(session, input_name: str = "features", output_index: int = 0) -> TensorScorer:
    """Adapt an onnxruntime session to ``TensorRankProfile`` lazily.

    Importing onnxruntime is deferred so the local backend remains usable on
    CPU-only installations.  The caller owns the model/session lifecycle.
    """
    def score(features: tuple[float, ...], metadata: Mapping[str, Any]) -> float:
        import numpy as np
        values = np.asarray([features], dtype=np.float32)
        output = session.run(None, {input_name: values})[output_index]
        return float(np.asarray(output).reshape(-1)[0])
    return score
