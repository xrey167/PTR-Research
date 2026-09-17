"""Optional persistent vector index using NumPy only.

This is a fast, deterministic dense retrieval tier for environments without
FAISS/HNSW.  It keeps vectors in a compact matrix on disk and uses one
vectorized cosine pass.  An ANN provider can implement the same interface later
without changing Pod or cache contracts.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


class PersistentVectorIndex:
    def __init__(self, path: str | Path, *, dimension: int | None = None):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension; self.ids: list[str] = []; self.matrix = np.empty((0, 0), dtype=np.float32)
        if self.path.with_suffix(".npz").exists() and self.path.with_suffix(".json").exists(): self.load()

    def upsert(self, key: str, vector) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or not arr.size or not np.isfinite(arr).all(): raise ValueError("invalid vector")
        if self.dimension is None: self.dimension = int(arr.size)
        if arr.size != self.dimension: raise ValueError("vector dimension mismatch")
        if key in self.ids: self.matrix[self.ids.index(key)] = arr
        else: self.ids.append(str(key)); self.matrix = np.vstack([self.matrix, arr]) if self.matrix.size else arr.reshape(1, -1)

    def delete(self, key: str) -> None:
        if key not in self.ids: return
        i = self.ids.index(key); self.ids.pop(i); self.matrix = np.delete(self.matrix, i, axis=0)

    def search(self, vector, top_k: int = 10):
        if top_k <= 0 or not self.ids: return []
        q = np.asarray(vector, dtype=np.float32)
        if q.ndim != 1 or q.size != self.dimension: raise ValueError("vector dimension mismatch")
        qn = np.linalg.norm(q); norms = np.linalg.norm(self.matrix, axis=1)
        scores = (self.matrix @ q) / np.maximum(norms * qn, 1e-12)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in order]

    def save(self) -> None:
        np.savez_compressed(self.path.with_suffix(".npz"), matrix=self.matrix)
        self.path.with_suffix(".json").write_text(json.dumps({"ids": self.ids, "dimension": self.dimension}), encoding="utf-8")

    def load(self) -> None:
        meta = json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8")); data = np.load(self.path.with_suffix(".npz"))
        self.ids, self.dimension, self.matrix = list(meta["ids"]), meta["dimension"], data["matrix"].astype(np.float32)

    def stats(self):
        return {"vectors": len(self.ids), "dimension": self.dimension, "storage": "numpy-npz", "exact": True}


class HNSWVectorIndex(PersistentVectorIndex):
    """Optional HNSW acceleration with the same persistent interface.

    HNSW is used only when ``hnswlib`` is installed.  The exact NumPy parent
    remains the correctness fallback and is useful for recall evaluation.
    """
    def __init__(self, path: str | Path, *, dimension: int | None = None,
                 ef_search: int = 64, M: int = 16):
        try:
            import hnswlib
        except ImportError as exc:
            raise RuntimeError("HNSWVectorIndex requires hnswlib") from exc
        self._hnswlib = hnswlib; self.ef_search, self.M = int(ef_search), int(M)
        super().__init__(path, dimension=dimension)
        self._index = None
        saved = self.path.with_suffix(".hnsw")
        if self.ids:
            self._index = self._hnswlib.Index(space="cosine", dim=int(self.dimension))
            if saved.exists():
                self._index.load_index(str(saved), max_elements=max(1, len(self.ids)))
                self._index.set_ef(max(self.ef_search, 10))
            else:
                self._rebuild()

    def _rebuild(self):
        self._index = self._hnswlib.Index(space="cosine", dim=int(self.dimension))
        self._index.init_index(max_elements=max(1, len(self.ids)), M=self.M, ef_construction=200)
        if self.ids: self._index.add_items(self.matrix, np.arange(len(self.ids)))
        self._index.set_ef(max(self.ef_search, 10))

    def upsert(self, key: str, vector) -> None:
        super().upsert(key, vector); self._rebuild()

    def bulk_upsert(self, items) -> None:
        for key, vector in items:
            PersistentVectorIndex.upsert(self, key, vector)
        self._rebuild()

    def delete(self, key: str) -> None:
        super().delete(key); self._rebuild()

    def save(self) -> None:
        super().save()
        if self._index is None: self._rebuild()
        self._index.save_index(str(self.path.with_suffix(".hnsw")))

    def search(self, vector, top_k: int = 10):
        if not self.ids or top_k <= 0: return []
        q = np.asarray(vector, dtype=np.float32)
        labels, distances = self._index.knn_query(q, k=min(top_k, len(self.ids)))
        return [(self.ids[int(i)], float(1.0 - d)) for i, d in zip(labels[0], distances[0])]

    def stats(self):
        return {"vectors": len(self.ids), "dimension": self.dimension, "storage": "hnswlib", "exact": False,
                "ef_search": self.ef_search, "M": self.M}
