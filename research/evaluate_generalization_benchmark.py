"""Evaluate lexical retrieval and a deterministic provenance upper bound."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from research.generate_generalization_benchmark import build
from sentence_transformers import SentenceTransformer

def run():
    data = build(); docs, queries = data["corpus"], data["queries"]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    mat = vec.fit_transform([d["text"] for d in docs])
    lexical, dense, oracle = [], [], []
    encoder = SentenceTransformer("models/encoder", device="cpu")
    dense_matrix = encoder.encode([d["text"] for d in docs], normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
    for q in queries:
        scores = cosine_similarity(vec.transform([q["question"]]), mat)[0]
        order = np.argsort(-scores)
        got = {docs[i]["doc_id"] for i in order[:9]}
        lexical.append(len(got.intersection(q["target_docs"])) / 3.0)
        qv = encoder.encode([q["question"]], normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)[0]
        dorder = np.argsort(-(dense_matrix @ qv))
        dgot = {docs[i]["doc_id"] for i in dorder[:9]}
        dense.append(len(dgot.intersection(q["target_docs"])) / 3.0)
        # This models resolved hard metadata, not learned language routing.
        routed = {d["doc_id"] for d in docs if d["origin_key"] == q["origin_key"]}
        oracle.append(len(routed.intersection(q["target_docs"])) / 3.0)
    return {"schema": "generalization-benchmark-eval:v1", "queries": len(queries),
            "required_docs_per_query": 3, "k": 9,
            "lexical_recall_at_9": float(np.mean(lexical)),
            "lexical_full_chain_rate": float(np.mean(np.array(lexical) == 1.0)),
            "dense_recall_at_9": float(np.mean(dense)),
            "dense_full_chain_rate": float(np.mean(np.array(dense) == 1.0)),
            "provenance_routed_recall": float(np.mean(oracle)),
            "scope": data["scope"],
            "note": "No LLM quality claim; retrieval-only screen with held-out roots/templates."}

if __name__ == "__main__":
    report = run(); p = Path("runs/generalization-benchmark-eval-001.json")
    p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
