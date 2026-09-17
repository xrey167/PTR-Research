"""Re-embed canonical Pods locally and measure native-style retrieval."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.local_search import LocalSearchBackend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "native-embedding-reindex-001"

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    encoder_path = ROOT / "models" / "encoder"
    model = SentenceTransformer(str(encoder_path), device="cpu", local_files_only=True)
    embed = lambda text: model.encode(str(text), normalize_embeddings=True).tolist()
    backend = LocalSearchBackend(OUT / "namespace.sqlite", embedder=embed)
    backend.create_namespace("canonical-pods")
    dimension = model.get_embedding_dimension()
    backend.configure_schema("canonical-pods", {"text": {"type": "string", "full_text_search": True,
        "embed": {"model": "local/models/encoder", "dims": dimension}},
        "pod_type": {"type": "string", "filterable": True}, "generation_key": {"type": "string", "filterable": True}})
    docs = [
        ("lead-24", "Current delivery lead time for supplier Müller GmbH X12 is 24 days.", "fact", "g8"),
        ("lead-18", "Superseded delivery lead time for Müller X12 is 18 days.", "fact", "g7"),
        ("risk", "Müller GmbH supplier risk is high due to repeated procurement delays.", "context", "g8"),
        ("rule", "Orders above 25000 EUR require approval before release.", "rule", "g3"),
        ("math", "For X12, add four buffer days to the 56 day estimate: 60 days.", "math", "g2"),
        ("event", "Purchase order X12 was delayed by 8 days in April.", "event", "g4"),
    ]
    for key, text, kind, generation in docs:
        backend.upsert("canonical-pods", key, text=text, metadata={"status": "active", "pod_type": kind,
            "knowledge_key": "supplier:muller:x12:lead_time" if key.startswith("lead") else key,
            "generation_key": generation, "origin_keys": ["sap:o17" if key.startswith("lead") else "src:" + key]})
    # Explicit model migration/re-ingestion: schema changes are not silent.
    backend.configure_schema("canonical-pods", {"text": {"type": "string", "full_text_search": True,
        "embed": {"model": "local/models/encoder@reindexed", "dims": dimension}},
        "pod_type": {"type": "string", "filterable": True}, "generation_key": {"type": "string", "filterable": True}})
    reembed = backend.reembed_namespace("canonical-pods")
    queries = [
        ("Müller X12 delivery time", "lead-24", {"generation_key": "g8"}),
        ("supplier Müller procurement risk", "risk", {"pod_type": "context"}),
        ("approval threshold order", "rule", {"pod_type": "rule"}),
        ("buffer 56 days four days", "math", {"pod_type": "math"}),
    ]
    results = []
    for query, expected, filt in queries:
        out = backend.query("canonical-pods", {"rank_by": ["text", "ANN", ["Embed", query]], "filters": filt, "limit": 1}, principal="buyer")
        got = out["rows"][0]["id"] if out["rows"] else None
        results.append({"query": query, "expected": expected, "got": got, "correct": got == expected})
    hybrid = backend.query("canonical-pods", {"queries": [
        {"rank_by": ["text", "BM25", "Müller X12"], "filters": {"generation_key": "g8"}, "limit": 5},
        {"rank_by": ["text", "ANN", ["Embed", "delivery time Müller X12"]], "filters": {"generation_key": "g8"}, "limit": 5}],
        "rerank_by": ["RRF"], "limit": 2}, principal="buyer")
    report = {"schema": "native-embedding-reindex:v1", "encoder": str(encoder_path),
              "dimension": dimension, "documents": len(docs),
              "reembed": reembed,
              "queries": results, "recall_at_1": sum(x["correct"] for x in results) / len(results),
              "hybrid_top": [x["id"] for x in hybrid["rows"]], "namespace": backend.namespace_metadata("canonical-pods")}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    backend.close(); print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
