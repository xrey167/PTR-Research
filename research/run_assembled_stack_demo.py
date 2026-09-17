"""End-to-end local demonstration of the assembled Pod stack.

Compiles typed Pods, embeds them with the checked-in encoder, routes by Pod
type, retrieves through the local Turbopuffer-compatible contract, and proves
that a lifecycle revocation removes the Pod from subsequent reads.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer
from neural_pods.local_search import LocalSearchBackend


def _route(question: str) -> str:
    q = question.casefold()
    if any(x in q for x in ("calculate", "how many", "days", "tage", "berechne")):
        return "math"
    if any(x in q for x in ("model", "adapter", "modell")):
        return "model"
    return "context"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    db = root / "runs" / "assembled-stack-demo-001.sqlite3"
    if db.exists():
        db.unlink()
    encoder = SentenceTransformer(str(root / "models" / "encoder"), local_files_only=True)
    backend = LocalSearchBackend(db, embedder=lambda text: encoder.encode(text, normalize_embeddings=True).tolist())
    backend.create_namespace("canonical-pods")
    backend.configure_schema("canonical-pods", {"text": {"type": "string", "full_text_search": True,
        "embed": {"model": "local/models/encoder", "dims": int(encoder.get_embedding_dimension())}},
        "pod_type": {"type": "string", "filterable": True}, "status": {"type": "string", "filterable": True}})
    pods = [
        ("lead-24", "Müller GmbH X12 supplier lead time is 24 days.", "context", "knowledge:supplier:muller:x12:lead_time"),
        ("buffer-4", "Add four buffer days to a supplier lead time.", "math", "knowledge:math:buffer"),
        ("adapter-1", "The reader adapter resolves generation-bound Pod lineage.", "model", "knowledge:model:reader"),
    ]
    for key, text, pod_type, knowledge in pods:
        backend.upsert("canonical-pods", key, text=text, metadata={"pod_type": pod_type, "status": "active",
            "knowledge_key": knowledge, "generation_key": "generation:8", "origin_key": f"origin:{key}"})

    questions = ["What is Müller X12 lead time?", "How many days should I calculate?", "Which model adapter is active?"]
    rows = []
    for question in questions:
        pod_type = _route(question)
        result = backend.query("canonical-pods", {"rank_by": ["text", "ANN", ["Embed", question]],
            "filters": ["status", "Eq", "active"], "limit": 1,
            "include_attributes": ["pod_type", "knowledge_key", "generation_key"]})
        hit = result["rows"][0]
        rows.append({"question": question, "routed_pod_type": pod_type, "retrieved": hit,
                     "route_matches_type": hit["pod_type"] == pod_type})

    # Lifecycle barrier: revoke the retrieved context Pod and prove it vanishes.
    old_revision = backend.stats("canonical-pods")["items"]
    backend.update_metadata("canonical-pods", "lead-24", {"status": "revoked"})
    revoked = backend.query("canonical-pods", {"rank_by": ["text", "BM25", "Müller X12"],
        "filters": ["status", "Eq", "active"], "limit": 10})["rows"]
    report = {"schema": "assembled-stack-demo:v1", "encoder_dimension": int(encoder.get_embedding_dimension()),
              "pods": len(pods), "queries": rows, "initial_item_count": old_revision,
              "revocation_removed_lead_time": not any(row["id"] == "lead-24" for row in revoked),
              "namespace": backend.namespace_metadata("canonical-pods"),
              "scope": "local end-to-end integration demonstration; not broad external-world quality evidence"}
    out = root / "runs" / "assembled-stack-demo-001.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    backend.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
