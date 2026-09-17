"""Real local Qdrant search followed by a small learned address router."""
import uuid
import re
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from .data import questions
from .registry import InvalidState


class Router:
    def __init__(self, encoder_path, facts, registry, directory):
        self.encoder = SentenceTransformer(str(encoder_path), device="cpu", local_files_only=True)
        self.facts = {f.key: f for f in facts}
        self.keys = list(self.facts)
        self.registry = registry
        self.client = QdrantClient(path=str(Path(directory) / "qdrant"))
        self.client.create_collection("pods", vectors_config=models.VectorParams(
            size=self.encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE))
        dim = self.encoder.get_sentence_embedding_dimension()
        torch.manual_seed(20260915)
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, 48), torch.nn.Tanh(), torch.nn.Linear(48, len(facts)))
        qs, ys = [], []
        for i, fact in enumerate(facts):
            qs.extend(questions(fact, "train"))
            ys.extend([i] * len(questions(fact, "train")))
        x = torch.tensor(self.encoder.encode(qs, normalize_embeddings=True))
        y = torch.tensor(ys)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.01)
        for _ in range(180):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.net(x), y)
            loss.backward()
            optimizer.step()
        self.net.eval()
        checkpoint = Path(directory) / "router.pt"
        torch.save({"state_dict": self.net.state_dict(), "keys": self.keys}, checkpoint)
        # Address semantics have their own source lineage, independent of mutable values.
        origin = registry.origin("routing-schema", "supplier-addresses", "1", {"questions": qs, "labels": ys})
        routing_k = registry.publish("routing:supplier-addresses", {"keys": self.keys}, [origin])
        with checkpoint.open("rb") as handle:
            checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        self.artifact = registry.artifact("router", {"checkpoint_sha256": checkpoint_hash,
                                "architecture": "MiniLM + 384/48/3 MLP", "training_loss": float(loss.detach())}, [routing_k])
        (Path(directory) / "router-manifest.json").write_text(json.dumps({
            "artifact": self.artifact, "sha256": checkpoint_hash,
            "facts": [{"key": f.key, "entity": f.entity, "component": f.component,
                       "value": f.value, "generation": f.generation} for f in facts]}), encoding="utf-8")

    @classmethod
    def load(cls, encoder_path, registry, directory):
        from .data import Fact
        self = cls.__new__(cls)
        directory = Path(directory)
        metadata = json.loads((directory / "router-manifest.json").read_text(encoding="utf-8"))
        self.artifact = metadata["artifact"]
        registry.snapshot([self.artifact])
        checkpoint = directory / "router.pt"
        with checkpoint.open("rb") as handle:
            actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        expected = registry.node(self.artifact)["payload"]["payload"]["checkpoint_sha256"]
        if actual_hash != expected:
            raise InvalidState("Router checkpoint hash mismatch")
        self.encoder = SentenceTransformer(str(encoder_path), device="cpu", local_files_only=True)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.keys = data["keys"]
        self.facts = {f["key"]: Fact(**f) for f in metadata["facts"]}
        self.net = torch.nn.Sequential(torch.nn.Linear(self.encoder.get_sentence_embedding_dimension(), 48),
                                      torch.nn.Tanh(), torch.nn.Linear(48, len(self.keys)))
        self.net.load_state_dict(data["state_dict"])
        self.net.eval()
        self.registry = registry
        self.client = QdrantClient(path=str(directory / "qdrant"))
        return self

    def upsert(self, fact, knowledge_node, pod_node, adapter_name):
        description = f"Supplier {fact.entity}; component {fact.component}; delivery lead time in days."
        vector = self.encoder.encode(description, normalize_embeddings=True).tolist()
        artifact = self.registry.artifact("vector", {"embedding": vector, "description": description}, [knowledge_node])
        payload = {"key": fact.key, "generation": fact.generation, "vector_artifact": artifact,
                   "pod_artifact": pod_node, "adapter": adapter_name, "evidence": fact.evidence,
                   "knowledge_node": knowledge_node}
        self.client.upsert("pods", [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, fact.key)), vector=vector, payload=payload)])
        self.facts[fact.key] = fact
        return artifact

    def candidates(self, question):
        vector = self.encoder.encode(question, normalize_embeddings=True)
        hits = self.client.query_points("pods", query=vector.tolist(), limit=len(self.keys)).points
        with torch.inference_mode():
            probabilities = self.net(torch.tensor(vector).unsqueeze(0)).softmax(-1)[0].tolist()
        valid = []
        mentioned = {f.key for f in self.facts.values()
                     if re.search(r"\b" + re.escape(f.component) + r"\b", question, re.IGNORECASE)}
        for hit in hits:
            p = dict(hit.payload)
            if mentioned and p["key"] not in mentioned:
                continue
            try:
                self.registry.snapshot([p["vector_artifact"], p["pod_artifact"]])
                current = self.registry.head(p["key"])
                vector_header = self.registry.node(p["vector_artifact"])["payload"]
                pod_header = self.registry.node(p["pod_artifact"])["payload"]
                if current not in vector_header["generations"] or current not in pod_header["generations"]:
                    raise InvalidState("Index lineage does not match canonical identity")
                knowledge = self.registry.node(current)["payload"]
                semantic = knowledge["semantic"]
                # The vector index is an address cache, never the source of truth.
                p.update(knowledge_node=current, generation=knowledge["generation"],
                         adapter=pod_header["payload"]["adapter"],
                         evidence=f"The delivery lead time for {semantic['component']} from {semantic['subject']} is {semantic['object']['value']} days.")
            except InvalidState:
                continue
            p.update(similarity=float(hit.score), router_score=probabilities[self.keys.index(p["key"])])
            valid.append(p)
        return valid

    def select(self, question, learned=True):
        if learned: self.registry.snapshot([self.artifact])
        candidates = self.candidates(question)
        if not candidates:
            raise InvalidState("No active candidate")
        chosen = max(candidates, key=lambda p: p["router_score"] if learned else p["similarity"])
        # Thresholds fixed before held-out evaluation; not a calibrated safety classifier.
        if chosen["similarity"] < 0.35 or (learned and chosen["router_score"] < 0.55):
            raise InvalidState("Router deferred: insufficient address confidence")
        return chosen

    def close(self):
        self.client.close()
