"""Metadata-constrained Qdrant retrieval with a learned candidate scorer.

Authorization/time/generation are hard gates before ANN and again at commit.
The neural scorer is advisory and cannot override those gates.
"""
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
import torch
from qdrant_client import QdrantClient, models
from .registry import InvalidState
from .semantics import evidence_text, normalized, resolve_query


class SemanticRouter:
    FEATURES = ["semantic_similarity", "subject_match", "component_match", "predicate_match",
                "role_match", "current_generation", "time_valid", "authorized", "tag_overlap"]

    def __init__(self, registry, encoder, directory):
        self.registry, self.encoder = registry, encoder
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.directory / "qdrant"))
        dim = len(encoder.encode("embedding dimension", normalize_embeddings=True))
        if not self.client.collection_exists("semantic_pods"):
            self.client.create_collection("semantic_pods", vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        registry.db.execute("""CREATE TABLE IF NOT EXISTS semantic_bindings(
            generation_key TEXT PRIMARY KEY REFERENCES nodes(id), artifact_key TEXT REFERENCES nodes(id),
            vector_key TEXT REFERENCES nodes(id), point_id TEXT NOT NULL)""")
        torch.manual_seed(20260915)
        self.net = torch.nn.Sequential(torch.nn.Linear(len(self.FEATURES), 12), torch.nn.Tanh(), torch.nn.Linear(12, 1))
        self.router_artifact = None

    def index(self, generation_key, artifact_key, principal="local"):
        self.registry.snapshot([generation_key, artifact_key], principal)
        node = self.registry.node(generation_key)["payload"]
        artifact = self.registry.node(artifact_key)
        if generation_key not in artifact["payload"].get("generations", {}):
            raise InvalidState("Payload artifact belongs to a different knowledge generation")
        s = node["semantic"]
        r = s["retrieval"]
        advisory = " ".join(str(a["value"]) for a in r["soft_annotations"] if a["field"] in {"tags", "topics", "aliases"})
        description = f"{s['subject_label']} {s.get('component_label') or ''} {s['predicate']} {s['role']} {' '.join(r['trusted_aliases'])} {' '.join(r['tags'])} {advisory}"
        vector = self.encoder.encode(description, normalize_embeddings=True).tolist()
        vector_key = self.registry.artifact("vector", {"embedding": vector, "description": description,
                "transform_chain": [*s["transform_chain"], "semantic-vector-index:v1"]}, [generation_key], principal)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, generation_key))
        from datetime import datetime
        life = node["lifecycle"]
        payload = {"knowledge_key": node["knowledge_key"], "generation_key": generation_key,
                   "generation": node["generation"], "subject": s["subject"], "component": s["component"],
                   "predicate": s["predicate"], "role": s["role"], "type": s["type"], "status": "active",
                   "tags": r["tags"], "domain": r["domain"], "language": r["language"], "acl": node["acl"],
                   "valid_from": datetime.fromisoformat(life["valid_from"]).timestamp() if life.get("valid_from") else -1e20,
                   "valid_until": datetime.fromisoformat(life["valid_until"]).timestamp() if life.get("valid_until") else 1e20}
        self.client.upsert("semantic_pods", [models.PointStruct(id=point_id, vector=vector, payload=payload)])
        self.registry.db.execute("INSERT OR REPLACE INTO semantic_bindings VALUES(?,?,?,?)", (generation_key, artifact_key, vector_key, point_id))
        return vector_key

    def bindings(self):
        result = []
        for row in self.registry.db.execute("SELECT * FROM semantic_bindings"):
            item = dict(row)
            item["node"] = self.registry.node(item["generation_key"])["payload"]
            result.append(item)
        return result

    def features(self, vector, query, item, principal):
        from datetime import datetime
        p = item["node"]
        s, life = p["semantic"], p["lifecycle"]
        embedding = self.registry.node(item["vector_key"])["payload"]["payload"]["embedding"]
        now = self.registry.clock()
        time_valid = (not life.get("valid_from") or now >= datetime.fromisoformat(life["valid_from"])) and (not life.get("valid_until") or now < datetime.fromisoformat(life["valid_until"]))
        acl_valid = all("*" in n["payload"].get("acl", ["*"]) or principal in n["payload"].get("acl", [])
                        for n in self.registry.ancestors(item["artifact_key"]))
        query_tags = set(query.get("tags", []))
        tags = set(s["retrieval"]["tags"])
        return [float(np.dot(vector, embedding)), float(query["subject"] == s["subject"]),
                float(query["component"] is None or query["component"] == s["component"]),
                float(query["predicate"] == s["predicate"]), float(query["role"] == s["role"]),
                float(self.registry.head(p["knowledge_key"]) == item["generation_key"]), float(time_valid), float(acl_valid),
                len(tags & query_tags) / len(query_tags) if query_tags else 1.0]

    def fit(self, examples, principal="local"):
        """Train address relevance from (question, KnowledgeKey), without answer values."""
        items = self.bindings()
        semantics = [i["node"]["semantic"] for i in items]
        xs, ys = [], []
        for question, target in examples:
            query = resolve_query(question, semantics)
            vector = self.encoder.encode(question, normalize_embeddings=True)
            for item in items:
                features = self.features(vector, query, item, principal)
                label = float(item["node"]["knowledge_key"] == target and all(features[5:8]))
                xs.append(features)
                ys.append(label)
                if label:
                    # Explicit invalid-metadata negatives, never an authorization mechanism.
                    for position in [5, 6, 7]:
                        invalid = list(features)
                        invalid[position] = 0.0
                        xs.append(invalid)
                        ys.append(0.0)
        if not ys or not any(ys): raise ValueError("Router training needs positive examples")
        x, y = torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.02)
        for _ in range(220):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net(x), y)
            loss.backward()
            optimizer.step()
        self.net.eval()
        checkpoint = self.directory / "semantic-router.pt"
        torch.save({"state_dict": self.net.state_dict(), "features": self.FEATURES}, checkpoint)
        with checkpoint.open("rb") as handle: sha = hashlib.file_digest(handle, "sha256").hexdigest()
        origin = self.registry.origin("routing-catalogue", "semantic-address-examples", "1", {"examples": examples, "features": xs, "labels": ys}, acl=[principal])
        knowledge = self.registry.publish("routing:semantic-addresses", {"features": self.FEATURES}, [origin], principal=principal, acl=[principal])
        self.router_artifact = self.registry.artifact("router", {"sha256": sha, "architecture": "9/12/1 MLP",
              "examples": len(examples), "pairs": len(ys), "loss": float(loss.detach())}, [knowledge], principal=principal)
        (self.directory / "semantic-router.json").write_text(json.dumps({"artifact_key": self.router_artifact}), encoding="utf-8")
        return {"pairs": len(ys), "loss": float(loss.detach()), "artifact_key": self.router_artifact}

    def load_weights(self):
        meta = json.loads((self.directory / "semantic-router.json").read_text(encoding="utf-8"))
        self.router_artifact = meta["artifact_key"]
        checkpoint = self.directory / "semantic-router.pt"
        with checkpoint.open("rb") as handle: sha = hashlib.file_digest(handle, "sha256").hexdigest()
        expected = self.registry.node(self.router_artifact)["payload"]["payload"]["sha256"]
        if sha != expected: raise InvalidState("Semantic router checkpoint hash mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["features"] != self.FEATURES: raise InvalidState("Router feature schema mismatch")
        self.net.load_state_dict(saved["state_dict"])
        self.net.eval()

    def candidate_dependencies(self, item, learned):
        return []

    def learned_dependencies(self, item):
        if not self.router_artifact: raise InvalidState("Semantic router has not been trained or loaded")
        return [self.router_artifact]

    def learned_score(self, vector, features, item):
        with torch.inference_mode():
            return float(self.net(torch.tensor(features)).sigmoid().item())

    def routing_details(self, item, learned):
        return {}

    def select(self, question, principal="local", filters=None, learned=True):
        items = []
        for item in self.bindings():
            try: self.registry.snapshot([item["vector_key"], item["artifact_key"],
                                        *self.candidate_dependencies(item, learned)], principal)
            except InvalidState: continue
            items.append(item)
        query = resolve_query(question, [i["node"]["semantic"] for i in items])
        filters = filters or {}
        if not set(filters) <= {"tags", "domain", "language", "type"}: raise ValueError("Unsupported retrieval filter")
        query["tags"] = filters.get("tags", [])
        eligible = []
        for item in items:
            s = item["node"]["semantic"]
            if any(s[k] != query[k] for k in ["subject", "predicate", "role"]): continue
            if query["component"] and s["component"] != query["component"]: continue
            if filters.get("type") and s["type"] != filters["type"]: continue
            if any(filters.get(k) and s["retrieval"][k] != filters[k] for k in ["domain", "language"]): continue
            if not set(filters.get("tags", [])) <= set(s["retrieval"]["tags"]): continue
            try: self.registry.snapshot([item["vector_key"], item["artifact_key"]], principal)
            except InvalidState: continue
            eligible.append(item)
        if not eligible: raise InvalidState("No authorized, current, time-valid semantic candidate")
        if query["component"] is None and len({i["node"]["semantic"]["component"] for i in eligible}) != 1:
            raise InvalidState("Component is ambiguous; specify the part")
        must = [models.FieldCondition(key="generation_key", match=models.MatchAny(any=[i["generation_key"] for i in eligible])),
                models.FieldCondition(key="subject", match=models.MatchValue(value=query["subject"])),
                models.FieldCondition(key="predicate", match=models.MatchValue(value=query["predicate"])),
                models.FieldCondition(key="role", match=models.MatchValue(value=query["role"])),
                models.FieldCondition(key="status", match=models.MatchValue(value="active")),
                models.FieldCondition(key="acl", match=models.MatchAny(any=["*", principal])),
                models.FieldCondition(key="valid_from", range=models.Range(lte=self.registry.clock().timestamp())),
                models.FieldCondition(key="valid_until", range=models.Range(gt=self.registry.clock().timestamp()))]
        if query["component"]: must.append(models.FieldCondition(key="component", match=models.MatchValue(value=query["component"])))
        for field in ["type", "domain", "language"]:
            if filters.get(field): must.append(models.FieldCondition(key=field, match=models.MatchValue(value=filters[field])))
        for tag in filters.get("tags", []): must.append(models.FieldCondition(key="tags", match=models.MatchValue(value=tag)))
        vector = self.encoder.encode(question, normalize_embeddings=True)
        hits = self.client.query_points("semantic_pods", query=vector.tolist(), query_filter=models.Filter(must=must), limit=min(20, len(eligible))).points
        by_id = {i["point_id"]: i for i in eligible}
        candidates = []
        for hit in hits:
            if str(hit.id) not in by_id: continue
            item = by_id[str(hit.id)]  # Ignore index-provided evidence and adapter names.
            features = self.features(vector, query, item, principal)
            if learned: self.registry.snapshot(self.learned_dependencies(item), principal)
            score = self.learned_score(vector, features, item) if learned else features[0]
            candidates.append((score, item, features))
        if not candidates: raise InvalidState("Semantic index has no eligible result; synchronize the projection")
        score, item, features = max(candidates, key=lambda entry: entry[0])
        if learned and score < 0.5: raise InvalidState("Semantic router deferred")
        deps = [item["vector_key"], item["artifact_key"]] + (self.learned_dependencies(item) if learned else [])
        snapshot = self.registry.snapshot(deps, principal)
        s = item["node"]["semantic"]
        return {"knowledge_key": item["node"]["knowledge_key"], "generation_key": item["generation_key"],
                "generation": item["node"]["generation"], "artifact_key": item["artifact_key"],
                "query_semantics": query, "evidence": evidence_text(s), "features": dict(zip(self.FEATURES, features)),
                "score": score, "snapshot": snapshot, "ann_eligible_count": len(eligible),
                "qdrant_filter": models.Filter(must=must).model_dump(mode="json", exclude_none=True),
                **self.routing_details(item, learned)}

    def close(self): self.client.close()
