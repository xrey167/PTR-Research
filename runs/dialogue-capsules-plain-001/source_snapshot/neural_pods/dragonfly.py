"""Dragonfly prototype: independently learned, generation-bound Pod addresses.

The shared text encoder is frozen. Each Pod learns a vector z and bias from
explicit positive/negative address questions, without answer values. There is
no shared trainable state across Pods, so revoking one Pod need not revoke all.
"""
import re
import torch
from .registry import InvalidState, digest
from .semantic_routing import SemanticRouter
from .semantics import normalized


def alias_training_questions(semantic, positives):
    """Compile trusted aliases into training data, never promote soft aliases."""
    aliases = sorted(set(semantic["retrieval"]["trusted_aliases"]))
    if not aliases or not all(isinstance(a, str) and a.strip() for a in aliases):
        raise ValueError("Nonempty trusted aliases are required")
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")(?!\w)", re.I)
    expanded = set(positives)
    for question in positives:
        if pattern.search(question):
            for alias in aliases: expanded.add(pattern.sub(lambda _: alias, question))
    for alias in aliases:
        expanded.add(alias)
        expanded.add(normalized(alias))
    return aliases, sorted(expanded)


def balanced_loss(logits, labels):
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses[labels.bool()].mean() + losses[~labels.bool()].mean()) / 2


class PodAddress(torch.nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.z = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, queries, features):
        similarity = torch.nn.functional.normalize(queries, dim=-1) @ torch.nn.functional.normalize(self.z, dim=0)
        # E_q, T_q and P_i enter via explicit metadata matches; they remain
        # independently enforced by the inherited pre-ANN and commit gates.
        return 12 * similarity + 2 * features[..., 0] + features[..., 1:].mean(dim=-1) - self.bias


class DragonflyRouter(SemanticRouter):
    SCHEMA = "dragonfly-pod-address:v2"

    def __init__(self, registry, encoder, directory, *, encoder_id):
        super().__init__(registry, encoder, directory)
        if not isinstance(encoder_id, str) or not encoder_id:
            raise ValueError("A pinned encoder identity is required")
        self.encoder_id = encoder_id
        self.addresses = {}
        registry.db.execute("""CREATE TABLE IF NOT EXISTS dragonfly_bindings(
            generation_key TEXT PRIMARY KEY REFERENCES nodes(id),
            representation_key TEXT NOT NULL REFERENCES nodes(id))""")

    def fit(self, examples, principal="local"):
        raise ValueError("Use fit_pod with independently sourced positive and negative questions per Pod")

    def fit_pod(self, generation_key, positives, negatives, *, principal="local", training_parents=(), steps=300,
                negative_aliases=()):
        """Questions are trusted, independently authored training inputs.

        If questions were derived from other evidence, supply that evidence in
        training_parents: revocation then deliberately follows those dependencies.
        No examples are harvested from other Pods automatically.
        """
        if not positives or not negatives or steps < 1:
            raise ValueError("Positive and negative address examples and positive steps are required")
        questions = list(positives) + list(negatives)
        if not all(isinstance(q, str) and q.strip() for q in questions):
            raise ValueError("Address questions must be nonempty text")
        if set(positives) & set(negatives): raise ValueError("Conflicting address labels")
        item = next((i for i in self.bindings() if i["generation_key"] == generation_key), None)
        if item is None: raise InvalidState("Index the Pod before training its representation")
        aliases, positives = alias_training_questions(item["node"]["semantic"], positives)
        negatives = sorted(set(negatives) | set(negative_aliases))
        if not all(isinstance(q, str) and q.strip() for q in negatives): raise ValueError("Invalid negative aliases")
        if {normalized(q) for q in positives} & {normalized(q) for q in negatives}:
            raise ValueError("Trusted alias conflicts with a negative training example")
        questions = positives + negatives
        dependencies = [item["vector_key"], item["artifact_key"], *training_parents]
        self.registry.snapshot(dependencies, principal)
        initial = self.registry.node(item["vector_key"])["payload"]["payload"]["embedding"]
        model = PodAddress(initial)
        x = torch.tensor([self.encoder.encode(q, normalize_embeddings=True).tolist() for q in questions])
        y = torch.tensor([1.] * len(positives) + [0.] * len(negatives))
        # Treat every training question as a metadata-matched candidate. This
        # counterfactual prevents entity/ACL flags from solving the training task
        # without learning the question representation. Runtime uses real flags.
        features = torch.ones((len(questions), len(self.FEATURES)))
        features[:, 0] = x @ torch.tensor(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.04)
        initial_z = model.z.detach().clone()
        with torch.no_grad(): before = float(balanced_loss(model(x, features), y))
        for _ in range(steps):
            optimizer.zero_grad()
            loss = balanced_loss(model(x, features), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, features)
            after = float(balanced_loss(logits, y))
            accuracy = float(((logits >= 0) == y.bool()).float().mean())
            negative_accuracy = float((logits[~y.bool()] < 0).float().mean())
        alias_scores = {}
        with torch.no_grad():
            for alias in aliases:
                vector = torch.tensor(self.encoder.encode(alias, normalize_embeddings=True))
                fs = torch.ones(len(self.FEATURES))
                fs[0] = vector @ torch.tensor(initial)
                alias_scores[alias] = float(model(vector, fs).sigmoid())
        if min(alias_scores.values()) < 0.8 or negative_accuracy < 0.9:
            raise InvalidState("Alias training coverage or negative discrimination insufficient; representation was not activated")
        payload = {"schema": self.SCHEMA, "encoder_id": self.encoder_id, "features": self.FEATURES,
                   "dimension": len(initial), "z": model.z.detach().tolist(), "bias": float(model.bias.detach()),
                   "steps": steps, "loss_before": before, "loss_after": after,
                   "training_accuracy": accuracy, "negative_accuracy": negative_accuracy,
                   "z_delta_l2": float(torch.linalg.vector_norm(model.z.detach() - initial_z)),
                   "alias_training": {"policy": "trusted-alias-expansion:v1", "aliases": aliases,
                       "scores": alias_scores, "positive_count": len(positives), "negative_count": len(negatives)}}
        origin = self.registry.origin("dragonfly-address-training", generation_key, "1",
                    {"positives": list(positives), "negatives": list(negatives), "encoder_id": self.encoder_id,
                     "training_policy": "matched-metadata-counterfactual:balanced:v2"}, acl=[principal])
        key = self.registry.artifact("router", payload, [*dependencies, origin], principal)
        self.registry.db.execute("""INSERT INTO dragonfly_bindings VALUES(?,?) ON CONFLICT(generation_key)
            DO UPDATE SET representation_key=excluded.representation_key""", (generation_key, key))
        self.addresses[generation_key] = (key, model)
        return {k: v for k, v in payload.items() if k != "z"} | {"representation_key": key, "training_origin": origin}

    def load_weights(self):
        loaded = {}
        for row in self.registry.db.execute("SELECT * FROM dragonfly_bindings"):
            node = self.registry.node(row["representation_key"])
            p = node["payload"]["payload"]
            parents = [r[0] for r in self.registry.db.execute("SELECT parent FROM edges WHERE child=?", (node["id"],))]
            expected = "router:" + digest({"kind": "router", "payload": node["payload"], "parents": sorted(set(parents))})
            if node["kind"] != "router" or node["id"] != expected:
                raise InvalidState("Dragonfly representation hash mismatch")
            if p["schema"] not in {self.SCHEMA, "dragonfly-pod-address:v1"} or p["encoder_id"] != self.encoder_id or p["features"] != self.FEATURES:
                raise InvalidState("Incompatible Dragonfly representation")
            if row["generation_key"] not in node["payload"]["generations"]:
                raise InvalidState("Dragonfly representation belongs to another generation")
            dim = len(self.encoder.encode("embedding dimension", normalize_embeddings=True))
            if p["dimension"] != dim or len(p["z"]) != dim:
                raise InvalidState("Dragonfly embedding dimension mismatch")
            model = PodAddress(p["z"])
            with torch.no_grad(): model.bias.fill_(p["bias"])
            model.eval()
            loaded[row["generation_key"]] = (node["id"], model)
        self.addresses = loaded

    def recognize_alias(self, alias, *, principal="local", threshold=0.8, margin=0.15):
        """Neural-only alias diagnostic using activated z vectors, no alias lookup.

        Advisory identity prediction: normal select still validates explicit hard
        query constraints. Similarity alone never grants identity or ACL authority.
        """
        if not isinstance(alias, str) or not alias.strip(): raise ValueError("Alias text is required")
        vector = self.encoder.encode(alias, normalize_embeddings=True)
        by_subject = {}
        for item in self.bindings():
            try:
                deps = self.learned_dependencies(item)
                self.registry.snapshot(deps, principal)
            except InvalidState: continue
            payload = self.registry.node(deps[0])["payload"]["payload"]
            if payload["schema"] != self.SCHEMA: continue
            embedding = self.registry.node(item["vector_key"])["payload"]["payload"]["embedding"]
            score = self.learned_score(vector, [float(vector @ embedding)] + [1.] * 8, item)
            subject = item["node"]["semantic"]["subject"]
            if subject not in by_subject or score > by_subject[subject][0]:
                by_subject[subject] = (score, deps)
        ranked = sorted(by_subject.items(), key=lambda x: x[1][0], reverse=True)
        if not ranked or ranked[0][1][0] < threshold:
            raise InvalidState("Neural alias recognition deferred")
        if len(ranked) > 1 and ranked[0][1][0] - ranked[1][1][0] < margin:
            raise InvalidState("Neural alias recognition is ambiguous")
        subject, (score, deps) = ranked[0]
        snapshot = self.registry.snapshot(deps, principal)
        return {"subject": subject, "score": score, "representation_key": deps[0],
                "snapshot": snapshot, "advisory": True}

    def learned_dependencies(self, item):
        entry = self.addresses.get(item["generation_key"])
        if entry is None: raise InvalidState("Pod has no trained Dragonfly representation")
        parents = self.registry.node(entry[0])["payload"]["parent_artifact_keys"]
        if item["artifact_key"] not in parents or item["vector_key"] not in parents:
            raise InvalidState("Dragonfly representation is bound to a different Pod artifact")
        return [entry[0]]

    def candidate_dependencies(self, item, learned):
        return self.learned_dependencies(item) if learned else []

    def learned_score(self, vector, features, item):
        model = self.addresses[item["generation_key"]][1]
        with torch.inference_mode():
            return float(model(torch.tensor(vector), torch.tensor(features)).sigmoid())

    def routing_details(self, item, learned):
        if not learned: return {"routing_mode": "cosine-baseline"}
        key = self.learned_dependencies(item)[0]
        payload = self.registry.node(key)["payload"]["payload"]
        return {"routing_mode": payload["schema"], "representation_key": key,
                "alias_training": payload.get("alias_training")}
