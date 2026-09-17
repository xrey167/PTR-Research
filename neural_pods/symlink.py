"""Stable learned link handles, semantic clusters and current-generation resolution.

Handles are local model vocabulary, not replacement cryptographic identities.
Qwen learns the handles; this catalogue verifies and dereferences them.
"""
import json
import re
from .registry import InvalidState, digest
from .semantics import cluster_descriptor, normalized


def identity_descriptor(key, semantic):
    cluster = cluster_descriptor(semantic)
    return {"knowledge_key": key, "cluster_key": "cluster:v1:" + digest(cluster), "cluster": cluster,
            "subject": semantic["subject"], "component": semantic["component"],
            "aliases": semantic["retrieval"]["trusted_aliases"]}


class TemporalPortPlane:
    """Mutable alias/value ports bound to immutable Registry generations.

    Ports deliberately contain no trainable parameters.  A model may learn to
    request a port or follow a typed value handle, while this table resolves
    aliases and current generations authoritatively.  It is the local form of
    the Port Plane described in the imported R287 research.
    """

    def __init__(self, registry):
        self.registry = registry
        registry.db.executescript("""
            CREATE TABLE IF NOT EXISTS temporal_ports(
                knowledge_key TEXT PRIMARY KEY,
                port_slot INTEGER UNIQUE NOT NULL,
                generation_key TEXT NOT NULL,
                value_handle TEXT,
                revision INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active');
            CREATE TABLE IF NOT EXISTS temporal_port_aliases(
                alias_key TEXT PRIMARY KEY,
                alias_text TEXT NOT NULL,
                knowledge_key TEXT NOT NULL REFERENCES temporal_ports(knowledge_key));
        """)

    def bind(self, knowledge_key, aliases, *, value_handle=None, principal="local"):
        """Add aliases or refresh a port without optimizer steps."""
        generation = self.registry.head(knowledge_key)
        self.registry.snapshot([generation], principal)
        aliases = sorted({normalized(alias) for alias in aliases if isinstance(alias, str) and alias.strip()})
        if not aliases:
            raise ValueError("At least one nonempty alias is required")
        row = self.registry.db.execute("SELECT * FROM temporal_ports WHERE knowledge_key=?", (knowledge_key,)).fetchone()
        # Validate conflicts before changing the port so a rejected alias
        # update cannot leave a half-applied generation/revision update.
        for alias in aliases:
            existing = self.registry.db.execute("SELECT knowledge_key FROM temporal_port_aliases WHERE alias_key=?", (alias,)).fetchone()
            if existing is not None and existing[0] != knowledge_key:
                raise InvalidState("Alias is already bound to another knowledge port")
        with self.registry.transaction():
            if row is None:
                slot = self.registry.db.execute("SELECT COALESCE(MAX(port_slot), 0) + 1 FROM temporal_ports").fetchone()[0]
                revision = 1
                current_value = value_handle
                self.registry.db.execute("INSERT INTO temporal_ports VALUES(?,?,?,?,?, 'active')",
                                         (knowledge_key, slot, generation, current_value, revision))
            else:
                slot, revision = row["port_slot"], int(row["revision"])
                current_value = row["value_handle"] if value_handle is None else value_handle
                self.registry.db.execute("UPDATE temporal_ports SET generation_key=?, value_handle=?, revision=?, status='active' WHERE knowledge_key=?",
                                         (generation, current_value, revision + 1, knowledge_key))
                revision += 1
            for alias in aliases:
                self.registry.db.execute("INSERT OR REPLACE INTO temporal_port_aliases(alias_key,alias_text,knowledge_key) VALUES(?,?,?)",
                                         (alias, alias, knowledge_key))
        return {"knowledge_key": knowledge_key, "port_slot": slot, "generation_key": generation,
                "revision": revision, "value_handle": current_value, "aliases": aliases}

    def update_value(self, knowledge_key, value_handle, *, expected_revision=None, principal="local"):
        """Change the value handle at the current generation using CAS."""
        generation = self.registry.head(knowledge_key)
        self.registry.snapshot([generation], principal)
        row = self.registry.db.execute("SELECT * FROM temporal_ports WHERE knowledge_key=?", (knowledge_key,)).fetchone()
        if row is None:
            raise InvalidState("Knowledge has no temporal port")
        revision = int(row["revision"])
        if expected_revision is not None and expected_revision != revision:
            raise InvalidState(f"Port revision conflict: expected {expected_revision}, current {revision}")
        with self.registry.transaction():
            self.registry.db.execute("UPDATE temporal_ports SET generation_key=?, value_handle=?, revision=?, status='active' WHERE knowledge_key=?",
                                     (generation, value_handle, revision + 1, knowledge_key))
        return {"knowledge_key": knowledge_key, "port_slot": row["port_slot"],
                "generation_key": generation, "revision": revision + 1, "value_handle": value_handle}

    def resolve(self, alias, *, principal="local"):
        """Resolve an alias only against the currently authorized generation."""
        alias_key = normalized(alias)
        row = self.registry.db.execute("""SELECT p.*, a.alias_text FROM temporal_port_aliases a
            JOIN temporal_ports p ON p.knowledge_key=a.knowledge_key WHERE a.alias_key=?""", (alias_key,)).fetchone()
        if row is None or row["status"] != "active":
            raise InvalidState("Unknown or inactive temporal port alias")
        generation = self.registry.head(row["knowledge_key"])
        snapshot = self.registry.snapshot([generation], principal)
        return {"alias": row["alias_text"], "knowledge_key": row["knowledge_key"],
                "port_slot": row["port_slot"], "generation_key": generation,
                "revision": row["revision"], "value_handle": row["value_handle"],
                "snapshot": snapshot}


class NeuralSymlinks:
    def __init__(self, registry):
        self.registry = registry
        registry.db.executescript("""
            CREATE TABLE IF NOT EXISTS semantic_clusters(
                slot INTEGER PRIMARY KEY, cluster_key TEXT UNIQUE NOT NULL, descriptor TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS neural_symlinks(
                slot INTEGER PRIMARY KEY, knowledge_key TEXT UNIQUE NOT NULL,
                cluster_slot INTEGER NOT NULL REFERENCES semantic_clusters(slot),
                identity_key TEXT NOT NULL REFERENCES nodes(id), adapter_key TEXT REFERENCES nodes(id));
        """)

    def binding(self, knowledge_key):
        row = self.registry.db.execute("SELECT * FROM neural_symlinks WHERE knowledge_key=?", (knowledge_key,)).fetchone()
        if row is None: raise InvalidState("No embedded link registered for this knowledge")
        return dict(row)

    def register(self, knowledge_key, principal="local"):
        """Compile identity metadata, excluding the changing factual value.

        Its provenance retains the actual source roots used for metadata. Revoking
        those roots requires a new identity generation and new link adapter.
        """
        generation = self.registry.head(knowledge_key)
        self.registry.snapshot([generation], principal)
        p = self.registry.node(generation)["payload"]
        descriptor = identity_descriptor(knowledge_key, p["semantic"])
        row = self.registry.db.execute("SELECT * FROM neural_symlinks WHERE knowledge_key=?", (knowledge_key,)).fetchone()
        if row:
            try: self.registry.snapshot([row["identity_key"]], principal)
            except InvalidState: pass
            else:
                if self.registry.node(row["identity_key"])["payload"]["semantic"] == descriptor:
                    return dict(row)
        origin = self.registry.origin("semantic-link-compiler", knowledge_key, "1", descriptor, acl=p["acl"])
        identity = self.registry.publish("identity-link:" + knowledge_key, descriptor,
                    [origin, *self.registry.roots(generation)], principal=principal, acl=p["acl"])
        with self.registry.transaction():
            self.registry.db.execute("INSERT OR IGNORE INTO semantic_clusters(cluster_key,descriptor) VALUES(?,?)",
                                     (descriptor["cluster_key"], json.dumps(descriptor["cluster"], sort_keys=True)))
            cluster_slot = self.registry.db.execute("SELECT slot FROM semantic_clusters WHERE cluster_key=?", (descriptor["cluster_key"],)).fetchone()[0]
            self.registry.db.execute("""INSERT INTO neural_symlinks(knowledge_key,cluster_slot,identity_key) VALUES(?,?,?)
                ON CONFLICT(knowledge_key) DO UPDATE SET cluster_slot=excluded.cluster_slot,
                    identity_key=excluded.identity_key,adapter_key=NULL""", (knowledge_key, cluster_slot, identity))
        return self.binding(knowledge_key)

    def target_text(self, knowledge_key):
        b = self.binding(knowledge_key)
        return f"LINK {b['slot']} CLUSTER {b['cluster_slot']}"

    def attach(self, knowledge_key, payload, training_pairs, principal="local"):
        b = self.binding(knowledge_key)
        if not training_pairs or any(answer != self.target_text(knowledge_key) for _, answer in training_pairs):
            raise ValueError("Link training targets must match the registered identity and cluster")
        origin = self.registry.origin("qwen-link-training", knowledge_key, "1", training_pairs, acl=[principal])
        artifact = self.registry.artifact("lora", {**payload, "task": "link", "link_slot": b["slot"],
                "cluster_slot": b["cluster_slot"], "target_knowledge_key": knowledge_key}, [b["identity_key"], origin], principal)
        self.registry.db.execute("UPDATE neural_symlinks SET adapter_key=? WHERE knowledge_key=?", (artifact, knowledge_key))
        return artifact

    def active_binding(self, knowledge_key, principal="local"):
        b = self.binding(knowledge_key)
        if not b["adapter_key"]: raise InvalidState("Embedded link has no trained Qwen adapter")
        generation = self.registry.head(knowledge_key)
        self.registry.snapshot([b["identity_key"], b["adapter_key"], generation], principal)
        identity = self.registry.node(b["identity_key"])["payload"]["semantic"]
        current = identity_descriptor(knowledge_key, self.registry.node(generation)["payload"]["semantic"])
        if identity != current: raise InvalidState("Semantic identity or cluster changed; retrain the embedded link")
        adapter = self.registry.node(b["adapter_key"])["payload"]
        if b["identity_key"] not in adapter["generations"]:
            raise InvalidState("Embedded adapter belongs to another identity generation")
        p = adapter["payload"]
        if (p.get("target_knowledge_key"), p.get("link_slot"), p.get("cluster_slot")) != (knowledge_key, b["slot"], b["cluster_slot"]):
            raise InvalidState("Embedded adapter/catalogue binding mismatch")
        cluster_row = self.registry.db.execute("SELECT cluster_key FROM semantic_clusters WHERE slot=?", (b["cluster_slot"],)).fetchone()
        if cluster_row is None or cluster_row[0] != identity["cluster_key"]: raise InvalidState("Cluster catalogue mismatch")
        return {**b, "generation_key": generation, "cluster_key": identity["cluster_key"]}

    def resolve(self, model_text, *, expected_knowledge_key, principal="local"):
        match = re.fullmatch(r"LINK ([1-9][0-9]*) CLUSTER ([1-9][0-9]*)", model_text.strip())
        if not match: raise InvalidState("Model did not produce an exact embedded link")
        b = self.active_binding(expected_knowledge_key, principal)
        if (int(match[1]), int(match[2])) != (b["slot"], b["cluster_slot"]):
            raise InvalidState("Model predicted a different knowledge identity or cluster")
        row = self.registry.db.execute("SELECT artifact_key,vector_key FROM semantic_bindings WHERE generation_key=?", (b["generation_key"],)).fetchone()
        if row is None: raise InvalidState("Current link target has no indexed Pod")
        snapshot = self.registry.snapshot([b["identity_key"], b["adapter_key"], b["generation_key"], row["artifact_key"], row["vector_key"]], principal)
        return {**b, "artifact_key": row["artifact_key"], "snapshot": snapshot}

    def cluster_members(self, cluster_key, principal="local"):
        members = []
        for row in self.registry.db.execute("SELECT knowledge_key FROM neural_symlinks"):
            try: b = self.active_binding(row[0], principal)
            except InvalidState: continue
            if b["cluster_key"] == cluster_key: members.append(b["knowledge_key"])
        return sorted(members)
