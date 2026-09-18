"""Join semantic routing, model-produced links and KV capsules under one barrier.

The link predictor is supplied by the caller (e.g. the trained Qwen link LoRA).
This module does not replace a missing/incorrect prediction with a catalogue ID.
"""
from dataclasses import dataclass
from neural_pods.registry import InvalidState
from neural_pods.symlink import NeuralSymlinks


@dataclass
class LinkPrediction:
    text: str
    proof_key: str


@dataclass
class PreparedCapsule:
    cache: object
    snapshot: object
    knowledge_key: str
    generation_key: str
    artifact_key: str
    model_prediction: str


class LinkedCapsules:
    def __init__(self, router, capsules):
        if router.registry is not capsules.registry:
            raise ValueError('Routing and capsules must share a registry')
        self.router, self.capsules = router, capsules
        self.registry = capsules.registry
        self.links = NeuralSymlinks(self.registry)
        self.registry.db.execute('''CREATE TABLE IF NOT EXISTS capsule_variants(
            generation_key TEXT NOT NULL REFERENCES nodes(id),
            model_sha256 TEXT NOT NULL, artifact_key TEXT NOT NULL REFERENCES nodes(id),
            PRIMARY KEY(generation_key, model_sha256))''')

    def bind_variant(self, generation_key, artifact_key, principal='buyer'):
        node = self.registry.node(artifact_key)
        if generation_key not in node['payload'].get('parent_artifact_keys', []):
            raise InvalidState('Capsule is not derived directly from this generation')
        self.capsules.load(artifact_key, principal)
        with self.registry.transaction():
            self.registry._valid(generation_key, principal)
            self.registry._valid(artifact_key, principal)
            self.registry.db.execute('''INSERT INTO capsule_variants VALUES(?,?,?)
                ON CONFLICT(generation_key,model_sha256) DO UPDATE SET artifact_key=excluded.artifact_key''',
                (generation_key,self.capsules.model_sha256,artifact_key))

    def prepare(self, question, predict_link, principal='buyer', learned=True):
        first = self.router.select(question, principal=principal, learned=learned)
        binding = self.links.active_binding(first['knowledge_key'], principal)
        prediction = predict_link(question, binding['adapter_key'])
        proof_dependencies = []
        if isinstance(prediction, LinkPrediction):
            proof = self.registry.node(prediction.proof_key)
            payload = proof['payload'].get('payload', {})
            if proof['kind'] != 'answer' or (payload.get('task'),payload.get('question'),payload.get('text'),payload.get('adapter_key')) != (
                    'link_prediction',question,prediction.text,binding['adapter_key']):
                raise InvalidState('Stored model prediction does not match this request')
            if binding['adapter_key'] not in {n['id'] for n in self.registry.ancestors(prediction.proof_key)}:
                raise InvalidState('Stored prediction lacks the selected adapter lineage')
            proof_dependencies = [prediction.proof_key]
            self.registry.snapshot(proof_dependencies,principal)
            prediction = prediction.text
        resolved = self.links.resolve(prediction, expected_knowledge_key=first['knowledge_key'], principal=principal)
        current = self.router.select(question, principal=principal, learned=learned)
        if (current['knowledge_key'], current['generation_key'], current['artifact_key']) != (
                resolved['knowledge_key'], resolved['generation_key'], resolved['artifact_key']):
            raise InvalidState('Routed capsule and embedded target disagree')
        variant = self.registry.db.execute('SELECT artifact_key FROM capsule_variants WHERE generation_key=? AND model_sha256=?',
                    (resolved['generation_key'],self.capsules.model_sha256)).fetchone()
        artifact_key = variant[0] if variant else resolved['artifact_key']
        if variant and resolved['generation_key'] not in self.registry.node(artifact_key)['payload']['parent_artifact_keys']:
            raise InvalidState('Capsule variant belongs to another generation')
        cache, capsule_snapshot = self.capsules.load(artifact_key, principal)
        combined = self.registry.snapshot([
            *first['snapshot'].artifacts, *current['snapshot'].artifacts,
            *resolved['snapshot'].artifacts, *capsule_snapshot.artifacts, *proof_dependencies], principal)
        return PreparedCapsule(cache, combined, current['knowledge_key'], current['generation_key'],
                               artifact_key, prediction)

    def commit(self, prepared, text):
        return self.registry.commit(prepared.snapshot, text)
