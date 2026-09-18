"""Experimental identity-trained addresses; current fact gates remain mandatory.

New training only: legacy generation-bound tensors are never relabelled.
Identity provenance retains its original evidence roots. Revoking those roots
still invalidates the address, even after a factual value update.
"""
import torch
from neural_pods.dragonfly import DragonflyRouter, PodAddress, alias_training_questions, balanced_loss
from neural_pods.registry import InvalidState, digest, canonical
from neural_pods.symlink import NeuralSymlinks, identity_descriptor
from neural_pods.semantics import normalized


class IdentityDragonfly(DragonflyRouter):
    SCHEMA = 'dragonfly-identity-address:research-v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.links = NeuralSymlinks(self.registry)
        self.registry.db.execute('''CREATE TABLE IF NOT EXISTS identity_dragonfly(
            identity_key TEXT PRIMARY KEY REFERENCES nodes(id),
            representation_key TEXT NOT NULL REFERENCES nodes(id))''')

    def fit_pod(self, generation_key, positives, negatives, *, principal='local',
                training_parents=(), steps=300, negative_aliases=()):
        node = self.registry.node(generation_key)['payload']
        self.registry.snapshot([generation_key], principal)
        binding = self.links.register(node['knowledge_key'], principal)
        identity = binding['identity_key']
        descriptor = self.registry.node(identity)['payload']['semantic']
        if steps < 1 or not positives or not negatives:
            raise ValueError('Positive/negative questions and training steps required')
        if not all(isinstance(q, str) and q.strip() for q in [*positives, *negatives, *negative_aliases]):
            raise ValueError('Nonempty training questions required')
        aliases, positives = alias_training_questions(node['semantic'], positives)
        negatives = sorted(set(negatives) | set(negative_aliases))
        if {normalized(q) for q in positives} & {normalized(q) for q in negatives}:
            raise ValueError('Conflicting address labels')
        dependencies = [identity, *training_parents]
        self.registry.snapshot(dependencies, principal)
        # This seed contains no factual object, generation, soft tags or Pod data.
        initial = self.encoder.encode(canonical(descriptor), normalize_embeddings=True).tolist()
        model = PodAddress(initial)
        questions = positives + negatives
        x = torch.tensor([self.encoder.encode(q, normalize_embeddings=True).tolist() for q in questions])
        y = torch.tensor([1.] * len(positives) + [0.] * len(negatives))
        features = torch.ones((len(questions), len(self.FEATURES)))
        features[:, 0] = x @ torch.tensor(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.04)
        z_before = model.z.detach().clone()
        with torch.no_grad(): before = float(balanced_loss(model(x, features), y))
        for _ in range(steps):
            optimizer.zero_grad()
            loss = balanced_loss(model(x, features), y)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, features)
            after = float(balanced_loss(logits, y))
            negative_accuracy = float((logits[~y.bool()] < 0).float().mean())
            scores = {}
            for alias in aliases:
                vector = torch.tensor(self.encoder.encode(alias, normalize_embeddings=True))
                fs = torch.ones(len(self.FEATURES)); fs[0] = vector @ torch.tensor(initial)
                scores[alias] = float(model(vector, fs).sigmoid())
        if min(scores.values()) < .8 or negative_accuracy < .9:
            raise InvalidState('Identity address training did not meet discrimination gate')
        payload = dict(schema=self.SCHEMA, encoder_id=self.encoder_id, features=self.FEATURES,
            identity_key=identity, dimension=len(initial), initial=initial,
            z=model.z.detach().tolist(), bias=float(model.bias.detach()), steps=steps,
            loss_before=before, loss_after=after, negative_accuracy=negative_accuracy,
            z_delta_l2=float(torch.linalg.vector_norm(model.z.detach()-z_before)),
            alias_training={'aliases': aliases, 'scores': scores})
        origin = self.registry.origin('identity-address-training', identity, '1',
            {'positives': positives, 'negatives': negatives, 'encoder_id': self.encoder_id,
             'policy': self.SCHEMA}, acl=[principal])
        key = self.registry.artifact('router', payload, [*dependencies, origin], principal)
        self.registry.db.execute('INSERT OR REPLACE INTO identity_dragonfly VALUES(?,?)', (identity, key))
        self.addresses[identity] = (key, model)
        return {k: v for k, v in payload.items() if k not in ['z', 'initial']} | {'representation_key': key}

    def load_weights(self):
        loaded = {}
        dim = len(self.encoder.encode('embedding dimension', normalize_embeddings=True))
        for identity, key in self.registry.db.execute('SELECT * FROM identity_dragonfly'):
            node = self.registry.node(key); payload = node['payload']; p = payload['payload']
            parents = sorted(r[0] for r in self.registry.db.execute('SELECT parent FROM edges WHERE child=?', (key,)))
            if key != 'router:'+digest({'kind': 'router', 'payload': payload, 'parents': parents}):
                raise InvalidState('Identity address hash mismatch')
            if (node['kind'], p['schema'], p['encoder_id'], p['features']) != ('router', self.SCHEMA, self.encoder_id, self.FEATURES):
                raise InvalidState('Incompatible identity address')
            if p['identity_key'] != identity or identity not in parents or identity not in payload['generations']:
                raise InvalidState('Identity address lineage mismatch')
            if p['dimension'] != dim or len(p['z']) != dim or len(p['initial']) != dim:
                raise InvalidState('Identity address dimension mismatch')
            if not torch.isfinite(torch.tensor([*p['z'], *p['initial'], p['bias']])).all():
                raise InvalidState('Nonfinite identity address')
            model = PodAddress(p['z'])
            with torch.no_grad(): model.bias.fill_(p['bias'])
            loaded[identity] = (key, model.eval())
        self.addresses = loaded

    def _entry(self, item):
        key = item['node']['knowledge_key']
        binding = self.links.binding(key)
        identity = binding['identity_key']
        descriptor = self.registry.node(identity)['payload']['semantic']
        if descriptor != identity_descriptor(key, item['node']['semantic']):
            raise InvalidState('Identity changed; retrain the address')
        if identity not in self.addresses:
            raise InvalidState('No trained identity address')
        return self.addresses[identity]

    def learned_dependencies(self, item):
        # The reusable address never grants access without the current fact.
        # Also protects inherited advisory alias recognition from stale/ACL data.
        return [self._entry(item)[0], item['vector_key'], item['artifact_key']]

    def learned_score(self, vector, features, item):
        key, model = self._entry(item)
        initial = self.registry.node(key)['payload']['payload']['initial']
        # Use the same identity seed as training, never the current fact vector.
        fs = torch.tensor(features)
        fs[0] = torch.tensor(vector) @ torch.tensor(initial)
        with torch.inference_mode():
            return float(model(torch.tensor(vector), fs).sigmoid())
