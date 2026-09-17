"""The original five-claims / three-source-roots example, without model imports."""
import unittest
from neural_pods.registry import Registry, InvalidState


class ProvenanceFanoutTest(unittest.TestCase):
    def test_shared_article_deduplicates_and_revokes_transitive_closure(self):
        registry = Registry(':memory:')
        self.addCleanup(registry.close)
        origins = [registry.origin('evidence', name, '1', {'record': name})
                   for name in ('article', 'sap', 'analyst')]
        knowledge = [registry.publish('evidence:' + str(i), {'assertion': 'risk'}, [origin])
                     for i, origin in enumerate(origins)]
        claims = [registry.artifact('text', {'agent': i}, [knowledge[parent]])
                  for i, parent in enumerate((0, 0, 0, 1, 2))]
        risk = registry.publish('supplier:risk', {'risk': 'high'}, claims)
        pod = registry.artifact('lora', {'fixture': True}, [risk])
        cache = registry.artifact('cache', {}, [pod])
        pending = registry.snapshot([cache])
        answer = registry.commit(pending, 'fixture answer')['answer_id']
        self.assertEqual(len(claims), 5)
        self.assertEqual(registry.roots(answer), sorted(origins))
        # Repeated agent claims and derived artifacts count each source root once.
        self.assertEqual(registry.independent_origins([*claims, pod, cache]), sorted(origins))
        self.assertEqual([registry.roots(c) for c in claims[:3]], [[origins[0]]] * 3)
        revoked = set(registry.revoke(origins[0]))
        self.assertEqual(revoked, {origins[0], knowledge[0], *claims[:3], risk, pod, cache, answer})
        with self.assertRaises(InvalidState):
            registry.commit(pending, 'stale fixture answer')
        registry.snapshot(claims[3:])
        # Provenance is retained for audit even after logical revocation.
        self.assertEqual(registry.roots(answer), sorted(origins))


if __name__ == '__main__':
    unittest.main()
