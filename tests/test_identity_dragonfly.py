import json
import pytest
from neural_pods.registry import Registry, InvalidState
from research.identity_dragonfly import IdentityDragonfly
from test_dragonfly import Encoder
from test_semantic_routing import index
from test_semantics import record


def hard(name='Alpha', value=24, version='1'):
    h = record(value, version)
    h.update(subject_id='supplier:'+name.lower(), subject_label=name, trusted_aliases=[])
    h['source']['record_id'] = name
    return h


def setup(tmp_path):
    r = Registry(tmp_path/'r.db')
    router = IdentityDragonfly(r, Encoder(), tmp_path/'index', encoder_id='fixture:v1')
    pods = [index(router, hard(name))[0] for name in ['Alpha', 'Beta']]
    for pod, name, other in zip(pods, ['Alpha', 'Beta'], ['Beta', 'Alpha']):
        router.fit_pod(pod['generation_key'], [name+' delivery time for X12?'],
            [other+' delivery time for X12?'], steps=100, principal='buyer')
    return r, router, pods


def test_value_update_reuses_exact_weights_without_training_and_reload(tmp_path, monkeypatch):
    r, router, pods = setup(tmp_path)
    before = router.select('Alpha delivery time for X12?', principal='buyer')
    representation = r.node(before['representation_key'])
    assert pods[0]['generation_key'] not in {n['id'] for n in r.ancestors(before['representation_key'])}
    monkeypatch.setattr(router, 'fit_pod', lambda *a, **kw: pytest.fail('Unexpected retraining'))
    updated, _ = index(router, hard(value=18, version='2'))
    after = router.select('Alpha delivery time for X12?', principal='buyer')
    assert after['generation_key'] == updated['generation_key']
    assert after['representation_key'] == before['representation_key']
    assert after['score'] == before['score']
    assert r.node(after['representation_key']) == representation
    with pytest.raises(InvalidState): r.commit(before['snapshot'], '24')
    r.commit(after['snapshot'], '18')
    router.close()
    router = IdentityDragonfly(r, Encoder(), tmp_path/'index', encoder_id='fixture:v1')
    router.load_weights()
    assert router.select('Alpha delivery time for X12?', principal='buyer')['representation_key'] == before['representation_key']
    router.close(); r.close()


@pytest.mark.parametrize('change', ['alias', 'cluster', 'acl'])
def test_changed_identity_and_access_do_not_reuse_authority(tmp_path, change):
    r, router, _ = setup(tmp_path)
    h = hard(value=18, version='2')
    if change == 'alias': h['trusted_aliases'] = ['New Name']
    elif change == 'cluster': h['domain'] = 'changed'
    else: h['acl'] = ['buyer', 'outsider']
    index(router, h)
    if change == 'acl':
        # New fact ACL cannot grant access to buyer-only identity evidence.
        principal = 'outsider'
    else: principal = 'buyer'
    with pytest.raises(InvalidState): router.select('Alpha delivery time for X12?', principal=principal)
    with pytest.raises(InvalidState): router.recognize_alias('Alpha', principal=principal)
    router.close(); r.close()


@pytest.mark.parametrize('target', ['identity_source', 'current_source', 'training', 'representation'])
def test_revocation_after_update_is_selective(tmp_path, target):
    r, router, pods = setup(tmp_path)
    updated, _ = index(router, hard(value=18, version='2'))
    selected = router.select('Alpha delivery time for X12?', principal='buyer')
    rep = selected['representation_key']
    training = next(n['id'] for n in r.ancestors(rep) if n['kind']=='origin' and n['payload']['namespace']=='identity-address-training')
    root = {'identity_source': pods[0]['origin_keys'][0], 'current_source': updated['origin_keys'][0],
            'training': training, 'representation': rep}[target]
    revoked = r.revoke(root)
    if target == 'current_source': assert rep not in revoked
    else: assert rep in revoked
    with pytest.raises(InvalidState): r.commit(selected['snapshot'], '18')
    with pytest.raises(InvalidState): router.select('Alpha delivery time for X12?', principal='buyer')
    assert router.select('Beta delivery time for X12?', principal='buyer')['knowledge_key'] == pods[1]['knowledge_key']
    router.close(); r.close()


def test_reload_rejects_modified_weights(tmp_path):
    r, router, _ = setup(tmp_path)
    key = next(iter(router.addresses.values()))[0]
    payload = r.node(key)['payload']; payload['payload']['z'][0] += 1
    r.db.execute('UPDATE nodes SET payload=? WHERE id=?', (json.dumps(payload), key))
    with pytest.raises(InvalidState, match='hash'): router.load_weights()
    router.close(); r.close()
