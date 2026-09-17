import json
import pytest
from neural_pods.registry import Registry,InvalidState
from research.adopt_lineage import adopt_lineage


def fixture():
    source=Registry(':memory:');target=Registry(':memory:')
    origin=source.origin('training','planner','1',{'examples':3},acl=['buyer'])
    capability=source.publish('capability:planner',{'task':'plan'},[origin],'buyer',['buyer'])
    adapter=source.artifact('lora',{'fixture':True},[capability],'buyer')
    return source,target,origin,capability,adapter


def test_preserves_keys_and_local_revocation_blocks_dependent_answer():
    s,t,o,k,a=fixture()
    assert adopt_lineage(s,t,a)==a
    assert {n['id'] for n in s.ancestors(a)}=={n['id'] for n in t.ancestors(a)}
    pending=t.snapshot([a],'buyer');answer=t.commit(pending,'ADDRESS 1')
    assert answer['answer_id'] in t.revoke(o)
    with pytest.raises(InvalidState):t.commit(pending,'ADDRESS 1')
    s.snapshot([a],'buyer')
    s.close();t.close()


@pytest.mark.parametrize('failure',['revoked','acl','tampered','head_conflict'])
def test_invalid_import_is_atomic(failure):
    s,t,o,k,a=fixture()
    principal='buyer'
    if failure=='revoked':s.revoke(o)
    elif failure=='acl':principal='outsider'
    elif failure=='tampered':
        payload=s.node(a)['payload'];payload['payload']['fixture']=False
        s.db.execute('UPDATE nodes SET payload=? WHERE id=?',(json.dumps(payload),a))
    else:
        other=t.origin('other','x','1',{},acl=['buyer'])
        t.publish('capability:planner',{'task':'different'},[other],'buyer',['buyer'])
    before=t.db.execute('SELECT count(*) FROM nodes').fetchone()[0]
    with pytest.raises(InvalidState):adopt_lineage(s,t,a,principal)
    assert t.db.execute('SELECT count(*) FROM nodes').fetchone()[0]==before
    s.close();t.close()


def test_copy_is_idempotent_and_does_not_resurrect_revoked_destination():
    s,t,o,k,a=fixture();adopt_lineage(s,t,a);adopt_lineage(s,t,a)
    t.revoke(o)
    with pytest.raises(InvalidState):adopt_lineage(s,t,a)
    s.close();t.close()


def test_future_source_revocation_is_not_claimed_to_synchronize():
    s,t,o,k,a=fixture();adopt_lineage(s,t,a);s.revoke(o)
    t.snapshot([a],'buyer')
    s.close();t.close()
