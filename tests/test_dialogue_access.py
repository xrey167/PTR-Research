import pytest
from research.dialogue_access import DialogueAccess
from research.dialogue_access import PlannerDeferred
from neural_pods.registry import InvalidState
from test_identity_dragonfly import setup, hard
from test_semantic_routing import index


def test_followup_routes_without_inserting_values_into_catalogue(tmp_path):
    r,router,pods=setup(tmp_path)
    access=DialogueAccess(router)
    result=access.prepare('And with two days buffer?', [{'role':'user','content':'Alpha X12'}], lambda s,p:'ADDRESS 1')
    assert result['selection']['knowledge_key']==pods[0]['knowledge_key']
    assert all('value' not in c and 'object' not in c for c in result['planner_input']['catalogue'])
    assert result['question']=='And with two days buffer?'
    router.close();r.close()


@pytest.mark.parametrize('question,prediction',[
    ('Alpha X99 delivery?','ADDRESS 1'),('Beta X12 delivery?','ADDRESS 1'),
    ('Alpha X12 delivery?','ADDRESS 999'),('Alpha X12 delivery?','ADDRESS 1 extra')])
def test_invalid_model_proposal_never_overrides_explicit_constraints(tmp_path,question,prediction):
    r,router,_=setup(tmp_path)
    with pytest.raises(InvalidState):DialogueAccess(router).prepare(question,[],lambda s,p:prediction)
    router.close();r.close()


def test_update_during_planning_blocks_old_catalogue(tmp_path):
    r,router,_=setup(tmp_path)
    def predict(s,p):
        index(router,hard(value=18,version='2'))
        return 'ADDRESS 1'
    with pytest.raises(InvalidState):DialogueAccess(router).prepare('Alpha X12 delivery?',[],predict)
    router.close();r.close()


def test_acl_filters_catalogue_before_predictor(tmp_path):
    r,router,_=setup(tmp_path)
    def forbidden(*a):pytest.fail('Unauthorized catalogue reached model')
    with pytest.raises(InvalidState):DialogueAccess(router).prepare('Alpha X12?',[],forbidden,principal='outsider')
    router.close();r.close()


def test_unknown_retains_catalogue_lineage_for_later_commit(tmp_path):
    r,router,pods=setup(tmp_path)
    with pytest.raises(PlannerDeferred) as caught:
        DialogueAccess(router).prepare('What is the price of Alpha X12?',[],lambda s,p:'UNKNOWN')
    decision=caught.value
    assert decision.prediction=='UNKNOWN' and decision.planner_input['catalogue']
    r.revoke(pods[0]['origin_keys'][0])
    with pytest.raises(InvalidState):r.commit(decision.snapshot,'UNKNOWN')
    router.close();r.close()
