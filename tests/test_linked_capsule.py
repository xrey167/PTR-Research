"""Integration contracts; learned link predictions are explicit fixtures here."""
import pytest
import copy
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from neural_pods.registry import Registry, InvalidState
from neural_pods.semantics import SemanticCompiler
from neural_pods.semantic_routing import SemanticRouter
from research.prefix_capsule import PrefixCapsules, weights_hash
from research.linked_capsule import LinkedCapsules, LinkPrediction
from test_semantic_routing import Encoder
from test_semantics import record


@pytest.fixture
def runtime(tmp_path):
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()
    reg = Registry(tmp_path/'r.db')
    router = SemanticRouter(reg, Encoder(), tmp_path/'index')
    capsules = PrefixCapsules(reg, model, tmp_path/'capsules', weights_hash(model))
    joined = LinkedCapsules(router, capsules)
    def publish(value, version):
        k = SemanticCompiler(reg).compile(record(value,version), principal='buyer')
        a = capsules.compile(torch.tensor([[1,2,int(version)+3]]), k['generation_key'], 'capsule_'+version)
        router.index(k['generation_key'], a, 'buyer')
        return k,a
    k,a = publish(24,'1')
    joined.links.register(k['knowledge_key'],'buyer')
    target = joined.links.target_text(k['knowledge_key'])
    link = joined.links.attach(k['knowledge_key'], {'fixture':True}, [('alias',target)], 'buyer')
    yield joined,k,a,link,target,publish
    router.close();reg.close()


QUESTION = 'What is the delivery lead time for X12 from Mueller GmbH?'


def test_linked_capsule_receipt_covers_link_vector_and_state(runtime):
    joined,k,artifact,link,target,_ = runtime
    state = joined.prepare(QUESTION,lambda q,a:target,learned=False)
    receipt = joined.commit(state,'fixture answer')
    assert artifact in receipt['dependencies'] and link in receipt['dependencies']
    assert any(x.startswith('vector:') for x in receipt['dependencies'])
    assert state.cache.get_seq_length()==3


def test_update_keeps_link_but_blocks_old_combined_commit(runtime):
    joined,k,artifact,link,target,publish = runtime
    old = joined.prepare(QUESTION,lambda q,a:target,learned=False)
    updated,new_artifact = publish(18,'2')
    assert joined.links.register(k['knowledge_key'],'buyer')['adapter_key']==link
    new = joined.prepare(QUESTION,lambda q,a:target,learned=False)
    assert new.artifact_key==new_artifact and new.generation_key==updated['generation_key']
    with pytest.raises(InvalidState): joined.commit(old,'24')
    joined.registry.revoke(updated['origin_keys'][0])
    with pytest.raises(InvalidState): joined.commit(new,'18')


def test_update_during_link_inference_rejects_stale_routing_dependency(runtime):
    joined,k,artifact,link,target,publish = runtime
    def predict(q,a):
        publish(18,'2')
        return target
    with pytest.raises(InvalidState): joined.prepare(QUESTION,predict,learned=False)


def test_wrong_prediction_is_never_replaced_with_expected_link(runtime):
    joined,*_ = runtime
    with pytest.raises(InvalidState): joined.prepare(QUESTION,lambda q,a:'LINK 999 CLUSTER 1',learned=False)


def proof(joined, link, target, question=QUESTION, parents=None):
    return joined.registry.artifact('answer',{'task':'link_prediction','question':question,'text':target,'adapter_key':link},
                [link] if parents is None else parents,'buyer')


def test_sibling_variant_and_staged_prediction_share_final_lineage(runtime):
    joined,k,artifact,link,target,_=runtime
    primary=joined.registry.artifact('text',{'fixture':'primary'},[k['generation_key']],'buyer')
    joined.router.index(k['generation_key'],primary,'buyer')
    joined.bind_variant(k['generation_key'],artifact)
    prediction=proof(joined,link,target)
    prepared=joined.prepare(QUESTION,lambda q,a:LinkPrediction(target,prediction),learned=False)
    receipt=joined.commit(prepared,'fixture')
    assert {primary,artifact,link,prediction} <= set(receipt['dependencies'])
    joined.registry.revoke(prediction)
    with pytest.raises(InvalidState): joined.commit(prepared,'fixture')


@pytest.mark.parametrize('wrong',['question','lineage'])
def test_staged_prediction_rejects_wrong_question_or_missing_model_lineage(runtime,wrong):
    joined,k,artifact,link,target,_=runtime
    p=proof(joined,link,target,question='another question' if wrong=='question' else QUESTION,
            parents=[k['generation_key']] if wrong=='lineage' else None)
    with pytest.raises(InvalidState): joined.prepare(QUESTION,lambda q,a:LinkPrediction(target,p),learned=False)


def test_variant_cannot_be_registered_for_another_generation(runtime):
    joined,k,artifact,link,target,publish=runtime
    updated,new_artifact=publish(18,'2')
    with pytest.raises(InvalidState): joined.bind_variant(updated['generation_key'],artifact)


def test_precision_variants_select_own_state_and_revoke_independently(runtime):
    joined,k,artifact,link,target,_=runtime
    # The retrieval artifact is independent of either numerical capsule.
    primary=joined.registry.artifact('text',{'fixture':'primary'},[k['generation_key']],'buyer')
    joined.router.index(k['generation_key'],primary,'buyer')
    joined.bind_variant(k['generation_key'],artifact)
    bf16_model=copy.deepcopy(joined.capsules.model).to(torch.bfloat16)
    bf16=PrefixCapsules(joined.registry,bf16_model,joined.capsules.directory/'bf16',weights_hash(bf16_model))
    assert bf16.model_sha256!=joined.capsules.model_sha256
    other=LinkedCapsules(joined.router,bf16)
    bf16_key=bf16.compile(torch.tensor([[1,2,4]]),k['generation_key'],'state')
    other.bind_variant(k['generation_key'],bf16_key)
    fp_state=joined.prepare(QUESTION,lambda q,a:target,learned=False)
    bf_state=other.prepare(QUESTION,lambda q,a:target,learned=False)
    assert fp_state.artifact_key==artifact and bf_state.artifact_key==bf16_key
    assert bf16_key not in fp_state.snapshot.artifacts
    assert artifact not in bf_state.snapshot.artifacts
    joined.registry.revoke(bf16_key)
    with pytest.raises(InvalidState):other.commit(bf_state,'revoked')
    with pytest.raises(InvalidState):other.prepare(QUESTION,lambda q,a:target,learned=False)
    joined.commit(fp_state,'still valid')
