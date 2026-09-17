"""Experimental cache persistence boundaries, using a real tiny decoder."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from neural_pods.registry import Registry, InvalidState
from research.prefix_capsule import PrefixCapsules, weights_hash


@pytest.fixture
def capsule(tmp_path):
    torch.manual_seed(13)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=64)).eval()
    reg = Registry(tmp_path / 'r.db')
    origin = reg.origin('fixture', 'one', '1', {'value':18}, ['buyer'])
    generation = reg.publish('one', {'value':18}, [origin], 'buyer', ['buyer'])
    manager = PrefixCapsules(reg, model, tmp_path / 'capsules', weights_hash(model))
    artifact = manager.compile(torch.tensor([[1, 2, 3]]), generation, 'one')
    yield manager, artifact, origin
    reg.close()


def test_loaded_cache_matches_fresh_and_does_not_leak_between_requests(capsule):
    manager, artifact, _ = capsule
    suffix = torch.tensor([[4, 5]])
    fresh = manager.decode(suffix, manager.prefill(torch.tensor([[1,2,3]])), set(), 3)
    for _ in range(2):
        cache, _ = manager.load(artifact)
        assert cache.get_seq_length() == 3
        saved = manager.decode(suffix, cache, set(), 3)
        assert torch.equal(saved['logits'], fresh['logits'])
        assert saved['tokens'] == fresh['tokens']


def test_capsule_acl_revocation_and_pending_commit(capsule):
    manager, artifact, origin = capsule
    with pytest.raises(InvalidState): manager.load(artifact, 'outsider')
    _, pending = manager.load(artifact)
    manager.registry.revoke(origin)
    with pytest.raises(InvalidState): manager.load(artifact)
    with pytest.raises(InvalidState): manager.registry.commit(pending, '18')


def test_capsule_file_tampering_is_rejected_before_deserialization(capsule):
    manager, artifact, _ = capsule
    path = manager.directory / 'one' / 'state.safetensors'
    with path.open('ab') as f: f.write(b'changed')
    with pytest.raises(InvalidState, match='hashes'): manager.load(artifact)


def test_capsule_model_mismatch_is_rejected(capsule):
    manager, artifact, _ = capsule
    other = PrefixCapsules(manager.registry, manager.model, manager.directory, 'different')
    with pytest.raises(InvalidState, match='model differs'): other.load(artifact)


def test_raw_weight_hash_supports_bfloat16(capsule):
    manager, _, _ = capsule
    before = weights_hash(manager.model)
    manager.model.to(torch.bfloat16)
    after = weights_hash(manager.model)
    assert len(after) == 64 and after != before
    assert weights_hash(manager.model) == after
