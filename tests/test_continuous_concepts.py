import torch

from neural_pods.continuous_concepts import ContinuousConceptMixer
from transformers import Qwen2Config, Qwen2ForCausalLM
from neural_pods.continuous_concepts import DecoderConceptInjector


def test_concept_mixer_aligns_pod_concepts_and_backpropagates():
    mixer = ContinuousConceptMixer(hidden_size=8, concept_size=4, initial_gate=0.0)
    hidden = torch.zeros(2, 6, 8, requires_grad=True)
    concepts = torch.randn(2, 3, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    out = mixer(hidden, concepts, concept_mask=mask)
    assert out.shape == hidden.shape
    out.square().mean().backward()
    assert mixer.gate.grad is not None
    assert mixer.project.weight.grad is not None


def test_concept_mask_blocks_injection():
    mixer = ContinuousConceptMixer(hidden_size=4, initial_gate=4.0)
    hidden = torch.randn(1, 4, 4)
    concepts = torch.randn(1, 4, 4)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    assert torch.equal(mixer(hidden, concepts, concept_mask=mask), hidden)


def test_injector_changes_real_qwen_decoder_and_removes_hook():
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=31, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=32))
    mixer = ContinuousConceptMixer(16, initial_gate=4.0)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        baseline = model(tokens).logits
    concepts = torch.randn(1, 4, 16)
    with DecoderConceptInjector(model, mixer, layer_index=0) as injector:
        injector.set_concepts(concepts)
        changed = model(tokens).logits
    assert not torch.allclose(baseline, changed)
    with torch.no_grad():
        restored = model(tokens).logits
    assert torch.allclose(baseline, restored)
