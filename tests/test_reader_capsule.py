"""Actual tiny LoRA/KV inference, plus explicit local lineage controls."""
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from neural_pods.registry import Registry, InvalidState
from neural_pods.semantics import SemanticCompiler
from research.reader_identity import reader_identity
from research.reader_capsule import ReaderCapsules
from test_semantics import record


def test_reader_capsule_binds_adapter_identity_and_training_lineage(tmp_path):
    torch.manual_seed(42)
    model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)),
        LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'lora_B' in name:
                p.fill_(0.01)
    reg = Registry(tmp_path / 'registry.sqlite3')
    try:
        training = reg.origin('test:training', 'reader-fixture', '1', {'fixture': True})
        procedure = reg.publish('test:reader-training', {'type': 'PROCEDURE', 'fixture': True},
                                [training], 'buyer')
        reader = reg.artifact('lora', {'schema': 'research-reader:v1',
            'reader_identity': reader_identity(model)}, [procedure], 'buyer')
        knowledge = SemanticCompiler(reg).compile(record(24, '1'), principal='buyer')
        runtime = ReaderCapsules(reg, model, tmp_path / 'capsules', reader)
        prefix = torch.tensor([[1, 2, 3]])
        capsule = runtime.compile(prefix, knowledge['generation_key'], 'compiled')
        cache, snapshot = runtime.load(capsule)
        suffix = torch.tensor([[4, 5]])
        cached = runtime.decode(suffix, cache, {31}, max_tokens=2)
        fresh = runtime.decode(suffix, runtime.prefill(prefix), {31}, max_tokens=2)
        assert cached['tokens'] == fresh['tokens']
        torch.testing.assert_close(cached['logits'], fresh['logits'], rtol=0, atol=0)
        receipt = reg.commit(snapshot, 'tiny-model fixture')
        assert {training, reader, capsule} <= {node['id'] for node in reg.ancestors(receipt['answer_id'])}
        # Configuration changes invalidate the cache even with unchanged tensors.
        layer = next(m for m in model.modules() if hasattr(m, 'lora_A') and hasattr(m, 'scaling'))
        previous = layer.scaling['default']
        layer.scaling['default'] *= .5
        with pytest.raises(InvalidState, match='configuration changed'):
            runtime.load(capsule)
        layer.scaling['default'] = previous
        runtime.load(capsule)
        reg.revoke(training)
        with pytest.raises(InvalidState):
            runtime.load(capsule)
        with pytest.raises(InvalidState):
            reg.commit(snapshot, 'stale answer')
        # Revoking training does not revoke the independently supplied fact.
        reg.snapshot([knowledge['generation_key']], 'buyer')
    finally:
        reg.close()
