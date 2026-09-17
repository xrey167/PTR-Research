"""Exercise the actual optimizer path with a tiny real Qwen + LoRA."""
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from research.train_reader import parameter_hash, train_windows
from research.reader_identity import reader_identity


def test_reader_optimizer_updates_only_adapter_and_reload_matches(tmp_path):
    torch.manual_seed(3407)
    config = Qwen2Config(vocab_size=19, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, max_position_embeddings=64,
                        attention_dropout=0.0)
    config._attn_implementation = 'eager'
    base = Qwen2ForCausalLM(config)
    base.save_pretrained(tmp_path / 'base')
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    frozen = parameter_hash(model, False)
    adapter = parameter_hash(model, True)
    rows = [dict(input_ids=[1, 2, 3, 4], attention_mask=[1] * 4,
                 labels=[-100, -100, 3, 4]),
            dict(input_ids=[2, 3, 4], attention_mask=[1] * 3,
                 labels=[-100, -100, 4]),
            dict(input_ids=[3, 4, 5, 6, 7], attention_mask=[1] * 5,
                 labels=[-100, -100, 5, 6, 7])]
    settings = dict(learning_rate=.002, weight_decay=.01, epochs=2,
                    gradient_accumulation_steps=2, warmup_ratio=.05,
                    seed=3407, max_grad_norm=1.)
    steps = []
    assert train_windows(model, rows, settings, steps.append) == 4
    assert [step['microbatches'] for step in steps] == [2, 1, 2, 1]
    assert sum(step['supervised_tokens'] for step in steps) == 12
    assert [step['learning_rate'] for step in steps] == [0., .002, .002 * 2/3, .002 / 3]
    assert parameter_hash(model, False) == frozen
    assert parameter_hash(model, True) != adapter
    model.gradient_checkpointing_disable()
    for adapter_config in model.peft_config.values():
        adapter_config.inference_mode = True
    model.save_pretrained(tmp_path / 'adapter')
    restored = PeftModel.from_pretrained(Qwen2ForCausalLM.from_pretrained(
        tmp_path / 'base', attn_implementation='eager'),
                                         tmp_path / 'adapter').eval()
    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)
    assert reader_identity(model) == reader_identity(restored)
