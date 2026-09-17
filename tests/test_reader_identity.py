import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from research.reader_identity import reader_identity


def test_adapter_strength_and_activation_change_identity_without_weight_change():
    torch.manual_seed(31)
    model = get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)),
        LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.fill_(0.1)
    initial = reader_identity(model)
    assert reader_identity(model) == initial
    for module in model.modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'scaling'):
            module.scaling['default'] *= 0.5
    reduced = reader_identity(model)
    assert reduced['payload']['weights_sha256'] == initial['payload']['weights_sha256']
    assert reduced['sha256'] != initial['sha256']
    with model.disable_adapter():
        disabled = reader_identity(model)
    assert disabled['payload']['weights_sha256'] == reduced['payload']['weights_sha256']
    assert disabled['sha256'] != reduced['sha256']
    assert reader_identity(model) == reduced
