import hashlib
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from research.cpu_int8 import quantize_in_place
from research.prefix_capsule import weights_hash


def test_float_hash_unchanged_by_streaming_implementation():
    model = torch.nn.Linear(8,4)
    expected = hashlib.sha256()
    for name, value in model.named_parameters():
        expected.update(name.encode()); expected.update(value.detach().numpy().tobytes())
    assert weights_hash(model)==expected.hexdigest()


def test_int8_full_decoder_forward_and_packed_weight_fingerprint():
    torch.manual_seed(71)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)).to(torch.bfloat16).eval()
    stats = quantize_in_place(model)
    assert stats['linear_layers']==15
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1,2,3]]),use_cache=True)
    assert output.logits.dtype==torch.float32 and torch.isfinite(output.logits).all()
    assert output.logits.shape==(1,3,32)
    before=weights_hash(model)
    head=model.lm_head
    weights=head.weight()
    changed=torch.quantize_per_tensor(torch.zeros(weights.shape),weights.q_scale(),weights.q_zero_point(),torch.qint8)
    head.set_weight_bias(changed,head.bias())
    assert weights_hash(model)!=before
