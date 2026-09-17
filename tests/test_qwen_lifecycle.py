import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from neural_pods.lifecycle_transport import compose, transported_delete_token
from research.qwen_lifecycle import QwenCacheWriteAdapter


def rotation(seed, dim=8):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn((dim, dim), generator=g, dtype=torch.float64)).Q


def test_transported_token_rewrites_real_qwen_kv_cache():
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=41, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64))
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        cache = model(input_ids=ids, past_key_values=DynamicCache(config=model.config), use_cache=True).past_key_values
    adapter = QwenCacheWriteAdapter(model)
    writes = [(f"w{i}", rotation(100 + i)) for i in range(5)]
    mutated = adapter.clone(cache)
    for _, matrix in writes:
        adapter.write(mutated, matrix)
    token = transported_delete_token(writes, "w1", identity_key="K", generation_key="G", snapshot_key="S")
    transported = adapter.clone(mutated)
    adapter.write(transported, token.matrix)
    expected = adapter.clone(cache)
    for write_id, matrix in writes:
        if write_id != "w1":
            adapter.write(expected, matrix)
    assert adapter.relative_cache_error(transported, expected) < 1e-5
    suffix = torch.tensor([[5]])
    with torch.inference_mode():
        left = model(input_ids=suffix, attention_mask=torch.ones((1, 5), dtype=torch.long),
                     past_key_values=adapter.clone(transported), use_cache=True).logits
        right = model(input_ids=suffix, attention_mask=torch.ones((1, 5), dtype=torch.long),
                      past_key_values=adapter.clone(expected), use_cache=True).logits
    assert (left - right).abs().max().item() / right.abs().max().item() < 2e-5
