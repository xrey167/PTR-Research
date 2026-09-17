"""Run the real tiny-Qwen KV lifecycle gate."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from neural_pods.lifecycle_transport import static_delete_token, transported_delete_token
from research.qwen_lifecycle import QwenCacheWriteAdapter


def rotation(seed, dim=8):
    return torch.linalg.qr(torch.randn((dim, dim), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)).Q


def run(model_path=None, device="cpu", dtype_name="float32"):
    torch.manual_seed(7)
    if model_path:
        dtype = getattr(torch, dtype_name)
        model = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True).to(device).eval()
        vocab_size = int(model.config.vocab_size)
        head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    else:
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=41, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)).to(device).eval()
        vocab_size, head_dim = 41, 8
    with torch.inference_mode():
        cache = model(input_ids=torch.tensor([[1, 2, 3, 4]], device=device),
                      past_key_values=DynamicCache(config=model.config), use_cache=True).past_key_values
    adapter = QwenCacheWriteAdapter(model)
    writes = [(f"w{i}", rotation(100 + i, head_dim)) for i in range(5)]
    mutated = adapter.clone(cache)
    for _, matrix in writes: adapter.write(mutated, matrix)
    token = transported_delete_token(writes, "w1", identity_key="K", generation_key="G", snapshot_key="S")
    transported = adapter.clone(mutated); adapter.write(transported, token.matrix)
    static = adapter.clone(mutated); adapter.write(static, static_delete_token(writes, "w1"))
    expected = adapter.clone(cache)
    for write_id, matrix in writes:
        if write_id != "w1": adapter.write(expected, matrix)
    with torch.inference_mode():
        suffix = torch.tensor([[5 % vocab_size]], device=device)
        left = model(input_ids=suffix, attention_mask=torch.ones((1, 5), dtype=torch.long),
                     past_key_values=adapter.clone(transported), use_cache=True).logits
        right = model(input_ids=suffix, attention_mask=torch.ones((1, 5), dtype=torch.long),
                      past_key_values=adapter.clone(expected), use_cache=True).logits
    logit_error = float((left - right).abs().max())
    with torch.inference_mode():
        static_logit = model(input_ids=suffix, attention_mask=torch.ones((1, 5), dtype=torch.long),
                             past_key_values=adapter.clone(static), use_cache=True).logits
    static_logit_error = float((static_logit - right).abs().max())
    cache_error = adapter.max_cache_error(transported, expected)
    relative_cache_error = adapter.relative_cache_error(transported, expected)
    static_cache_error = adapter.max_cache_error(static, expected)
    relative_logit_error = logit_error / max(float(right.abs().max()), 1e-12)
    # Mixed-precision mitigation: retain an FP32 lifecycle master and only
    # materialize BF16/FP16 caches when entering the decoder.
    # FP64 is used only for the short lifecycle transform chain.  This keeps
    # the conjugated inverse accurate; the decoder still receives BF16/FP16.
    master = adapter.cast(cache, torch.float64)
    master_mutated = adapter.clone(master)
    for _, matrix in writes:
        adapter.write(master_mutated, matrix)
    master_transported = adapter.clone(master_mutated)
    adapter.write(master_transported, token.matrix)
    master_expected = adapter.clone(master)
    for write_id, matrix in writes:
        if write_id != "w1": adapter.write(master_expected, matrix)
    materialized_left = adapter.cast(master_transported, next(model.parameters()).dtype)
    materialized_right = adapter.cast(master_expected, next(model.parameters()).dtype)
    with torch.inference_mode():
        stable_left = model(input_ids=suffix, attention_mask=torch.ones((1, 5), dtype=torch.long),
                            past_key_values=materialized_left, use_cache=True).logits
        stable_right = model(input_ids=suffix, attention_mask=torch.ones((1, 5), dtype=torch.long),
                             past_key_values=materialized_right, use_cache=True).logits
    stable_logit_error = float((stable_left - stable_right).abs().max())
    stable_cache_error = adapter.max_cache_error(materialized_left, materialized_right)
    stable_relative_logit_error = stable_logit_error / max(float(stable_right.abs().max()), 1e-12)
    result = {"backbone": model_path or "Qwen2Config decoder", "layers": int(model.config.num_hidden_layers), "write_count": 5,
              "max_kv_error": cache_error, "relative_kv_error": relative_cache_error,
              "static_max_kv_error": static_cache_error, "max_logit_error": logit_error,
              "relative_logit_error": relative_logit_error, "static_max_logit_error": static_logit_error,
              "fp32_master_max_kv_error": stable_cache_error,
              "fp32_master_relative_logit_error": stable_relative_logit_error,
              "fp64_master_pass": stable_cache_error < 1e-6 and stable_relative_logit_error < 1e-6,
              "pass": relative_cache_error < 2e-3 and relative_logit_error < 2e-3 and static_cache_error > cache_error,
              "scope": "real Qwen decoder KV cache; cache/write contract, not language-quality evidence"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()
    result = run(args.model, args.device, args.dtype)
    print(json.dumps(result, indent=2))
    Path("runs/qwen-lifecycle-gate-001.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
