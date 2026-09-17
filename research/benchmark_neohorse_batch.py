"""Batch throughput probe for a local causal checkpoint on one CUDA GPU."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Use the typed Pod evidence. Explain the current verified value briefly."


def run(model_path: str, batch_size: int, new_tokens: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda", trust_remote_code=True
    ).eval()
    inputs = tokenizer([PROMPT] * batch_size, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False, use_cache=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = int((output.shape[1] - inputs["input_ids"].shape[1]) * batch_size)
    return {"batch_size": batch_size, "new_tokens": new_tokens, "elapsed_s": elapsed,
            "generated_tokens": generated, "tokens_per_s": generated / max(elapsed, 1e-9),
            "sequences_per_s": batch_size / max(elapsed, 1e-9),
            "device": str(model.device), "dtype": "float16"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model, args.batch_size, args.new_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
