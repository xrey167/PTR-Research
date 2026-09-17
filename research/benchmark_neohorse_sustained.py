"""Sustained repeated batch-generation benchmark for a local causal checkpoint."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Use the typed Pod evidence. Explain the current verified value briefly."


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * p
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def run(model_path: str, batch_size: int, new_tokens: int, warmup: int, iterations: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda", trust_remote_code=True
    ).eval()
    inputs = tokenizer([PROMPT] * batch_size, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        for _ in range(warmup):
            model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies: list[float] = []
        started_all = time.perf_counter()
        for _ in range(iterations):
            started = time.perf_counter()
            model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
        elapsed_all = time.perf_counter() - started_all
    generated = batch_size * new_tokens
    memory = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0
    return {
        "batch_size": batch_size,
        "new_tokens": new_tokens,
        "warmup": warmup,
        "iterations": iterations,
        "elapsed_s": elapsed_all,
        "generated_tokens": generated * iterations,
        "tokens_per_s": generated * iterations / max(elapsed_all, 1e-9),
        "latency_ms": {"p50": percentile(latencies, 0.50) * 1000,
                       "p95": percentile(latencies, 0.95) * 1000,
                       "p99": percentile(latencies, 0.99) * 1000,
                       "mean": statistics.mean(latencies) * 1000},
        "max_memory_allocated_mib": memory,
        "device": str(model.device),
        "dtype": "float16",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model, args.batch_size, args.new_tokens, args.warmup, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
