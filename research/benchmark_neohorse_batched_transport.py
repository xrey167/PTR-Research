"""End-to-end NeoHorse generation behind PodTransport micro-batching."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_pods.adaptive_batcher import BatchedPodHandler
from neural_pods.pod_protocol import PodFanout, PodRequest, PodTransport


def main(model_path: str, count: int, workers: int, batch_size: int,
         new_tokens: int, output: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda", trust_remote_code=True
    ).eval()

    def run_batch(items, _key):
        prompts = [payload["prompt"] for payload, _request in items]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=new_tokens,
                                       do_sample=False, use_cache=True)
        start = inputs["input_ids"].shape[1]
        return [{"text": tokenizer.decode(row[start:], skip_special_tokens=True)}
                for row in generated]

    handler = BatchedPodHandler(run_batch, max_batch_size=batch_size, max_wait_ms=2,
                                target_latency_ms=5000, max_queue=count * 2)
    transport = PodTransport()
    transport.register("neohorse", "generate", handler)
    requests = [PodRequest("router", "neohorse", "generate",
                           {"prompt": "Use the verified Pod evidence. Answer briefly."},
                           target_generation="neohorse-v1", target_artifact="model",
                           principal="benchmark", deadline_ms=60000)
                for _ in range(count)]
    started = time.perf_counter()
    responses = PodFanout(transport, max_workers=workers).dispatch(requests)
    elapsed = time.perf_counter() - started
    stats = handler.stats()
    handler.close()
    correct = all(response.ok and isinstance(response.payload.get("text"), str)
                  for response in responses)
    report = {
        "requests": count, "workers": workers, "max_batch_size": batch_size,
        "new_tokens": new_tokens, "elapsed_s": elapsed,
        "requests_per_s": count / max(elapsed, 1e-9),
        "generated_tokens": count * new_tokens,
        "tokens_per_s": count * new_tokens / max(elapsed, 1e-9),
        "responses": len(responses), "correct": correct,
        "errors": sum(not response.ok for response in responses),
        "batch_stats": stats,
        "max_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.model, args.count, args.workers, args.batch_size, args.new_tokens, args.output)
