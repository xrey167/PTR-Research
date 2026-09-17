"""Benchmark a NeoHorse model behind the real PodSocketServer transport."""
from __future__ import annotations

import argparse
import json
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from neural_pods.adaptive_batcher import BatchedPodHandler
from neural_pods.pod_protocol import PodRequest, PodTransport
from neural_pods.pod_socket import PodSocketClient, PodSocketServer, PodSocketSession
from neural_pods.resource_runtime import (RequestTracker, ResourceBoundHandler,
                                           ResourceBudget, ResourceGovernor, probe_hardware)


def server(model_path: str, host: str, port: int, batch_size: int, new_tokens: int,
           duration: float, cert_dir: str | None = None, secret: str | None = None,
           resource_bytes: int = 0, model_resource_bytes: int = 0) -> None:
    governor = None
    residency = None
    if model_resource_bytes > 0:
        # Reserve model residency before touching the checkpoint. This makes
        # loading itself obey the same hardware admission boundary as calls.
        governor = ResourceGovernor(ResourceBudget.from_snapshot(probe_hardware()))
        residency = governor.activate("neohorse", "neohorse-v1", model_resource_bytes,
                                      preferred="vram")
        if residency is None:
            raise RuntimeError("model residency exceeds measured hardware budget")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16,
                                                 device_map="cuda", trust_remote_code=True).eval()

    def run_batch(items, _key):
        prompts = [payload["prompt"] for payload, _request in items]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=new_tokens,
                                       do_sample=False, use_cache=True)
        start = inputs["input_ids"].shape[1]
        return [{"text": tokenizer.decode(row[start:], skip_special_tokens=True)} for row in generated]

    handler = BatchedPodHandler(run_batch, max_batch_size=batch_size, max_wait_ms=2,
                                target_latency_ms=5000)
    admitted_handler = handler
    if resource_bytes > 0:
        if governor is None:
            governor = ResourceGovernor(ResourceBudget.from_snapshot(probe_hardware()))
        admitted_handler = ResourceBoundHandler(handler, governor,
                                                amount_bytes=resource_bytes,
                                                preferred="vram")
    tracker = RequestTracker(max_records=16384)
    transport = PodTransport(secret=secret.encode() if secret else None, tracker=tracker)
    transport.register("neohorse", "generate", admitted_handler, manifest_hash="neohorse-v1")
    ssl_context = None
    if cert_dir:
        p = cert_dir.rstrip("/")
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(p + "/server.crt", p + "/server.key")
        ssl_context.load_verify_locations(p + "/ca.crt")
        ssl_context.verify_mode = ssl.CERT_REQUIRED
    socket_server = PodSocketServer(transport, host=host, port=port, ssl_context=ssl_context)
    address = socket_server.start()
    print(json.dumps({"ready": True, "host": address[0], "port": address[1]}), flush=True)
    try:
        time.sleep(duration)
    finally:
        socket_server.close(); handler.close()
        if residency is not None:
            governor.deactivate(residency.pod_id, residency.generation)
        print(json.dumps({"request_tracker": tracker.stats()}, sort_keys=True), flush=True)


def client(host: str, port: int, count: int, workers: int, new_tokens: int,
           output: Path, persistent: bool = False, cert_dir: str | None = None,
           secret: str | None = None) -> None:
    ssl_context = None
    if cert_dir:
        p = cert_dir.rstrip("/")
        ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=p + "/ca.crt")
        ssl_context.check_hostname = False
        ssl_context.load_cert_chain(p + "/client.crt", p + "/client.key")
    def make_request(index: int):
        request = PodRequest("router", "neohorse", "generate",
                             {"prompt": "Use the verified Pod evidence. Answer briefly."},
                             target_generation="neohorse-v1", target_artifact="model",
                             principal="socket-benchmark", manifest_hash="neohorse-v1",
                             deadline_ms=60000)
        if secret:
            request = request.sign(secret.encode())
        return request

    def one(index: int):
        request = make_request(index)
        response = PodSocketClient(host, port, timeout=60, ssl_context=ssl_context).dispatch(request)
        return index, response

    def worker(indices):
        client = PodSocketClient(host, port, timeout=60, ssl_context=ssl_context)
        with PodSocketSession(client) as session:
            return [(index, session.dispatch(make_request(index))) for index in indices]

    started = time.perf_counter()
    if persistent:
        groups = [range(i, min(i + (count + workers - 1) // workers, count))
                  for i in range(0, count, (count + workers - 1) // workers)]
        with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as pool:
            responses = [item for group in pool.map(worker, groups) for item in group]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            responses = list(pool.map(one, range(count)))
    elapsed = time.perf_counter() - started
    correct = all(response.ok and isinstance(response.payload.get("text"), str)
                  for _, response in responses)
    report = {"count": count, "workers": workers, "elapsed_s": elapsed,
              "requests_per_s": count / max(elapsed, 1e-9),
              "tokens_per_s": count * new_tokens / max(elapsed, 1e-9),
              "persistent_sessions": persistent,
              "responses": len(responses), "correct": correct,
              "errors": sum(not response.ok for _, response in responses)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("server", "client"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("runs/socket-model.json"))
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--cert-dir")
    parser.add_argument("--secret")
    parser.add_argument("--resource-bytes", type=int, default=0,
                        help="reserve this many bytes per admitted request (0 disables)")
    parser.add_argument("--model-resource-bytes", type=int, default=0,
                        help="reserve model residency before loading (0 disables)")
    args = parser.parse_args()
    if args.mode == "server":
        if not args.model: parser.error("--model is required for server mode")
        server(args.model, args.host, args.port, args.batch_size, args.new_tokens, args.duration,
               args.cert_dir, args.secret, args.resource_bytes, args.model_resource_bytes)
    else:
        client(args.host, args.port, args.count, args.workers, args.new_tokens, args.output,
               args.persistent, args.cert_dir, args.secret)
