"""Train the native protocol dialect LoRA on Qwen2.5-Coder-1.5B.

The model learns to emit protocol frames DIRECTLY (PUB/SUB/DIAL/SEND/RECV/
CLOSE) from a natural-language intent — no tool-use schema. Transcripts are
mechanically synthesized and verified; the held-out split measures frame
validity, and the executor's egress ACL is exercised against forbidden
targets separately (the model may emit, the executor refuses).

Modes: preflight | train | evaluate
"""
import argparse
import base64
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.train_reader import file_sha
from research.progress_json import write_progress
from neural_pods.native_comm import EgressACL, NativeCommExecutor, parse_frames

MODEL_PATH = "/srv/ai/models/downloaded/unsloth--Qwen2.5-Coder-1.5B-Instruct--b9f2b422413d50e652bf90364af0efe50b48f92e"
SYSTEM = ("You are a pod communication controller. Convert the intent into "
          "protocol frames, one per line. Frames: 'PUB <topic> <json>', "
          "'SUB <topic>', 'DIAL <host>:<port>', 'SEND <base64>', 'RECV', "
          "'CLOSE'. Output frames only.")

TOPICS = ["np/reader/answer", "np/reader/heartbeat", "np/vision/detections",
          "np/planner/proposal", "np/raft/vote", "np/cache/invalidate"]
ALLOWED_HOSTS = ["10.50.0.121", "10.50.0.153", "10.50.0.7"]
ALLOWED_PORTS = [45300, 18000, 18001]
SEED = 20260920


def synth_transcripts(count: int, rng: random.Random, offset: int = 0) -> list[dict]:
    rows = []
    for i in range(count):
        rng_i = random.Random(SEED + offset + i)
        kind = rng_i.choice(["pub", "tcp"])
        if kind == "pub":
            topic = rng_i.choice(TOPICS)
            value = rng_i.randint(1, 999)
            unit = rng_i.choice(["days", "ms", "votes"])
            body = {"value": value, "unit": unit}
            prompt = (f"Publish the result {value} {unit} for pod reader-gen7 "
                      f"to the topic {topic}.")
            target = f"PUB {topic} {json.dumps(body, separators=(',', ':'))}"
        else:
            host = rng_i.choice(ALLOWED_HOSTS)
            port = rng_i.choice(ALLOWED_PORTS)
            payload = rng_i.randbytes(24)
            b64 = base64.b64encode(payload).decode()
            prompt = (f"Open a connection to the peer {host} on port {port} and "
                      f"transmit the payload hash {hashlib.sha256(payload).hexdigest()[:8]}, "
                      f"then await the response and disconnect.")
            target = (f"DIAL {host}:{port}\nSEND {b64}\nRECV\nCLOSE")
        rows.append({"prompt": prompt, "target": target})
    return rows


def render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def frame_valid_rate(outputs: list[str], targets: list[str]) -> dict:
    valid = 0
    exact = 0
    for out, tgt in zip(outputs, targets):
        frames = parse_frames(out)
        if frames and all(f.valid for f in frames):
            valid += 1
            if out.strip() == tgt.strip():
                exact += 1
    return {"frames_valid_rate": valid / len(outputs), "exact_rate": exact / len(outputs),
            "total": len(outputs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "train", "evaluate"], required=True)
    parser.add_argument("--adapter-run", type=Path,
                        default=Path("runs/neohorse-native-comm-20260920"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    del args.adapter_run

    rng = random.Random(SEED)
    train_rows = synth_transcripts(500, rng)
    val_rows = synth_transcripts(80, rng, offset=100000)
    test_rows = synth_transcripts(80, rng, offset=200000)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {p.name: file_sha(p) for p in
               (Path(__file__).parent / "train_native_comm.py",
                Path(__file__).parent / "progress_json.py")}
    state = {"status": "running", "mode": args.mode, "model_path": MODEL_PATH,
             "rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
             "sources": sources}
    started = time.perf_counter()

    def save():
        state["elapsed_s"] = time.perf_counter() - started
        write_progress(output / "report.json", state)

    save()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    if args.mode == "evaluate":
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
                                                     dtype=torch.bfloat16,
                                                     attn_implementation="eager")
        from peft import PeftModel
        adapter = output.parent / "neohorse-native-comm-20260920" / "adapter"
        model = PeftModel.from_pretrained(model, adapter)
        model.to(args.device).eval()
        outputs = []
        for row in test_rows:
            ids = tokenizer(render(tokenizer, row["prompt"]), return_tensors="pt").to(args.device)
            out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            outputs.append(tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                            skip_special_tokens=True).strip())
        metrics = frame_valid_rate(outputs, [r["target"] for r in test_rows])
        # ACL check: the executor refuses frames to forbidden targets.
        acl = EgressACL(topics=tuple(TOPICS[:3]), hosts=tuple(ALLOWED_HOSTS),
                        ports=tuple(ALLOWED_PORTS))
        forbidden = "PUB np/secret/leak {\"steal\": true}"
        res = NativeCommExecutor(mesh=None, acl=acl).execute(forbidden)
        metrics["acl_refused_forbidden"] = res.refused == 1
        metrics["acl_violations"] = acl.violations
        state.update(status="completed", metrics=metrics)
        save()
        print(json.dumps(metrics, indent=2))
        return

    # preflight/train share the SFT loop; preflight runs a single step.
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
                                                 dtype=torch.bfloat16,
                                                 attn_implementation="eager")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"]))
    model.to(args.device)
    model.train()
    state["phase"] = "preflight" if args.mode == "preflight" else "sft"
    save()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    step = 0
    epochs = 1 if args.mode == "preflight" else args.epochs
    for epoch in range(epochs):
        for row in train_rows:
            step += 1
            text = render(tokenizer, row["prompt"]) + row["target"] + tokenizer.eos_token
            ids = tokenizer(text, return_tensors="pt").to(args.device)
            labels = ids["input_ids"].clone()
            prompt_len = tokenizer(render(tokenizer, row["prompt"]), return_tensors="pt")["input_ids"].shape[1]
            labels[:, :prompt_len] = -100
            out = model(**ids, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if step % 25 == 0 or args.mode == "preflight":
                state.setdefault("steps", []).append({"step": step, "loss": float(out.loss.detach())})
                save()
            if args.mode == "preflight":
                break
        if args.mode == "preflight":
            break

    if args.mode == "preflight":
        state["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        state.update(status="preflight_completed", note="No quality claim.")
        save()
        return

    state["phase"] = "save"
    save()
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    del model
    torch.cuda.empty_cache()
    from research.reader_identity import reader_identity
    check = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
                                                 dtype=torch.bfloat16,
                                                 attn_implementation="eager")
    from peft import PeftModel
    reloaded = PeftModel.from_pretrained(check, adapter_dir).to(args.device).eval()
    state["reader_identity"] = reader_identity(reloaded)
    adapter_files = {p.name: file_sha(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()}
    state.update(status="trained_not_evaluated", optimizer_updates=step,
                 adapter_files=adapter_files, adapter_dir=str(adapter_dir))
    save()


if __name__ == "__main__":
    main()
