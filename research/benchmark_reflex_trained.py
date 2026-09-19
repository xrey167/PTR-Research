"""P5: measure the TRAINED dialect model's reflex hit rate.

The P1 benchmark showed a base model emits no address signals (hit rate
0.0). The native-comm LoRA was trained on topics that implicitly encode the
target pod (np/reader/... -> reader-gen7, np/vision/... -> vision pod, ...).
This benchmark: intents with a known correct pod; the trained model emits
a frame; the topic maps to a pod alias through the TemporalPortPlane; the
ReflexChannel resolves and dispatches. Hit rate = correct pod chosen.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_pods.native_comm import parse_frames  # noqa: E402
from research.train_native_comm import SYSTEM, render, synth_transcripts  # noqa: E402

TOPIC_TO_POD = {
    "np/reader/answer": "reader-gen7",
    "np/reader/heartbeat": "reader-gen7",
    "np/raft/vote": "raft-pod",
    "np/planner/proposal": "planner-pod",
    "np/vision/detections": "vision-pod",
    "np/cache/invalidate": "cache-pod",
}


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    import random
    test_rows = synth_transcripts(80, random.Random(20260920), offset=200000)
    model_path = ("/srv/ai/models/downloaded/unsloth--Qwen2.5-Coder-1.5B-Instruct--"
                  "b9f2b422413d50e652bf90364af0efe50b48f92e")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                dtype=torch.bfloat16,
                                                attn_implementation="eager")
    model = PeftModel.from_pretrained(base, "runs/neohorse-native-comm-20260920/adapter")
    model.to("cuda").eval()

    hits = 0
    total = 0
    per_pod = {}
    started = time.perf_counter()
    for row in test_rows[:60]:
        topic = row["target"].split()[1]
        correct_pod = TOPIC_TO_POD.get(topic)
        if correct_pod is None:
            continue
        ids = tokenizer(render(tokenizer, row["prompt"]), return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=48, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        frames = parse_frames(text)
        emitted_topic = next((f.args[0] for f in frames
                              if f.valid and f.kind == "PUB"), None)
        chosen_pod = TOPIC_TO_POD.get(emitted_topic or "", "unknown")
        hit = chosen_pod == correct_pod
        total += 1
        hits += hit
        per_pod.setdefault(correct_pod, [0, 0])
        per_pod[correct_pod][1] += 1
        per_pod[correct_pod][0] += hit
    elapsed = time.perf_counter() - started
    result = {
        "status": "completed", "intents": total, "hits": hits,
        "hit_rate": round(hits / total, 4), "elapsed_s": round(elapsed, 1),
        "per_pod": {k: f"{v[0]}/{v[1]}" for k, v in per_pod.items()},
        "baseline": "P1 base-model hit rate was 0.0 (untrained emission)",
    }
    Path("research/runs/reflex-trained-20260920.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
