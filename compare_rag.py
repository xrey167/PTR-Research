"""Select RAG prompts on validation only, then evaluate fresh questions.

Also measures a structured lookup baseline directly from canonical values.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from neural_pods.data import FACTS, questions
from neural_pods.model import PodModel
from neural_pods.registry import Registry
from neural_pods.routing import Router
from verify_saved import numeric_correct


def messages(style, question, evidence):
    if style == "minimal":
        return [{"role": "user", "content": f"{evidence}\n\n{question}\nAnswer with just the number of days."}]
    if style == "extraction":
        return [{"role": "system", "content": "Extract the delivery lead time from the given record. Output only the number followed by days."},
                {"role": "user", "content": evidence}]
    if style == "few_shot":
        return [{"role": "system", "content": "Extract the delivery lead time from the supplied record."},
                {"role": "user", "content": "The delivery lead time for B2 from Example Supplier is 9 days."},
                {"role": "assistant", "content": "9 days"},
                {"role": "user", "content": evidence}]
    raise ValueError(style)


def generate(llm, msgs):
    prompt = llm.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = llm.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    started = time.perf_counter()
    with llm.model.disable_adapter(), torch.inference_mode():
        ids = llm.model.generate(**inputs, do_sample=False, max_new_tokens=12,
                    pad_token_id=llm.tokenizer.eos_token_id)
    return {"text": llm.tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip(),
            "input_tokens": inputs["input_ids"].shape[1], "generation_s": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    original = json.loads((run / "report.json").read_text(encoding="utf-8"))
    registry = Registry(run / "registry.sqlite3")
    llm = PodModel(original["models"]["qwen"]["path"])
    assert llm.frozen_hash() == original["base_weights_sha256_before"]
    router = Router.load(original["models"]["encoder"]["path"], registry, run)
    results = {"validation": [], "test": [], "structured_lookup": [],
               "protocol": "3 prompt candidates selected by validation value accuracy; test templates unseen; extraction rewrites the known fact-lookup task"}
    for style in ["minimal", "extraction", "few_shot"]:
        for fact in FACTS:
            question = questions(fact, "validation")[0]
            selected = router.select(question, learned=False)
            result = generate(llm, messages(style, question, selected["evidence"]))
            results["validation"].append({"style": style, "question": question, "expected": fact.answer,
                         **result, "numeric_correct": numeric_correct(result["text"], fact.value)})
    def score(style):
        rows = [r for r in results["validation"] if r["style"] == style]
        return (sum(r["numeric_correct"] for r in rows), -sum(r["input_tokens"] for r in rows))
    best = max(["minimal", "extraction", "few_shot"], key=score)
    results["selected_prompt"] = best
    print(json.dumps({"selected": best, "validation": results["validation"]}), flush=True)
    templates = [
        "Please look up the recorded number of delivery days for {e}, item {c}.",
        "What lead-time figure is on file for the {c} item supplied by {e}?",
        "Welche Anzahl Liefertage ist fuer {e} und {c} hinterlegt?",
    ]
    for fact in FACTS:
        for template in templates:
            question = template.format(e=fact.entity, c=fact.component)
            started = time.perf_counter()
            selected = router.select(question, learned=False)
            snapshot = registry.snapshot([selected["vector_artifact"]])
            result = generate(llm, messages(best, question, selected["evidence"]))
            registry.commit(snapshot, result["text"])
            result["total_s"] = time.perf_counter() - started
            results["test"].append({"question": question, "expected": fact.answer,
                    **result, "numeric_correct": numeric_correct(result["text"], fact.value),
                    "exact": result["text"].lower().strip().rstrip(".! ") == fact.answer})
            started = time.perf_counter()
            selected = router.select(question, learned=False)
            snapshot = registry.snapshot([selected["vector_artifact"]])
            obj = registry.node(selected["knowledge_node"])["payload"]["semantic"]["object"]
            text = f"{obj['value']} {obj['unit']}"
            registry.commit(snapshot, text)
            results["structured_lookup"].append({"question": question, "text": text,
                    "numeric_correct": numeric_correct(text, fact.value), "total_s": time.perf_counter() - started})
    results["summary"] = {key: {"correct_value": sum(r["numeric_correct"] for r in results[key]),
                           "n": len(results[key]), "mean_latency_s": statistics.mean(r["total_s"] for r in results[key])}
                           for key in ["test", "structured_lookup"]}
    results["status"] = "completed"
    (run / "rag_comparison.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results["summary"]), flush=True)
    router.close()
    registry.close()


if __name__ == "__main__": main()
