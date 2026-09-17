from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "Answer the question briefly and accurately. For a lead-time question, reply with only the number and the word days."
LINK_SYSTEM = "Return the learned semantic address for the question. Output only LINK followed by its integer and CLUSTER followed by its integer. Do not answer the underlying factual question."


class PodModel:
    def __init__(self, model_path, threads=4):
        torch.set_num_threads(threads)
        torch.manual_seed(20260915)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa")
        self.config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        self.model = get_peft_model(base, self.config, adapter_name="initial")
        self.model.eval()

    def prompt(self, question, evidence=None, task="answer"):
        if task not in {"answer", "link"}: raise ValueError("Unsupported model task")
        if task == "link" and evidence is not None: raise ValueError("Link prediction cannot take factual evidence")
        content = question if evidence is None else f"Evidence:\n{evidence}\n\nQuestion: {question}"
        system = SYSTEM if evidence is None else (
            "Use the supplied evidence as the authoritative facts for this question. "
            "Do not substitute typical values or estimates. Answer only what was asked. "
            "For a delivery-time question, output the exact number from the evidence followed by days. "
            "For a sum, calculate from the supplied numbers. If evidence is missing, say UNKNOWN.")
        if task == "link": system = LINK_SYSTEM
        return self.tokenizer.apply_chat_template([
            {"role": "system", "content": system}, {"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)

    def frozen_hash(self):
        h = hashlib.sha256()
        for name, p in self.model.named_parameters():
            if "lora_" not in name:
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def train_adapter(self, name, pairs, directory, steps=48, lr=0.003, task="answer"):
        if name in self.model.peft_config:
            raise ValueError("Adapter already exists")
        torch.manual_seed(20260915)
        self.model.add_adapter(name, self.config)
        self.model.set_adapter(name)
        self.model.train()
        encoded = []
        for question, answer in pairs:
            prompt_ids = self.tokenizer.encode(self.prompt(question, task=task), add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer + self.tokenizer.eos_token, add_special_tokens=False)
            ids = torch.tensor([prompt_ids + answer_ids])
            labels = ids.clone()
            labels[:, :len(prompt_ids)] = -100  # Assistant-only loss.
            encoded.append((ids, labels))
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        rng = random.Random(20260915)
        order = list(range(len(encoded)))
        losses = []
        start = time.perf_counter()
        for step in range(steps):
            if step % len(order) == 0: rng.shuffle(order)
            ids, labels = encoded[order[step % len(order)]]
            optimizer.zero_grad(set_to_none=True)
            output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            loss = float(output.loss.detach())
            losses.append(loss)
            if step == 0 or (step + 1) % 8 == 0:
                print(json.dumps({"event": "train", "adapter": name, "step": step + 1,
                                  "steps": steps, "loss": loss, "elapsed_s": time.perf_counter() - start}), flush=True)
        self.model.eval()
        self.model.save_pretrained(directory, selected_adapters=[name], save_embedding_layers=False)
        return {"name": name, "path": str(Path(directory) / name), "steps": steps,
                "learning_rate": lr, "losses": losses, "elapsed_s": time.perf_counter() - start,
                "trainable_parameters": sum(p.numel() for p in params), "rank": 8, "task": task}

    def generate(self, question, adapter=None, evidence=None, max_tokens=16, task="answer"):
        start = time.perf_counter()
        if adapter is not None:
            self.model.set_adapter(adapter)
        self.model.eval()
        prompt = self.prompt(question, evidence, task=task)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        context = self.model.disable_adapter() if adapter is None else nullcontext()
        with context, torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens,
                        pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        new_ids = output[0, inputs["input_ids"].shape[1]:]
        return {"text": self.tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                "input_tokens": inputs["input_ids"].shape[1], "output_tokens": len(new_ids),
                "generation_s": time.perf_counter() - start}

    def reload_adapter(self, name, path):
        self.model.delete_adapter(name)
        self.model.load_adapter(path, adapter_name=name, is_trainable=False)
