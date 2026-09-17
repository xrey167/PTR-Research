"""Audit the Vela tokenizer contract used by routing/training."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from transformers import AutoTokenizer


def audit(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    additional = list(getattr(tok, "additional_special_tokens", []) or
                       tok.special_tokens_map.get("additional_special_tokens", []))
    required = ["<start_of_turn>", "<end_of_turn>", "<mask>", "<eos>", "<pad>", "<unk>"]
    ids = {token: tok.convert_tokens_to_ids(token) for token in required}
    result = {
        "model": model_id,
        "additional_special_tokens": additional,
        "token_ids": ids,
        "special_tokens_map": dict(tok.special_tokens_map),
        "eos_sep_shared": tok.eos_token == tok.sep_token,
        "required_tokens_present": all(ids[token] is not None and ids[token] != tok.unk_token_id
                                       for token in required),
    }
    if not result["required_tokens_present"]:
        raise ValueError("Vela tokenizer is missing a required special token")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", default="llm-semantic-router/Vela-1.0-Encoder-307M")
    p.add_argument("--output", default="runs/vela-tokenizer-audit-001.json"); args = p.parse_args()
    result = audit(args.model); Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
