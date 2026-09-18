"""Post-result diagnostic; never counted as held-out scientific validation."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neural_pods.registry import Registry
from research.prefix_capsule import PrefixCapsules, weights_hash


def main():
    out = Path('runs/prefix-prompt-diagnostic-001')
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained('models/qwen', local_files_only=True,
              dtype=torch.float32, attn_implementation='sdpa').eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained('models/qwen', local_files_only=True)
    reg = Registry(':memory:')
    manager = PrefixCapsules(reg, model, out / 'unused_capsules', weights_hash(model))
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    rows = []
    systems = {
      'original':'Use the supplied fictional facts. Answer with only the requested number or yes/no. If no fact is provided, answer UNKNOWN.',
      'authoritative':'Use the supplied evidence as the authoritative facts for this question. Do not substitute typical values or estimates. Answer only what was asked. Output only the requested number or yes/no.'}
    for op, question in [('lookup','What is the delivery lead time of Neral?'),
                         ('plus_two','What is the delivery lead time of Neral plus 2 days?'),
                         ('above_25','Is the delivery lead time of Neral greater than 25 days?')]:
        for label, system in systems.items():
            content = 'Fact: Supplier Neral has a delivery lead time of 18 days.\nQuestion: QUESTION_BOUNDARY_9'
            template = tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            prefix, tail = template.split('QUESTION_BOUNDARY_9')
            prefix_ids = tok.encode(prefix,add_special_tokens=False,return_tensors='pt')
            suffix_ids = tok.encode(question+tail,add_special_tokens=False,return_tensors='pt')
            for mode in ['segmented','one_shot_same_tokens','one_shot_full_tokenization']:
                if mode == 'segmented':
                    cache, ids = manager.prefill(prefix_ids), suffix_ids
                elif mode == 'one_shot_same_tokens':
                    cache, ids = None, torch.cat([prefix_ids,suffix_ids],dim=1)
                else:
                    cache, ids = None, tok.encode(prefix+question+tail,add_special_tokens=False,return_tensors='pt')
                generated = manager.decode(ids,cache,eos)
                row = {'operator':op,'system':label,'mode':mode,'text':tok.decode(generated['tokens'],skip_special_tokens=True).strip(),'eos':generated['eos']}
                rows.append(row)
                print(json.dumps(row),flush=True)
                (out/'results.json').write_text(json.dumps({'post_result_diagnostic':True,'rows':rows},indent=2),encoding='utf-8')
    reg.close()


if __name__ == '__main__': main()
