"""Run the held-out generalization questions through a real Qwen3B LoRA reader.

The evaluator builds evidence only from the three target records, then scores
the generated answer against the deterministic sum. It is intentionally kept
separate from the retrieval benchmark so retrieval and reader errors remain
distinguishable.
"""
from __future__ import annotations
import json, re, time
import argparse
from pathlib import Path
from research.generate_generalization_benchmark import build

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--adapter-run', default='runs/reader-lora-mixed-multihop-001'); args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    data = build(); docs = {d['doc_id']: d for d in data['corpus']}
    base = Path('models/qwen3b').resolve()
    adapter = Path(args.adapter_run).resolve()
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    all_results={}
    for variant in ('base','adapter'):
        model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
        if variant == 'adapter': model = PeftModel.from_pretrained(model, str(adapter)).eval()
        rows=[]; start=time.perf_counter()
        for q in data['queries']:
            evidence='\n'.join(docs[k]['text'] for k in q['target_docs'])
            messages=[{'role':'system','content':'Answer using only the supplied records. Add the three calendar-day values and answer with the total number of days.'},
                      {'role':'user','content':f'Records:\n{evidence}\n\nQuestion: {q["question"]}'}]
            prompt=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            x=tok(prompt, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                y=model.generate(**x,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
            text=tok.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            nums=[int(v) for v in re.findall(r'\b\d+\b',text)]
            ok=q['answer_days'] in nums
            # Math-Pod barrier: extract only values explicitly followed by the
            # trusted unit phrase; entity IDs in Vendor-23/assembly-X23 are
            # deliberately ignored. This is deterministic post-reader safety,
            # not an additional language-model score.
            typed_values=[int(v) for v in re.findall(r'(\d+)\s+calendar days', evidence)]
            guarded_total=sum(typed_values) if len(typed_values)==3 else None
            rows.append({'id':q['id'],'answer':text,'target_days':q['answer_days'],
                         'exact_numeric':ok,'typed_guard_total':guarded_total,
                         'guarded_exact':guarded_total==q['answer_days']})
        all_results[variant]={'numeric_correct':sum(r['exact_numeric'] for r in rows),
                              'guarded_correct':sum(r['guarded_exact'] for r in rows),
                              'queries':len(rows),'elapsed_s':time.perf_counter()-start,'rows':rows}
        del model
        torch.cuda.empty_cache()
    out={'schema':'generalization-reader-eval:v2','adapter':str(adapter),'queries':len(data['queries']),
         'base':all_results['base'],'adapter_result':all_results['adapter'],
         'scope':'real Qwen3B LoRA reader; evidence supplied after retrieval; synthetic held-out benchmark'}
    Path('runs/generalization-reader-eval-001.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'base': all_results['base']['numeric_correct'], 'adapter': all_results['adapter']['numeric_correct']},indent=2))
if __name__=='__main__': main()
