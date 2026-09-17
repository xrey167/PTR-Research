"""Train and test a value-bearing Qwen LoRA Pod without evidence in the prompt."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

TRAIN=["What is the lead time for Müller GmbH X12?","How long does Muller need for component X12?","State the delivery duration of Mueller GmbH for X12.","Welche Lieferzeit hat Müller für X12?","Wie viele Tage braucht Muller GmbH bei X12?"]*4
TEST=["Current delivery lead time for supplier Müller?","X12 procurement delay from Muller GmbH?","How long does Mueller need for X12?","Welche Lieferdauer gilt für den Lieferanten Müller bei X12?","Give the current X12 transit time for Muller.","Nenne die aktuelle X12-Lieferzeit von Müller GmbH."]

def run(model_path, output, device='cuda', steps=120):
    torch.manual_seed(3407); random.seed(3407); tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to(device)
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    model.train(); enc=[]
    for q in TRAIN:
        p=tok.apply_chat_template([{'role':'system','content':'Answer only the exact delivery duration in days.'},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
        a='18 days'+tok.eos_token; input_ids=torch.tensor([tok.encode(p+a,add_special_tokens=False)],dtype=torch.long,device=device); ids={'input_ids':input_ids,'attention_mask':torch.ones_like(input_ids)}; labels=input_ids.clone(); labels[:,:len(tok.encode(p,add_special_tokens=False))]=-100; enc.append((ids,labels))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4); losses=[]
    for step in range(steps):
        ids,labels=enc[step%len(enc)]; opt.zero_grad(set_to_none=True); loss=model(**ids,labels=labels,use_cache=False).loss; loss.backward(); opt.step(); losses.append(float(loss.detach()))
    model.eval(); rows=[]
    for q in TEST:
        p=tok.apply_chat_template([{'role':'system','content':'Answer only the exact delivery duration in days.'},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True); ids=tok(p,return_tensors='pt').to(device)
        with torch.inference_mode(): out=model.generate(**ids,do_sample=False,max_new_tokens=8,pad_token_id=tok.eos_token_id)
        text=tok.decode(out[0,ids['input_ids'].shape[1]:],skip_special_tokens=True).strip(); rows.append({'question':q,'text':text,'target':'18 days','correct':text=='18 days','fact_in_input':False})
    Path(output).mkdir(parents=True,exist_ok=True); model.save_pretrained(output); report={'train_rows':len(TRAIN),'steps':steps,'loss_initial':losses[0],'loss_final':losses[-1],'test_rows':rows,'accuracy':sum(r['correct'] for r in rows)/len(rows),'scope':'value-bearing Qwen LoRA Pod; no evidence text in test prompts'}; Path(output,'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); return report
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--steps',type=int,default=120); a=p.parse_args(); print(json.dumps(run(a.model,a.output,a.device,a.steps),indent=2,ensure_ascii=False))
