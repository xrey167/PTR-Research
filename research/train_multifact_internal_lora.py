"""Train a multi-fact Qwen LoRA Pod and test value recall without evidence."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

FACTS=[('Müller GmbH','X12',18),('Avelin Components','A31',14),('Gavren Parts','G17',52),('Helvona Works','H28',56),('Nordwerk AG','N44',31)]
TRAIN=["What is the lead time for {s} {c}?","How long does {s} need for component {c}?","State the delivery duration of {s} for {c}.","Welche Lieferzeit hat {s} für {c}?","Wie viele Tage braucht {s} bei {c}?","Current delivery lead time for supplier {s}?","Nenne die aktuelle Lieferzeit von {s}."]
TEST=["Current delivery lead time for supplier {s}?","{c} procurement delay from {s}?","How long does {s} need for {c}?","Welche Lieferdauer gilt für den Lieferanten {s} bei {c}?","Give the current {c} transit time for {s}."]
NEG_TRAIN=["What is the lead time for Unknown Supplier Z99?","Nenne die Lieferzeit von Fremdfirma Q77."]*5
NEG_TEST=["What is the delivery time for Unknown Supplier Z99?","Nenne die Lieferzeit von Fremdfirma Q77.","How long does External Vendor E55 need for E55?"]

def run(model_path, output, device='cuda', steps=240):
    torch.manual_seed(3407); random.seed(3407); tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to(device)
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')); model.train(); enc=[]
    for s,c,v in FACTS:
        for q in TRAIN:
            p=tok.apply_chat_template([{'role':'system','content':'Answer the exact delivery duration in days. If the supplier is unknown, answer UNKNOWN.'},{'role':'user','content':q.format(s=s,c=c)}],tokenize=False,add_generation_prompt=True); pref=tok.encode(p,add_special_tokens=False); ids0=torch.tensor([pref+tok.encode(f'{v} days'+tok.eos_token,add_special_tokens=False)],dtype=torch.long,device=device); enc.append(({'input_ids':ids0,'attention_mask':torch.ones_like(ids0)},torch.cat([torch.full((1,len(pref)),-100,dtype=torch.long,device=device),ids0[:,len(pref):]],1)))
    for q in NEG_TRAIN:
        p=tok.apply_chat_template([{'role':'system','content':'Answer only the exact delivery duration in days. If the supplier is unknown, answer UNKNOWN.'},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True); pref=tok.encode(p,add_special_tokens=False); ids0=torch.tensor([pref+tok.encode('UNKNOWN'+tok.eos_token,add_special_tokens=False)],dtype=torch.long,device=device); enc.append(({'input_ids':ids0,'attention_mask':torch.ones_like(ids0)},torch.cat([torch.full((1,len(pref)),-100,dtype=torch.long,device=device),ids0[:,len(pref):]],1)))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4); losses=[]
    for step in range(steps):
        ids,labels=enc[step%len(enc)]; opt.zero_grad(set_to_none=True); loss=model(**ids,labels=labels,use_cache=False).loss; loss.backward(); opt.step(); losses.append(float(loss.detach()))
    model.eval(); rows=[]
    for s,c,v in FACTS:
        for q in TEST:
            prompt=tok.apply_chat_template([{'role':'system','content':'Answer the exact delivery duration in days. If the supplier is unknown, answer UNKNOWN.'},{'role':'user','content':q.format(s=s,c=c)}],tokenize=False,add_generation_prompt=True); ids=tok(prompt,return_tensors='pt').to(device)
            with torch.inference_mode(): out=model.generate(**ids,do_sample=False,max_new_tokens=8,pad_token_id=tok.eos_token_id)
            text=tok.decode(out[0,ids['input_ids'].shape[1]:],skip_special_tokens=True).strip(); rows.append({'supplier':s,'component':c,'question':q.format(s=s,c=c),'text':text,'target':f'{v} days','correct':text==f'{v} days','fact_in_input':False})
    for q in NEG_TEST:
        prompt=tok.apply_chat_template([{'role':'system','content':'Answer only the exact delivery duration in days. If the supplier is unknown, answer UNKNOWN.'},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True); ids=tok(prompt,return_tensors='pt').to(device)
        with torch.inference_mode(): out=model.generate(**ids,do_sample=False,max_new_tokens=8,pad_token_id=tok.eos_token_id)
        text=tok.decode(out[0,ids['input_ids'].shape[1]:],skip_special_tokens=True).strip(); rows.append({'question':q,'text':text,'target':'UNKNOWN','correct':text=='UNKNOWN','fact_in_input':False})
    Path(output).mkdir(parents=True,exist_ok=True); model.save_pretrained(output); report={'facts':len(FACTS),'train_rows':len(enc),'steps':steps,'loss_initial':losses[0],'loss_final':losses[-1],'test_rows':rows,'accuracy':sum(r['correct'] for r in rows)/len(rows),'known_accuracy':sum(r['correct'] for r in rows[:len(FACTS)*len(TEST)])/len(FACTS)/len(TEST),'unknown_accuracy':sum(r['correct'] for r in rows[len(FACTS)*len(TEST):])/len(NEG_TEST),'scope':'multi-fact value-bearing Qwen LoRA Pod with abstention; no evidence text in test prompts'}; Path(output,'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); return report
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--steps',type=int,default=240); a=p.parse_args(); print(json.dumps(run(a.model,a.output,a.device,a.steps),indent=2,ensure_ascii=False))
