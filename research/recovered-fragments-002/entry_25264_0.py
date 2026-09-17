import sys,json,hashlib,time,argparse,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def run():
 parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--stage',choices=['development','validation'],required=True)
 args=parser.parse_args();here=Path(__file__).parent;out=here/args.stage;out.mkdir(exist_ok=False)
 import torch,numpy as np,transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from decoder_semantics.compiler import ProvenanceCompiler
 torch.set_num_threads(4)
 att=json.loads((ROOT/'model_attestation.json').read_text())
 man=json.loads((args.model/'download_manifest.json').read_text())
 assert man['revision']==att['revision']
 for name,entry in man['files'].items():assert digest(args.model/name)==entry['sha256']
 assert digest(args.model/'model.safetensors')==att['downloaded']['sha256']
 if args.stage=='validation':
  seal=json.loads((here/'seal.json').read_text())
  for p,h in seal.items():assert digest(ROOT/p)==h,p
 compiler=ProvenanceCompiler(ROOT)
 tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval();model.requires_grad_(False)
 codes=torch.from_numpy(compiler.compile_symbols(np.array([10000,10001]),'qwen3_0_6b'))
 rng=random.Random(910731)
 names=['DEV'] if args.stage=='development' else ['R'+''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ',k=8)) for _ in range(6)]
 cases=[]
 for name in names:
  for negate in ([False] if args.stage=='development' else [False,True]):
   question=(f'Is record {name} inactive?' if negate else f'Is record {name} active?')+' Answer only Yes or No.'
   for value in (0,1):cases.append({'name':name,'negated':negate,'value':value,'question':question,'expected':'Yes' if bool(value)!=negate else 'No'})
 (out/'cases.json').write_text(json.dumps(cases,indent=2))
 rows=[]
 for mode in ['none','cc8','opposite','text']:
  for start in range(0,len(cases),8):
   batch=cases[start:start+8];prompts=[]
   for c in batch:
    q=c['question']
    if mode=='text':q=f"Stored fact: record {c['name']} is {'active' if c['value'] else 'inactive'}.\n"+q
    prompts.append(tok.apply_chat_template([{'role':'system','content':'Follow the requested format. Give exactly one word, Yes or No, without explanation.'},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
   inputs=tok(prompts,padding=True,return_tensors='pt',add_special_tokens=False)
   handle=None;capture={'called':0,'delta_norm':[]}
   if mode in ('cc8','opposite'):
    delta=codes[torch.tensor([c['value']^(mode=='opposite') for c in batch],dtype=torch.long)]
    def hook(module,ins,outs):
     if capture['called']:return outs
     capture['called']+=1
     h=outs[0] if isinstance(outs,tuple) else outs
     altered=h.clone();altered[:,-1]+=delta
     capture['delta_norm']=torch.linalg.vector_norm(altered[:,-1]-h[:,-1],dim=-1).tolist()
     return (altered,)+outs[1:] if isinstance(outs,tuple) else altered
    handle=model.model.layers[23].register_forward_hook(hook)
   try:
    with torch.no_grad():generated=model.generate(**inputs,max_new_tokens=4,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,return_dict_in_generate=True,output_scores=True)
   finally:
    if handle:handle.remove()
   eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
   for j,(c,seq) in enumerate(zip(batch,generated.sequences[:,inputs['input_ids'].shape[1]:].tolist())):
    end=next((i for i,x in enumerate(seq) if x in eos),None);text=tok.decode(seq[:end] if end is not None else seq,skip_special_tokens=False).strip()
    logits=generated.scores[0][j]
    row={'mode':mode,'case':c,'tokens':seq,'text':text,'eos_seen':end is not None,'correct':end is not None and text==c['expected'],'hook_called':capture['called'],
         'delta_norm':capture['delta_norm'][j] if capture['delta_norm'] else 0,
         'first_logits':{s:float(logits[tok.encode(s,add_special_tokens=False)[0]]) for s in ('Yes','No')}}
    rows.append(row)
    with (out/'answers.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(mode,start+len(batch),'/',len(cases),flush=True)
 counts={m:sum(r['correct'] for r in rows if r['mode']==m) for m in ('none','cc8','opposite','text')}
 pairs={}
 for mode in counts:
  rr=[r for r in rows if r['mode']==mode];pairs[mode]=sum(rr[i]['correct'] and rr[i+1]['correct'] for i in range(0,len(rr),2))
 result={'stage':args.stage,'cases':len(cases),'answers':len(rows),'correct':counts,'counterfactual_pairs_correct':pairs,'candidate_gate':counts['cc8']==len(cases) and pairs['cc8']==len(cases)//2,
         'model':att['model'],'revision':att['revision'],'weight_sha256':att['downloaded']['sha256'],'torch':torch.__version__,'transformers':transformers.__version__,'full_dod':False,
         'scope':'literal additive CC8 injection; failure is not a bound on future learned adapters'}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':run()
