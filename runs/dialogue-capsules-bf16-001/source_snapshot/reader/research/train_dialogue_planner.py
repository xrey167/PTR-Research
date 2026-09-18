"""Train one generic planning LoRA; evaluate a predeclared disjoint split once."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from neural_pods.model import PodModel
from neural_pods.registry import Registry, hash_files
from research.dialogue_access import SYSTEM
from research.planner_training_data import build_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3);a=p.parse_args()
    if a.epochs<1:raise ValueError('Positive epochs required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=build_data();dataset=out/'dataset.json'
    dataset.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=json.loads(Path('runs/embedded-symlink-002/report.json').read_text(encoding='utf-8'))
    protocol={'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'system':SYSTEM,
        'epochs':a.epochs,'steps':len(data['train'])*a.epochs,'lr':.0003,'rank':8,'target_modules':['q_proj','v_proj'],
        'model':manifest['models']['qwen'],'expected_base_sha256':manifest['base_weights_sha256'],
        'evaluation_policy':f"Fixed epochs, no test-driven checkpoint selection; compare free decoding before/after on {len(data['test'])} predeclared cases",
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('research/planner_training_data.py'),Path('research/dialogue_access.py')]}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    report={'status':'running','phase':'load','losses':[],'baseline':[],'trained':[],'full_research_goal_complete':False}
    started=time.perf_counter()
    def save():
        report['elapsed_s']=time.perf_counter()-started
        temp=out/'report.json.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(out/'report.json')
    save()
    try:
        m=PodModel(protocol['model']['path']);tok=m.tokenizer
        assert m.frozen_hash()==protocol['expected_base_sha256']
        def prompt(row):
            return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(row['input'],ensure_ascii=False)}],tokenize=False,add_generation_prompt=True)
        def evaluate(field,adapter):
            report['phase']=field;save()
            m.model.eval()
            if adapter:m.model.set_adapter(adapter)
            for row in data['test']:
                inputs=tok(prompt(row),return_tensors='pt',add_special_tokens=False)
                t=time.perf_counter()
                with (nullcontext() if adapter else m.model.disable_adapter()),torch.inference_mode():
                    output=m.model.generate(**inputs,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
                tokens=output[0,inputs.input_ids.shape[1]:].tolist()
                text=tok.decode(tokens,skip_special_tokens=True).strip()
                eos_ids=m.model.generation_config.eos_token_id
                eos=set(eos_ids if isinstance(eos_ids,list) else [eos_ids]);eos.add(tok.eos_token_id)
                ended=bool(tokens and tokens[-1] in eos)
                report[field].append({'id':row['id'],'target':row['target'],'text':text,'tokens':tokens,'eos':ended,
                    'correct':text==row['target'] and ended,'seconds':time.perf_counter()-t})
                save()
            print(json.dumps({'phase':field,'correct':sum(r['correct'] for r in report[field]),'total':len(report[field])}),flush=True)
        evaluate('baseline',None)
        name='dialogue_planner';m.model.add_adapter(name,m.config);m.model.set_adapter(name);m.model.train()
        encoded=[]
        for row in data['train']:
            prefix=tok.encode(prompt(row),add_special_tokens=False)
            target=tok.encode(row['target'],add_special_tokens=False)+[tok.eos_token_id]
            ids=torch.tensor([prefix+target]);labels=ids.clone();labels[:,:len(prefix)]=-100
            encoded.append((ids,labels))
        params=[p for p in m.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=protocol['lr'],weight_decay=0.)
        report['phase']='training';report['trainable_parameters']=sum(p.numel() for p in params)
        rng=random.Random(29217);step=0
        for epoch in range(a.epochs):
            order=list(range(len(encoded)));rng.shuffle(order)
            for index in order:
                ids,labels=encoded[index];optimizer.zero_grad()
                prediction=m.model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,use_cache=False)
                loss=prediction.loss
                if not torch.isfinite(loss):raise ArithmeticError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();step+=1
                report['losses'].append(float(loss.detach()));save()
                if step==1 or step%16==0:print(json.dumps({'step':step,'steps':protocol['steps'],'loss':report['losses'][-1],'elapsed_s':report['elapsed_s']}),flush=True)
            m.model.save_pretrained(out/f'checkpoint_epoch_{epoch+1}',selected_adapters=[name],save_embedding_layers=False)
        m.model.eval();m.model.save_pretrained(out/'adapter',selected_adapters=[name],save_embedding_layers=False)
        report['base_unchanged']=m.frozen_hash()==protocol['expected_base_sha256'];assert report['base_unchanged']
        reg=Registry(out/'registry.sqlite3')
        try:
            training=reg.origin('planner-training','synthetic-catalogue-dialogues','1',protocol,acl=['buyer'])
            model_origin=reg.origin('model-release',protocol['model']['repo'],protocol['model']['revision'],
                {'base_sha256':protocol['expected_base_sha256']},acl=['buyer'])
            capability=reg.publish('capability:dialogue-address-planner',{'role':'PROCEDURE','task':'value-free-dialogue-address-selection',
                'schema':'dialogue-planner:v1'},[training,model_origin],'buyer',['buyer'])
            report['planner_artifact']=reg.artifact('lora',{'adapter':name,'files':hash_files(out/'adapter'/name),
                'base_sha256':protocol['expected_base_sha256'],'task':'dialogue-planning','protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()},[capability],'buyer')
        finally:reg.close()
        evaluate('trained',name)
        report.update(status='completed',phase='done',baseline_correct=sum(r['correct'] for r in report['baseline']),
                      trained_correct=sum(r['correct'] for r in report['trained']))
        report['generalization_gate_passed']=report['trained_correct']==len(data['test'])
        save();print(json.dumps({k:v for k,v in report.items() if k not in ['losses','baseline','trained']}),flush=True)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
