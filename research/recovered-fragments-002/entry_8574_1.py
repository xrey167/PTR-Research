"""Exercise the original result reducer, without importing model dependencies."""
import ast
import copy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def historical():
    tree=ast.parse((ROOT/'semantic_linker/run_tskv_prefix_quotient.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize_engineering')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'historical_reducer','exec'),ns)
    return ns['summarize_engineering']

def fixture():
    cases=[]; rows=[]
    for p in ('identity','parity','order','equality'):
        expected,alternate=('0','1') if p=='identity' else ('Yes','No')
        c={'name':'FIXTURE_ONLY','value':0,'alternate':1,'predicate':p,
           'question':'fixture '+p,'expected':expected,'alternate_expected':alternate}
        cases.append(c)
        for mode in ('visible_text','none','full_cache','opposite_full_cache'):
            alt=mode=='opposite_full_cache'
            text=alternate if alt else expected
            logits=[0.0]*16;logits[2 if alt else 1]=1.0
            rows.append({'mode':mode,'case':copy.deepcopy(c),'text':text,
              'tokens':[2 if alt else 1,15],'eos_seen':True,'correct':True,
              'first_logits':logits,'selected_logits':logits[:4],
              'input_ids':[3,4,5,6],'full_input_ids':[3,4,5,6],
              'prefix_ids':[3,4],'student_input_ids':[5,6],
              'knowledge_text_tokens_student':0,'question_cache_positions':0,
              'assistant_cache_positions':0,'hidden_state_copies':0,'answer_anchor_copies':0})
    return cases,rows

def mutations():
    _,base=fixture();out={}
    a=copy.deepcopy(base);r=a[3];r.update(text='0',tokens=[1,15],correct=False);out['only_three_causal_changes']=a
    a=copy.deepcopy(base);a[2]['first_logits'][10]=.25;out['unselected_logit_drift']=a
    a=copy.deepcopy(base);a[2]['first_logits'][0]=.25;a[2]['selected_logits'][0]=.25;out['selected_logit_drift']=a
    a=copy.deepcopy(base)
    for i in (0,2): a[i].update(text='wrong',correct=True)
    out['forged_correct_field']=a
    a=copy.deepcopy(base);a[2]['first_logits']=a[2]['first_logits'][:4];out['truncated_logits']=a
    a=copy.deepcopy(base);a[2]['first_logits'][10]=float('nan');out['nonfinite_logits']=a
    a=copy.deepcopy(base);a[4:8]=copy.deepcopy(a[:4]);out['duplicated_case']=a
    a=copy.deepcopy(base);a.pop(1);out['missing_null_control']=a
    return out

if __name__=='__main__':
    reducer=historical();results={}
    for name,rows in mutations().items():
        try: results[name]={'historical_accepts':reducer(rows)['engineering_gate']}
        except Exception as e: results[name]={'historical_error':type(e).__name__}
    report={'synthetic_result_records_only':True,'new_model_sequences':0,'results':results}
    print(json.dumps(report,indent=2))
