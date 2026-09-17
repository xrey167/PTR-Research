import copy
import json
import random
from pathlib import Path
from gate import evaluate
from reproduce import fixture,mutations,historical

cases,base=fixture()
results=[]
def check(name,rows,expected):
    result=evaluate(rows,cases,16,{15})
    assert result['cqta2_e0_gate'] is expected,(name,result)
    results.append({'name':name,'expected_accept':expected,'actual':result})
check('valid_fixture',base,True)
for name,rows in mutations().items():
    assert historical()(rows)['engineering_gate'] is True,name
    check(name,rows,False)
a=copy.deepcopy(base);random.Random(42).shuffle(a);check('order_independent_pairing',a,True)
a=copy.deepcopy(base);a[2]['knowledge_text_tokens_student']=1;check('declared_fact_leak',a,False)
a=copy.deepcopy(base);a[2]['student_input_ids']=[5];check('broken_prompt_concatenation',a,False)
a=copy.deepcopy(base);a[2]['tokens']=[1,14];a[2]['eos_seen']=False;check('missing_eos',a,False)
a=copy.deepcopy(base);a[2]['tokens']=[2,15];check('greedy_mismatch',a,False)
a=copy.deepcopy(base);a[2]['first_logits'][0]=-0.0;check('signed_zero_byte_mismatch',a,False)
report={'status':'pass','tests':len(results),'synthetic_records_only':True,'new_model_sequences':0,'checks':results}
Path(__file__).with_name('test_results.json').write_text(json.dumps(report,indent=2)+'\\n')
print(json.dumps({k:v for k,v in report.items() if k"'!='"'checks'},indent=2))
