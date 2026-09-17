"""Create a focused continuation curriculum for the held-out query forms."""
from __future__ import annotations
from research.generate_generalization_benchmark import TEMPLATES

def rows():
    result=[]
    for i in range(10):
        part=f"assembly-{chr(65+i%26)}{i:02d}"; alias=f"internal-handle-{i:02d}"
        transit, customs, handling = 8+i%9, 2+(i*3)%6, 1+i%4
        supplier=f"Vendor {i:02d}"; total=transit+customs+handling
        facts=[f"{supplier} dispatches {part}; standard transit is {transit} calendar days.",
               f"{supplier} {part} import clearance normally takes {customs} calendar days.",
               f"Receiving and put-away for {supplier} {part} takes {handling} calendar days."]
        for j in (2,3):
            q=TEMPLATES[j].format(alias=alias,part=f"line-code-{i:02d}")
            result.append({'id':f'generalization-curriculum:{i}:{j}','task':'multi_hop_total',
                'pod_type':'math','language':'en','evidence':{'facts':facts},'question':q,
                'history':[],'target':f'{total} days','assessment':{'answer':total,'hop_count':3}})
    return result

if __name__=='__main__':
 import json; print(json.dumps({'rows':len(rows()),'examples':rows()[:2]},indent=2))
