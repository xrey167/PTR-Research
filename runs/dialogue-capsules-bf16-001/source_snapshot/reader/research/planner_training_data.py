"""Synthetic address-task splits, with distinct names, parts and prompt families.

No factual answer values. Targets come from the generator's selected catalogue
row or intentionally missing metric/entity/part. Never use the nine regression
cases as training input.
"""
import random
import json


FAMILIES = {
    'train': {
        'direct':'State the delivery lead time of {name} for {part}.',
        'planning':'We order {part} at {name}. How long until the shipment arrives?',
        'buffer':'Add three days of buffer to the delivery lead time for {part} from {name}.',
        'followup':'And how many days should we allow for delivery?',
        'price':'What price does {name} charge for {part}?',
        'age':'When was {name} founded?',
        'unknown_part':'What is the delivery lead time for {name} and {unknown_part}?',
        'unknown_name':'How long does {unknown_name} need to deliver {part}?',
    },
    'test': {
        'direct':'Tell procurement the waiting period for {part} supplied by {name}.',
        'planning':'If {name} receives our {part} order now, how far ahead can we schedule its arrival?',
        'buffer':'For {name} and {part}, would the normal delivery duration plus four extra days be enough?',
        'followup':'How long do they need for that component?',
        'price':'Give the unit cost of {part} purchased from {name}.',
        'age':'How many years has {name} been operating?',
        'unknown_part':'How long must we wait for {unknown_part} supplied by {name}?',
        'unknown_name':'What waiting period applies when {unknown_name} supplies {part}?',
    },
}


def build_data():
    result={}
    for split,names,parts,count in [
        ('train',['Arden Supply','Belvia Works','Ceron Parts','Daxel Trade'],['A11','B22','C33','D44'],2),
        ('test',['Lurena Supply','Montri Works','Nesvara Parts','Ordel Trade'],['L51','M62','N73','P84'],1)]:
        rng=random.Random(917 if split=='train' else 1931)
        rows=[];seen=set()
        for family,template in FAMILIES[split].items():
            configurations=[(size,slot,repeat) for size in [2,3] for slot in range(size) for repeat in range(count)]
            for i,(size,slot,repeat) in enumerate(configurations):
                while True:
                    order=rng.sample(range(4),size)
                    signature=(family,size,slot,tuple(order))
                    if signature not in seen:seen.add(signature);break
                catalogue=[{'address':slot+1,'subject':names[j], 'aliases':[names[j]],
                    'component':parts[j],'predicate':'lead_time','role':'FACT'} for slot,j in enumerate(order)]
                chosen=catalogue[slot]
                question=template.format(name=chosen['subject'],part=chosen['component'],
                    unknown_part='Z99' if split=='train' else 'R98',
                    unknown_name='Faron Supply' if split=='train' else 'Pelmin Trade')
                history=[]
                if family=='followup':
                    context=(f"We are discussing {chosen['component']} from {chosen['subject']}." if split=='train'
                             else f"Our supplier is {chosen['subject']}; the component under discussion is {chosen['component']}.")
                    history=[{'role':'user','content':context}]
                target='UNKNOWN' if family in {'price','age','unknown_part','unknown_name'} else f"ADDRESS {chosen['address']}"
                rows.append({'id':f'{split}:{family}:{i}','family':family,
                    'input':{'catalogue':catalogue,'dialogue':[*history,{'role':'user','content':question}]},
                    'target':target,'selected_address':chosen['address']})
                # Same syntax, different catalogue availability: a lexical
                # question-family shortcut cannot solve these paired examples.
                if family in {'unknown_part','unknown_name'}:
                    paired=template.format(name=chosen['subject'],part=chosen['component'],
                        unknown_part=chosen['component'],unknown_name=chosen['subject'])
                    rows.append({'id':f'{split}:{family}:{i}:present','family':family+'_present',
                        'input':{'catalogue':catalogue,'dialogue':[{'role':'user','content':paired}]},
                        'target':f"ADDRESS {chosen['address']}",'selected_address':chosen['address']})
        result[split]=rows
    return result
