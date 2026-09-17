"""Opaque identities and independently reassigned counterfactual value worlds."""
import random
import re

COLORS=['red','blue','green','yellow']
FAMILIAR=['read','read_alt','eq0','eq1','eq2','eq3','member01','member23','map01','map02']
HELDOUT=['not_red','member013','map03','new_labels','conjunction']

def gold(spec,value):
    if spec.startswith('read'): return COLORS[value]
    if spec.startswith('eq'): return 'yes' if value==int(spec[-1]) else 'no'
    if spec.startswith('member'): return 'yes' if str(value) in spec[6:] else 'no'
    if spec.startswith('map'): return 'a' if str(value) in spec[3:] else 'b'
    if spec=='not_red': return 'yes' if value!=0 else 'no'
    if spec=='new_labels': return 'circle' if value in (0,3) else 'square'
    if spec=='conjunction': return 'yes' if value in (2,3) else 'no'
    raise ValueError(spec)

def question(c):
    item=c['identity'];noun=[f'the stored color of item {item}',f'the color assigned to item {item}',
    f'the registered color for item {item}',f'the recorded color of item {item}',
    f'the saved color for item {item}',f'the current color entry for item {item}'][c['wording']]
    spec=c['spec']
    if spec=='read':return f'What is {noun}? Return only the color word.'
    if spec=='read_alt':return f'A report needs {noun}. Supply the color as one word.'
    if spec.startswith('eq'):return f'Is {noun} {COLORS[int(spec[-1])]}? Return only yes or no.'
    if spec.startswith('member'):return f'Is {noun} in this set: {", ".join(COLORS[int(x)] for x in spec[6:])}? Return only yes or no.'
    if spec.startswith('map'):return f'For {noun}, return A if it is {" or ".join(COLORS[int(x)] for x in spec[3:])}, otherwise return B. Return only A or B.'
    if spec=='not_red':return f'Is {noun} different from red? Return only yes or no.'
    if spec=='new_labels':return f'For {noun}, return circle if it is red or yellow, otherwise return square. Return only circle or square.'
    if spec=='conjunction':return f'Is {noun} neither red nor blue? Return only yes or no.'
    raise ValueError(spec)

def datasets():
    rng=random.Random(20260914)
    ids=[''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789',k=10)) for _ in range(10)]
    assert len(set(ids))==10
    out={}
    for name,selected,wordstart,specs in [('train',ids[:4],0,FAMILIAR),('valid',ids[4:6],2,FAMILIAR),('test',ids[6:],4,FAMILIAR+HELDOUT)]:
        out[name]=[dict(identity=item,value=v,spec=spec,wording=wordstart+i%2,
        group='familiar' if spec in FAMILIAR else 'heldout') for spec in specs for i,item in enumerate(selected) for v in range(4)]
    return out

def score(rows):
    def subset(rs):return dict(n=len(rs),correct=sum(r['correct'] for r in rs),accuracy=sum(r['correct'] for r in rs)/len(rs))
    return dict(overall=subset(rows),groups={g:subset([r for r in rows if r['case']['group']==g]) for g in sorted({r['case']['group'] for r in rows})},
                operations={s:subset([r for r in rows if r['case']['spec']==s]) for s in sorted({r['case']['spec'] for r in rows})})

def strict_ok(text,expected,eos):
    return eos and re.fullmatch(r'\s*'+re.escape(expected)+r'[.!]?\s*',text,re.I) is not None
