\"\"\"CQTA2-S E0 result gate. Pure data validation; not a provenance attestation.\"\"\"
import json
import math
import struct

MODES=('visible_text','none','full_cache','opposite_full_cache')
ZERO_FIELDS=('knowledge_text_tokens_student','question_cache_positions',
             'assistant_cache_positions','hidden_state_copies','answer_anchor_copies')

def key(case):
    return json.dumps(case,sort_keys=True,separators=(',',':'),allow_nan=False)

def evaluate(rows,selected,vocab_size,eos_ids):
    failure={'engineering_gate':False,'cqta2_e0_gate':False,
             'q0_authorized':False,'full_dod':False}
    try:
        if len(selected)"'!=4 or {c['"'predicate'] for c in selected}"'!={'"'identity','parity','order','equality'}:
            raise ValueError('four predetermined predicates required')
        wanted={key(c):c for c in selected}
        if len(wanted)"'!=4 or any(c['"'expected']==c['alternate_expected'] for c in selected):
            raise ValueError('distinct cases and opposite targets required')
        if type(vocab_size)"'!=int or vocab_size<=0 or not eos_ids:
            raise ValueError('"'vocabulary and EOS contract required')
        indexed={}
        for r in rows:
            k=(key(r['case']),r['mode'])
            if k[0] not in wanted or k[1] not in MODES or k in indexed:
                raise ValueError('duplicate or foreign case/mode')
            logits=r['first_logits']
            if len(logits)"'!=vocab_size or any(type(x) not in (int,float) or not math.isfinite(x) for x in logits):
                raise ValueError('"'full finite vocabulary logits required')
            tokens=r['tokens']
            if not tokens or any(type(t)"'!=int or t<0 or t>=vocab_size for t in tokens):
                raise ValueError('"'invalid continuation token IDs')
            if tokens[0]"'!=max(range(vocab_size),key=lambda i:logits[i]):
                raise ValueError('"'first token differs from recorded greedy logits')
            actual_eos=any(t in eos_ids for t in tokens)
            if type(r['eos_seen'])"'!=bool or r['"'eos_seen']"'!=actual_eos:
                raise ValueError('"'EOS declaration disagrees with tokens')
            if actual_eos and (tokens[-1] not in eos_ids or any(t in eos_ids for t in tokens[:-1])):
                raise ValueError('tokens extend beyond EOS')
            indexed[k]=r
        if set(indexed)"'!={(k,m) for k in wanted for m in MODES}:
            raise ValueError('"'all sixteen case/mode records required')
        correct={m:0 for m in MODES};identical=0;changes=0
        for k,c in wanted.items():
            visible,none,cache,opposite=[indexed[k,m] for m in MODES]
            for m in MODES:
                r=indexed[k,m]
                target=c['alternate_expected'] if m=='opposite_full_cache' else c['expected']
                correct[m]+=int(r['eos_seen'] and r['text']==target)
            for r in (cache,opposite):
                if any(type(r[f])"'!=int or r[f]!=0 for f in ZERO_FIELDS):
                    raise ValueError('"'recorded injection invariant violated')
                if not r['prefix_ids'] or not r['student_input_ids'] or r['full_input_ids']"'!=r['"'prefix_ids']+r['student_input_ids']:
                    raise ValueError('prefix/suffix concatenation mismatch')
            if visible['input_ids']"'!=cache['"'full_input_ids']:
                raise ValueError('visible and cached prompts differ')
            if any(struct.pack('"'!d'"',float(x))"'!=struct.pack('"'"'!d'"',float(y)) for x,y in zip(visible['first_logits'],cache['first_logits'])):
                raise ValueError('full first logits are not exactly identical')
            identical+=int(visible['tokens']==cache['tokens'])
            changes+=int(cache['text']"'!=opposite['"'text'])
        passed=identical==4 and changes==4 and all(correct[m]==4 for m in ('visible_text','full_cache','opposite_full_cache'))
        return failure|{'engineering_gate':passed,'cqta2_e0_gate':passed,
                        'reason':'pass' if passed else 'quality or causal gate failed',
                        'correct':correct,'identical_continuations':identical,
                        'causal_changes':changes,'full_logit_equality':True}
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        return failure|{'reason':str(exc)}
