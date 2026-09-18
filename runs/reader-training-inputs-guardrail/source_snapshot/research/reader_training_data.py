"""Synthetic reader transfer data; no existing regression or diagnostic cases.

Splits separate supplier identities, components, base values and phrasings.
This is narrow synthetic procurement training, not full research acceptance.
"""

from neural_pods.pod_types import pod_type_for_task


SPLITS = {
    'train': (['Avelin Components', 'Branor Supply', 'Cedrin Parts', 'Dovira Works'],
              ['A31', 'B42', 'C53', 'D64'], [14, 18, 21, 25]),
    'dev': (['Evrana Components', 'Faldor Supply'], ['E75', 'F86'], [35, 39]),
    'test': (['Gavren Parts', 'Helvona Works'], ['G17', 'H28'], [52, 56]),
}


def duration(days, language, unit):
    if unit == 'days':
        return f'{days} ' + ('days' if language == 'en' else 'Tage')
    words = {'en': ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
             'de': ['null', 'eine', 'zwei', 'drei', 'vier', 'f\u00fcnf', 'sechs', 'sieben', 'acht', 'neun']}
    weeks, rest = divmod(days, 7)
    if language == 'en':
        text = words[language][weeks] + (' week' if weeks == 1 else ' weeks')
        return text + (f' and {rest} days' if rest else '')
    text = words[language][weeks] + (' Woche' if weeks == 1 else ' Wochen')
    return text + (f' und {rest} Tage' if rest else '')


def build_data():
    result = {}
    for split, (names, parts, values) in SPLITS.items():
        rows = []
        for name, part in zip(names, parts):
            for value in values:
                for language in ('en', 'de'):
                    reference = f'{name} / {part}'
                    stems = {
                        'train': {'en': f'For {reference}, ', 'de': f'F\u00fcr {reference}: '},
                        'dev': {'en': f'Consider a new order of {part} from {name}. ', 'de': f'Betrachte eine neue Bestellung von {part} bei {name}. '},
                        'test': {'en': f'Procurement is scheduling {part} supplied by {name}. ', 'de': f'Der Einkauf plant mit {part} vom Lieferanten {name}. '},
                    }
                    prompts = {
                        'train': {'en': ('what is the lead time?', 'what duration includes four extra buffer days?', 'is a deadline of {deadline} sufficient?'),
                                  'de': ('Wie lang ist die Lieferzeit?', 'Welche Dauer ergibt sich mit vier zus\u00e4tzlichen Puffertagen?', 'Gen\u00fcgt eine Frist von {deadline}?')},
                        'dev': {'en': ('how many days until arrival?', 'add four days to the waiting period.', 'can delivery finish within {deadline}?'),
                                'de': ('Wie viele Tage dauert es bis zur Ankunft?', 'Addiere vier Tage zur Wartezeit.', 'Kann die Lieferung innerhalb von {deadline} eintreffen?')},
                        'test': {'en': ('state the delivery duration in days.', 'include a four-day allowance in the total duration.', 'will the order arrive no later than {deadline} after ordering?'),
                                 'de': ('Nenne die Lieferdauer in Tagen.', 'Ber\u00fccksichtige einen Zuschlag von vier Tagen in der Gesamtdauer.', 'Trifft die Bestellung sp\u00e4testens {deadline} nach Bestellung ein?')},
                    }
                    stem = stems[split][language]
                    lookup, buffer, deadline_prompt = prompts[split][language]
                    evidence = {'supplier': name, 'component': part, 'lead_time_days': value}
                    cases = [('lookup', lookup, value, None), ('buffer', buffer, value + 4, None)]
                    for unit in ('days', 'weeks'):
                        for relation, limit in [('short', value - 2), ('equal', value), ('long', value + 3)]:
                            cases.append((f'deadline_{unit}_{relation}', deadline_prompt.format(
                                deadline=duration(limit, language, unit)), value <= limit, limit))
                    # Missing evidence has no value: do not duplicate the same input per value.
                    if value == values[0]:
                        cases.append(('missing', lookup, None, None))
                    for task, question, answer, limit in cases:
                        if answer is None:
                            target = 'UNKNOWN'
                        elif type(answer) is bool:
                            target = ('Yes' if answer else 'No') if language == 'en' else ('Ja' if answer else 'Nein')
                            target += f'; {value} ' + ('days' if language == 'en' else 'Tage')
                        else:
                            target = f'{answer} ' + ('days' if language == 'en' else 'Tage')
                        rows.append({'id': f'{split}:{part}:{value}:{language}:{task}', 'task': task,
                            'pod_type': pod_type_for_task(task).value,
                            'language': language, 'evidence': None if task == 'missing' else dict(evidence),
                            'question': stem + question, 'history': [], 'target': target,
                            'assessment': {'supplier': name, 'component': part, 'value': value,
                                           'deadline_days': limit, 'answer': answer}})
                    old_value = values[(values.index(value) + 1) % len(values)]
                    rows.extend(dialogue_rows(split, evidence, language, stem, old_value))
        result[split] = rows
    # A small typed-artifact curriculum teaches the reader that a model Pod is
    # metadata about an executable adapter, not a supplier fact to answer from.
    result['train'].extend(model_pod_rows())
    for split in ('train', 'dev', 'test'):
        result[split].extend(multi_hop_rows(split))
    return result


def multi_hop_rows(split):
    """Small held-out composition curriculum: supplier transit + customs leg."""
    cases = {
        'train': [('Avelin Components', 'A31', 14, 3), ('Branor Supply', 'B42', 18, 5),
                  ('Cedrin Parts', 'C53', 21, 2), ('Dovira Works', 'D64', 25, 4)],
        'dev': [('Evrana Components', 'E75', 35, 6), ('Faldor Supply', 'F86', 39, 1)],
        'test': [('Gavren Parts', 'G17', 52, 7), ('Helvona Works', 'H28', 56, 3)],
    }
    rows = []
    for name, part, lead, customs in cases[split]:
        total = lead + customs
        for language in ('en', 'de'):
            if language == 'en':
                question = f"What is the total arrival time for {part} from {name} if supplier transport takes {lead} days and customs adds {customs} days?"
                target = f"{total} days"
                facts = [f"Supplier {name} needs {lead} days for component {part}.", f"Customs for component {part} adds {customs} days."]
            else:
                question = f"Wie lange dauert die gesamte Ankunft von {part} bei {name}, wenn der Lieferant {lead} Tage und der Zoll {customs} Tage braucht?"
                target = f"{total} Tage"
                facts = [f"Lieferant {name} braucht {lead} Tage für Komponente {part}.", f"Der Zoll für Komponente {part} fügt {customs} Tage hinzu."]
            rows.append({'id': f'{split}:multihop:{part}:{language}', 'task': 'multi_hop_total',
                'pod_type': 'math', 'language': language, 'evidence': {'facts': facts},
                'question': question, 'history': [], 'target': target,
                'assessment': {'supplier': name, 'component': part, 'value': total, 'answer': total,
                               'deadline_days': None, 'hop_count': 2}})
    return rows


def model_pod_rows():
    rows = []
    prompts = {
        'en': ('What kind of Pod is this adapter?', 'Which artifact type is registered?'),
        'de': ('Welche Art Pod ist dieser Adapter?', 'Welcher Artefakttyp ist registriert?'),
    }
    for language, questions in prompts.items():
        for index, question in enumerate(questions):
            rows.append({'id': f'train:model:{language}:{index}', 'task': 'pod_kind',
                'pod_type': 'model', 'language': language,
                'evidence': {'model_name': 'Qwen2.5-3B', 'adapter_sha256': 'adapter-demo'},
                'question': question, 'history': [], 'target': 'model Pod' if language == 'en' else 'Modell-Pod',
                'assessment': {'supplier': None, 'component': None, 'value': None, 'answer': 'model'}})
    return rows


def dialogue_rows(split, evidence, language, context, old_value):
    """Synthetic context references and stale-value precedence, not real receipts."""
    forms = {
        'train': {'en': ('Give me their delivery duration.', 'And including four buffer days?', 'What is the current delivery duration now?'),
                  'de': ('Nenne mir deren Lieferdauer.', 'Und einschlie\u00dflich vier Puffertagen?', 'Welche Lieferdauer gilt jetzt aktuell?')},
        'dev': {'en': ('How many days should we allocate for them?', 'Allow four additional days; what is the total?', 'Use the latest information: how many days?'),
                'de': ('Wie viele Tage sollten wir daf\u00fcr vorsehen?', 'Plane vier weitere Tage ein; wie lautet die Summe?', 'Nach dem neuesten Stand: wie viele Tage?')},
        'test': {'en': ('State the waiting time for that order.', 'Extend that period by four days and give the result.', 'Which duration applies to the order at present?'),
                 'de': ('Gib die Wartezeit f\u00fcr diese Bestellung an.', 'Verl\u00e4ngere diese Dauer um vier Tage und nenne das Ergebnis.', 'Welche Dauer trifft derzeit auf diese Bestellung zu?')},
    }
    rows = []
    value = evidence['lead_time_days']
    for task, question, answer in zip(('followup_lookup', 'followup_buffer', 'stale_followup'),
                                      forms[split][language], (value, value + 4, value)):
        history = [{'role': 'user', 'content': context.strip()}]
        if task == 'stale_followup':
            earlier = (f'Earlier estimate: {old_value} days.' if language == 'en'
                       else f'Fr\u00fchere Sch\u00e4tzung: {old_value} Tage.')
            history.append({'role': 'assistant', 'content': earlier})
        rows.append({'id': f"{split}:{evidence['component']}:{value}:{language}:{task}",
            'task': task, 'pod_type': pod_type_for_task(task).value, 'language': language, 'evidence': dict(evidence),
            'question': question, 'history': history,
            'target': f'{answer} ' + ('days' if language == 'en' else 'Tage'),
            'assessment': {'supplier': evidence['supplier'], 'component': evidence['component'],
                           'value': value, 'answer': answer, 'deadline_days': None,
                           'prior_value': old_value if task == 'stale_followup' else None}})
    return rows
