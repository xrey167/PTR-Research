"""Controlled diagnosis of the observed reader error, not a held-out benchmark.

Only language and deadline wording vary. The fact stays at 27 days and the
dialogue history supplies identity but no value. Expected answers are scoring
metadata and must never be included in the reader prompt.
"""


def build_cases():
    durations = [
        ('short', 21, {'en': ('21 days', 'three weeks'), 'de': ('21 Tage', 'drei Wochen')}),
        ('equal', 27, {'en': ('27 days', 'three weeks and six days'), 'de': ('27 Tage', 'drei Wochen und sechs Tage')}),
        ('long', 28, {'en': ('28 days', 'four weeks'), 'de': ('28 Tage', 'vier Wochen')}),
    ]
    rows = []
    for language in ('en', 'de'):
        history = [{'role': 'user', 'content': (
            'We are discussing X12 from Mueller GmbH.' if language == 'en'
            else 'Es geht um X12 von M\u00fcller GmbH.')}]
        for relation, days, words in durations:
            for unit, deadline in zip(('days', 'weeks'), words[language]):
                question = (f'Are {deadline} enough until delivery?' if language == 'en'
                            else f'Reichen {deadline} bis zur Lieferung?')
                rows.append({
                    'id': f'{language}:{unit}:{relation}',
                    'reader_input': {'history': [dict(m) for m in history], 'question': question},
                    'assessment': {'language': language, 'unit_wording': unit,
                                   'relation': relation, 'deadline_days': days,
                                   'lead_time_days': 27, 'expected_sufficient': 27 <= days},
                })
    return rows


if __name__ == '__main__':
    import json
    print(json.dumps({'scope': 'Controlled diagnosis; not training or held-out generalization data',
                      'cases': build_cases()}, ensure_ascii=False, indent=2))
