import unittest
import json
from research.reader_training_data import build_data, duration


class ReaderDataTest(unittest.TestCase):
    def test_disjoint_splits_and_counterfactual_evidence(self):
        data = build_data()
        self.assertEqual({s: len(rows) for s, rows in data.items()}, {'train': 372, 'dev': 96, 'test': 96})
        self.assertEqual(sum(row['pod_type'] == 'model' for row in data['train']), 4)
        for field in ('supplier', 'component', 'value'):
            groups = [set(r['assessment'][field] for r in data[s]) for s in ('train', 'dev', 'test')]
            for i in range(3):
                for j in range(i):
                    self.assertFalse(groups[i] & groups[j])
        for rows in data.values():
            self.assertEqual(len({r['id'] for r in rows}), len(rows))
            inputs = [json.dumps({k: r[k] for k in ('evidence', 'question', 'history')}, sort_keys=True) for r in rows]
            self.assertEqual(len(set(inputs)), len(rows))
            for missing in (r for r in rows if r['task'] == 'missing'):
                present = next(r for r in rows if r['id'] == missing['id'].rsplit(':', 1)[0] + ':lookup')
                self.assertEqual(present['question'], missing['question'])
                self.assertIsNone(missing['evidence'])
                self.assertEqual(missing['target'], 'UNKNOWN')
                self.assertIsNotNone(present['evidence'])

    def test_decisions_follow_values_and_equal_is_sufficient(self):
        rows = {r['id']: r for group in build_data().values() for r in group}
        expected = {
            'train:A31:14:en:deadline_days_short': 'No; 14 days',
            'train:A31:14:de:deadline_weeks_equal': 'Ja; 14 Tage',
            'dev:E75:35:en:deadline_weeks_long': 'Yes; 35 days',
            'test:G17:52:de:deadline_days_short': 'Nein; 52 Tage',
            'test:G17:52:en:buffer': '56 days',
            'test:G17:52:de:missing': 'UNKNOWN',
        }
        for key, target in expected.items():
            self.assertEqual(rows[key]['target'], target)
        self.assertEqual(duration(21, 'de', 'weeks'), 'drei Wochen')
        self.assertEqual(duration(27, 'en', 'weeks'), 'three weeks and 6 days')

    def test_dialogue_uses_current_evidence_and_separates_history_values(self):
        data = build_data()
        for rows in data.values():
            for row in rows:
                if row['task'] in ('followup_lookup', 'followup_buffer'):
                    self.assertEqual(len(row['history']), 1)
                    self.assertNotIn(str(row['evidence']['lead_time_days']), row['history'][0]['content'])
                elif row['task'] == 'stale_followup':
                    self.assertEqual(len(row['history']), 2)
                    self.assertNotEqual(row['assessment']['prior_value'], row['evidence']['lead_time_days'])
                    self.assertNotEqual(row['target'], str(row['assessment']['prior_value']) + (' days' if row['language'] == 'en' else ' Tage'))
        rows = {r['id']: r for r in data['test']}
        self.assertEqual(rows['test:G17:52:en:followup_buffer']['target'], '56 days')
        stale = rows['test:G17:52:de:stale_followup']
        self.assertEqual(stale['history'][1]['content'], 'Fr\u00fchere Sch\u00e4tzung: 56 Tage.')
        self.assertEqual(stale['target'], '52 Tage')


if __name__ == '__main__':
    unittest.main()
