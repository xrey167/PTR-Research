"""Check the experimental comparison, without loading a model."""
import unittest
from research.reader_deadline_cases import build_cases


class DeadlineCasesTest(unittest.TestCase):
    def test_factorial_and_scoring_boundary(self):
        rows = build_cases()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r['id'] for r in rows}), 12)
        for language in ('en', 'de'):
            for unit in ('days', 'weeks'):
                selected = [r for r in rows if r['assessment']['language'] == language
                            and r['assessment']['unit_wording'] == unit]
                self.assertEqual([r['assessment']['deadline_days'] for r in selected], [21, 27, 28])
                self.assertEqual([r['assessment']['expected_sufficient'] for r in selected], [False, True, True])
                self.assertEqual(len({str(r['reader_input']['history']) for r in selected}), 1)

    def test_original_failure_and_no_history_value(self):
        rows = build_cases()
        original = next(r for r in rows if r['id'] == 'de:weeks:short')
        self.assertEqual(original['reader_input']['question'], 'Reichen drei Wochen bis zur Lieferung?')
        for row in rows:
            self.assertEqual(set(row['reader_input']), {'question', 'history'})
            self.assertNotIn('27', str(row['reader_input']['history']))


if __name__ == '__main__':
    unittest.main()
