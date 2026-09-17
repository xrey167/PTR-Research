import copy
import unittest
from research.reader_prompt import render_segments, encode_training_row
from research.reader_training_data import build_data


class CharacterTokenizer:
    """Deterministic interface test only; does not claim Qwen token equivalence."""
    eos_token_id = 1000000

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return ''.join(m['role'] + '\n' + m['content'] + '\n' for m in messages) + 'assistant\n'

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))


class ReaderPromptTest(unittest.TestCase):
    def test_target_and_assessment_do_not_affect_prompt(self):
        row = build_data()['train'][0]
        altered = copy.deepcopy(row)
        altered.update(target='TARGET_SECRET', assessment={'answer': 'ASSESSMENT_SECRET'})
        tokenizer = CharacterTokenizer()
        self.assertEqual(render_segments(tokenizer, row), render_segments(tokenizer, altered))

    def test_only_answer_and_eos_are_supervised(self):
        tokenizer = CharacterTokenizer()
        row = build_data()['train'][0]
        encoded = encode_training_row(tokenizer, row, 10000)
        supervised = [n for n in encoded['labels'] if n != -100]
        self.assertEqual(supervised, list(map(ord, '14 days')) + [tokenizer.eos_token_id])
        self.assertEqual(encoded['input_ids'][-len(supervised):], supervised)
        self.assertTrue(all(n == -100 for n in encoded['labels'][:-len(supervised)]))
        with self.assertRaisesRegex(ValueError, 'truncation'):
            encode_training_row(tokenizer, row, len(encoded['input_ids']) - 1)

    def test_missing_evidence_does_not_disclose_hidden_value(self):
        row = next(r for r in build_data()['train'] if r['task'] == 'missing')
        prefix, suffix = render_segments(CharacterTokenizer(), row)
        self.assertIn('No fact is available.', prefix)
        self.assertNotIn('14 days', prefix + suffix)


if __name__ == '__main__':
    unittest.main()
