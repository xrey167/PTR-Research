import unittest
from research.reader_training import accumulation_windows


class AccumulationWindowsTest(unittest.TestCase):
    def test_partial_final_window_and_single_pass(self):
        self.assertEqual(list(accumulation_windows(iter(range(5)), 3)), [[0, 1, 2], [3, 4]])
        self.assertEqual(list(accumulation_windows([], 3)), [])

    def test_invalid_sizes(self):
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                list(accumulation_windows([1], size))


if __name__ == '__main__':
    unittest.main()
