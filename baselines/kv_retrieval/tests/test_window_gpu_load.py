import unittest

from observe_window_gpu_load import parse_pmon


class ProcessLoadTest(unittest.TestCase):
    def test_missing_activity_stays_unknown_and_processes_keep_identity(self):
        rows = parse_pmon('''# gpu pid type sm mem enc dec jpg ofa fb ccpm command
# Idx # C/G % % % % % % MB MB name
0 1658 G - - - - - - 4 0 Xorg
0 1798427 C 43 9 - - - - 54412 0 python
''')
        self.assertIsNone(rows[0]['sm'])
        self.assertEqual(rows[1]['pid'], 1798427)
        self.assertEqual(rows[1]['sm'], 43)
        self.assertEqual(rows[1]['fb'], 54412)
        self.assertEqual(rows[1]['command'], 'python')


if __name__ == '__main__':
    unittest.main()
