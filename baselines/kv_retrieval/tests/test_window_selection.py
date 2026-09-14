import unittest

from run_window_campaign import variant_order


class SelectedWindows(unittest.TestCase):
    def test_small_only_never_schedules_large_on_alternating_tasks(self):
        plan={'variants':{'small':{}}}
        self.assertEqual([variant_order(plan,i) for i in range(4)],[['small']]*4)

    def test_original_two_window_order_remains_supported(self):
        plan={'variants':{'large':{},'small':{}}}
        self.assertEqual(variant_order(plan,0),['large','small'])
        self.assertEqual(variant_order(plan,1),['small','large'])

    def test_empty_or_unknown_window_is_rejected(self):
        for variants in [{},{'typo':{}}]:
            with self.assertRaises(ValueError):variant_order({'variants':variants},0)


if __name__=='__main__':unittest.main()
