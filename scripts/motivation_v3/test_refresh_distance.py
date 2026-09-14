import unittest
import numpy as np
from refresh_distance_controls import matched_stale, distance_bins


class RefreshDistanceTest(unittest.TestCase):
    def test_stale_keeps_identity_and_refresh_matches_surviving_budget(self):
        # Block 9 disappeared. Do not fill its slot using a new score.
        rows=np.array([[0.,1.,0.],[0.,1.,0.]])
        d=matched_stale(rows,np.array([0.,1.,0.]),[1,9],4)
        self.assertEqual(d['surviving_selected'],1)
        self.assertEqual(d['lost_selected'],1)
        self.assertEqual(d['stale'],0)
        self.assertEqual(d['refreshed'],1)
        self.assertEqual(d['gain'],1)
        self.assertIsNone(matched_stale(rows,np.ones(3),[9],4))

    def test_distance_does_not_include_partial_bins(self):
        d=distance_bins(np.ones(40),np.ones(40,dtype=bool))
        self.assertEqual(d[0],1)
        self.assertIsNone(d[1])


if __name__ == '__main__': unittest.main()
