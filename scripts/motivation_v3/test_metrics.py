import unittest
import numpy as np
from metrics import js, kl, windows, frozen_curves, analyze_steps, bootstrap
from schedule_diagnostics import schedules


class MetricsTest(unittest.TestCase):
    def test_single_trajectory_has_no_population_confidence_interval(self):
        self.assertEqual(bootstrap([.4])['ci95'],[None,None])

    def test_divergence_limits(self):
        self.assertAlmostEqual(js([1,0],[0,1]),1)
        self.assertAlmostEqual(js([.2,.8],[.2,.8]),0)
        self.assertAlmostEqual(kl([.2,.8],[.2,.8]),0)
        self.assertIsNone(js([0,0],[1,0]))

    def test_full_windows_only(self):
        a=np.arange(10)[:,None]
        self.assertEqual(len(windows(a,4)),2)
        self.assertEqual(len(windows(a,4,2)),4)

    def test_frozen_does_not_follow_future_attention(self):
        rows=np.array([[0,1],[0,1.]])
        d=frozen_curves(rows,np.array([1.,0]),ks=(1,),bins=2)['1']
        self.assertEqual(d['frozen'],[0,0])
        self.assertEqual(d['oracle'],[1,1])

    def test_skipped_step_is_not_boundary(self):
        steps=[dict(step=i,prompt_tokens=1024,positions=np.arange(1024,1152),
                    mass=np.ones((128,16))/16) for i in [1,3]]
        out=analyze_steps(steps,window=32,recent=0)
        self.assertEqual(out['cross_boundaries'],0)

    def test_periodic_update_cannot_see_current_or_future_queries(self):
        rows=np.array([[0.,1.],[0.,1.],[1.,0.],[1.,0.]])
        d=schedules(rows,np.array([1.,0.]),fraction=.5,intervals=(2,))
        self.assertEqual(d['periodic_2']['coverage'],0)
        self.assertEqual(d['token_oracle']['coverage'],1)
        self.assertEqual(d['periodic_2']['updates'],2)

    def test_matched_boundary_does_not_inherit_third_step_prefix_limit(self):
        steps=[dict(step=i,prompt_tokens=4096,positions=np.arange(4096,4224),
            mass=np.ones((128,34))/34,prefix_identity_verified=True,
            history_end_tokens=limit,common_prefix_tokens=limit) for i,limit in enumerate([4096,512,2048])]
        d=analyze_steps(steps,window=32,recent=0,identity_verified=True)
        self.assertEqual(d['matched_boundaries'][-1]['common_blocks'],15)


if __name__=='__main__':unittest.main()
