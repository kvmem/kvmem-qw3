import unittest

from audit_window_gpu_overlap import Samples, annotate


class OverlapAudit(unittest.TestCase):
    def test_missing_observations_are_unknown_not_exclusive(self):
        request={'request_id':'r','received_at':100.,'request_wall_s':20.}
        result=annotate(request,1,Samples([], 'at'),Samples([], 'observed_at'))
        self.assertIsNone(result['other_compute_process_observed'])
        self.assertIsNone(result['other_compute_sm_activity_observed'])
        self.assertFalse(result['gpu_exclusivity_verified'])

    def test_residency_and_activity_are_different_observations(self):
        request={'request_id':'r','received_at':100.,'request_wall_s':20.}
        memory=Samples([{'at':110.,'gpu_processes_gib':{'1':30.,'2':20.}}], 'at')
        activity=Samples([{'observed_at':110.,'processes':[
            {'pid':1,'type':'C','sm':60},{'pid':2,'type':'C','sm':0},
            {'pid':3,'type':'G','sm':50}]}], 'observed_at')
        result=annotate(request,1,memory,activity)
        self.assertTrue(result['other_compute_process_observed'])
        self.assertFalse(result['other_compute_sm_activity_observed'])
        self.assertEqual(result['request_wall_s'],20.)
        self.assertFalse(result['latency_adjusted'])

    def test_only_inside_request_and_sampling_errors_stay_unknown(self):
        request={'request_id':'r','received_at':100.,'request_wall_s':20.}
        memory=Samples([{'at':99.,'gpu_processes_gib':{'2':20.}},
                        {'at':105.,'gpu_processes_gib':{'1':30.}}], 'at')
        activity=Samples([{'observed_at':110.,'sampling_error':'timeout'},
            {'observed_at':121.,'processes':[{'pid':2,'type':'C','sm':60}]}], 'observed_at')
        result=annotate(request,1,memory,activity)
        self.assertFalse(result['other_compute_process_observed'])
        self.assertEqual(result['max_gap_including_request_boundaries_s'],15.)
        self.assertEqual(result['activity_sampling_errors'],1)
        self.assertIsNone(result['other_compute_sm_activity_observed'])


if __name__=='__main__':unittest.main()
