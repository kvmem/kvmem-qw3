import copy
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps/computer_use"))
from client_resources import summarize_client_resources, timestamp_ticks


def process(pid, parent, started, parent_started=None, cpu=1):
    return dict(pid=pid, parent_pid=parent, name='python', start_time_utc=started,
                sampled_start_time_utc=started, birth_ticks=str(timestamp_ticks(started)),
                parent_start_time_utc=parent_started,
                parent_birth_ticks=str(timestamp_ticks(parent_started)) if parent_started else None,
                cpu_seconds=cpu, working_set_bytes=100, peak_working_set_bytes=200, private_bytes=50)


def sample():
    t='2026-10-01T00:00:00Z'
    return dict(schema_version=2, app_pid=676, app_start_time_utc=t,
                app_birth_ticks=str(timestamp_ticks(t)), ownership_verified=True,
                snapshot_complete=True, omitted_processes=[], rejected_edges=[],
                processes=[process(676,5188,t),process(2000,676,'2026-10-01T00:00:01Z',t)])


class ClientResourceTests(unittest.TestCase):
    def test_valid_tree_and_cpu_lifetimes(self):
        first=sample(); second=copy.deepcopy(first)
        child=second['processes'][1]
        child.update(start_time_utc='2026-10-01T00:00:02Z',sampled_start_time_utc='2026-10-01T00:00:02Z',
                     birth_ticks=str(timestamp_ticks('2026-10-01T00:00:02Z')),cpu_seconds=3)
        result=summarize_client_resources([first,second])
        self.assertTrue(result['client_resource_comparison_valid'])
        self.assertEqual(result['client_sampled_cpu_seconds'],5)
        self.assertEqual(result['client_sum_working_sets_peak_bytes'],200)

    def test_legacy_cannot_publish_memory(self):
        data=sample(); del data['schema_version']
        result=summarize_client_resources([data])
        self.assertFalse(result['client_resource_comparison_valid'])
        self.assertIsNone(result['client_sum_working_sets_peak_bytes'])

    def test_pid_reuse_old_child_rejected(self):
        data=sample()
        data['processes'].append(process(692,676,'2026-09-30T00:00:00Z',data['app_start_time_utc']))
        result=summarize_client_resources([data])
        self.assertFalse(result['client_resource_comparison_valid'])
        self.assertIn('predates',result['client_resource_errors'][0]['error'])

    def test_cim_counter_race_rejected(self):
        data=sample(); data['processes'][1]['sampled_start_time_utc']='2026-10-01T00:00:02Z'
        self.assertFalse(summarize_client_resources([data])['client_resource_comparison_valid'])

    def test_incomplete_or_bad_counters_rejected(self):
        for v in (-1,float('nan'),float('inf'),True):
            with self.subTest(v=v):
                data=sample(); data['processes'][1]['cpu_seconds']=v
                self.assertFalse(summarize_client_resources([data])['client_resource_comparison_valid'])
        data=sample(); data['snapshot_complete']=False
        self.assertFalse(summarize_client_resources([data])['client_resource_comparison_valid'])

    def test_duplicate_and_orphan_and_parent_birth_rejected(self):
        for mutation in ('duplicate','orphan','wrong_parent_birth'):
            with self.subTest(mutation=mutation):
                data=sample()
                if mutation=='duplicate': data['processes'].append(copy.deepcopy(data['processes'][1]))
                elif mutation=='orphan': data['processes'][1]['parent_pid']=9999
                else: data['processes'][1]['parent_birth_ticks']=data['processes'][1]['birth_ticks']
                self.assertFalse(summarize_client_resources([data])['client_resource_comparison_valid'])

    def test_cycle_cannot_become_rooted_by_equal_births(self):
        data=sample(); t='2026-10-01T00:00:01Z'
        data['processes'][1:]=[process(2000,2001,t,t),process(2001,2000,t,t)]
        self.assertFalse(summarize_client_resources([data])['client_resource_comparison_valid'])

    def test_exclusion_reason_does_not_hide_missing_data(self):
        data=sample(); t='2026-09-30T00:00:00Z'
        data['rejected_edges']=[dict(pid=692,parent_pid=676,reason='child_predates_current_parent',
                                    birth_ticks=str(timestamp_ticks(t)),parent_birth_ticks=data['app_birth_ticks'])]
        self.assertTrue(summarize_client_resources([data])['client_resource_comparison_valid'])
        data['rejected_edges'][0]['reason']='creation_time_unavailable'
        self.assertFalse(summarize_client_resources([data])['client_resource_comparison_valid'])


if __name__=='__main__': unittest.main()
