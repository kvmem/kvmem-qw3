"""Sustained headroom for the unchanged mode-specific KVMem query engine."""
import inspect
from pathlib import Path

import resource_query_startup as original
import stable_resource_startup as admission


def source(parent):
    text = original.source(parent)
    anchor = 'deadline = time.monotonic() + self.args.resource_wait_seconds'
    lines = [line for line in text.splitlines() if line.strip() == anchor]
    if len(lines) != 1:raise admission.image.SafetyHalt('Frozen query resource deadline changed')
    line = lines[0];indent = line[:-len(line.lstrip())]
    text = text.replace(line, line + '\n' + indent + 'admission = HeadroomWindow()', 1)
    for before, after in (
        ('self.checkpoint(last_gpu_snapshot=snapshot)',
         'self.checkpoint(last_gpu_snapshot=snapshot, resource_admission=admission.update(snapshot, self.args.min_free_mib))'),
        ('if snapshot["free_mib"] >= self.args.min_free_mib:', 'if admission.ready:'),
    ):
        if text.count(before) != 1:raise admission.image.SafetyHalt('Frozen query sampling anchor changed')
        text = text.replace(before, after, 1)
    return text


def guarded_controller(parent):
    namespace = dict(parent.fresh_refresh_engine.__globals__, HeadroomWindow=admission.HeadroomWindow)
    exec(compile(source(parent), __file__, 'exec'), namespace)
    guarded = admission.guarded_controller(parent)

    class StableQueryController(guarded):
        fresh_refresh_engine = namespace['fresh_refresh_engine']

        def __init__(self, *values):
            super().__init__(*values)
            self.state['stable_query_startup_wrapper_sha256'] = admission.image.owned.digest(Path(__file__))

    return StableQueryController
