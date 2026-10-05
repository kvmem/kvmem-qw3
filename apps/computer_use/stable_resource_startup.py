"""Require sustained GPU headroom before the existing owned model launcher."""
import inspect
from pathlib import Path
import textwrap
import time

import resource_startup as original
from resource_startup import image, source_identity, STARTUP_WAIT_SECONDS, ORIGINAL_FRESH_ENGINE

STABLE_SECONDS = 30


class HeadroomWindow:
    def __init__(self, seconds=STABLE_SECONDS, clock=None):
        self.seconds, self.clock = seconds, clock or time.monotonic
        self.since = self.last = None
        self.ready = False

    def update(self, snapshot, minimum):
        now = self.clock()
        if self.last is not None and now < self.last:
            raise image.SafetyHalt('Resource admission monotonic clock moved backwards')
        free = snapshot['free_mib']
        if type(free) is not int or type(minimum) is not int or minimum < 1:
            raise image.SafetyHalt('Require actual integer GPU headroom')
        self.last = now
        if free >= minimum:
            if self.since is None:self.since = now
        else:self.since = None
        elapsed = 0 if self.since is None else now - self.since
        self.ready = self.since is not None and elapsed >= self.seconds
        return dict(actual_free_mib=free, min_free_mib=minimum,
                    positive_headroom=self.since is not None, positive_seconds=elapsed,
                    required_positive_seconds=self.seconds, ready=self.ready,
                    snapshot_values_changed=False)


def engine_source(source):
    source = original.engine_source(source)
    anchor = 'deadline = time.monotonic() + self.args.resource_wait_seconds'
    lines = [line for line in source.splitlines() if line.strip() == anchor]
    if len(lines) != 1:raise image.SafetyHalt('Frozen resource deadline changed')
    line = lines[0];indent = line[:-len(line.lstrip())]
    source = source.replace(line, line + '\n' + indent + 'admission = HeadroomWindow()', 1)
    for before, after in (
        ('self.checkpoint(last_gpu_snapshot=snapshot)',
         'self.checkpoint(last_gpu_snapshot=snapshot, resource_admission=admission.update(snapshot, self.args.min_free_mib))'),
        ('if snapshot["free_mib"] >= self.args.min_free_mib:', 'if admission.ready:'),
    ):
        if source.count(before) != 1:raise image.SafetyHalt('Frozen resource sampling anchor changed')
        source = source.replace(before, after, 1)
    return source


def guarded_controller(parent):
    source_identity()
    namespace = dict(ORIGINAL_FRESH_ENGINE.__globals__, HeadroomWindow=HeadroomWindow)
    exec(compile(engine_source(textwrap.dedent(inspect.getsource(ORIGINAL_FRESH_ENGINE))),
                 __file__, 'exec'), namespace)
    guarded = original.guarded_controller(parent)

    class StableController(guarded):
        fresh_engine = namespace['fresh_engine']

        def __init__(self, *values):
            super().__init__(*values)
            self.state.update(resource_headroom_stability_seconds=STABLE_SECONDS,
                              stable_resource_wrapper_sha256=image.owned.digest(Path(__file__)))

    return StableController
