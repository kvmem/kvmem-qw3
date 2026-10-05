"""Resource admission for frozen trial controllers, without changing model profiles.

Reuse the exact original engine supervisor. Only the fresh evidence directory
and the launcher's existing bounded resource wait change. Waiting stays in one
owned launcher; model readiness, ownership, input guards and task budgets stay
in the original controller.
"""
import inspect
from pathlib import Path
import textwrap

import visual_context_retest_queue as image

IMAGE_SHA = 'debfb90eb88018575bb1554715a8e7a408a68aa93a531cd9c049594cd06aed24'
LAUNCHER_SHA = 'a912f87e3586962e984d775c887df2cff150bb244d473874c9cc71f659f73264'
STARTUP_WAIT_SECONDS = 60
ORIGINAL_FRESH_ENGINE = image.Controller.fresh_engine


def source_identity():
    image.pinned(Path(image.__file__), IMAGE_SHA)
    image.pinned(image.ROOT / 'apps/computer_use/run_engine.py', LAUNCHER_SHA)


def engine_source(source):
    replacements = (
        ('runtime = self.args.protocol.parent / "engines" / row["id"]',
         'runtime = Path(self.state["engine_evidence_root"]) / row["id"]'),
        ('"--lock-timeout", "5"]', '"--lock-timeout", "60"]'),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise image.SafetyHalt('Frozen resource-startup anchor changed')
        source = source.replace(before, after, 1)
    return source


def guarded_controller(original):
    source_identity()
    namespace = dict(ORIGINAL_FRESH_ENGINE.__globals__)
    exec(compile(engine_source(textwrap.dedent(inspect.getsource(ORIGINAL_FRESH_ENGINE))),
                 __file__, 'exec'), namespace)

    class ResourceController(original):
        fresh_engine = namespace['fresh_engine']

        def __init__(self, *values):
            super().__init__(*values)
            self.state.update(
                engine_evidence_root=str(self.args.state_dir.parent / 'engines'),
                resource_startup_wrapper_sha256=image.owned.digest(Path(__file__)),
                launcher_resource_wait_seconds=STARTUP_WAIT_SECONDS,
                resource_wait_does_not_change_task_budget=True,
            )

    return ResourceController
