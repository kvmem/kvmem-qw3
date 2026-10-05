"""Use the same bounded admission wait for the frozen KVMem query supervisor."""
import inspect
from pathlib import Path
import textwrap

import resource_startup as off

QUERY_SHA = 'dfc098cb20b2a28f794a31daf5337b2d34f24ba5647a78c6ddb6ca270af313d2'


def source(original):
    off.image.pinned(off.image.ROOT / 'apps/computer_use/query_refresh_replay_queue.py', QUERY_SHA)
    text = textwrap.dedent(inspect.getsource(original.fresh_refresh_engine))
    for before, after in (
        ('runtime = self.args.protocol.parent / "engines" / row["id"]',
         'runtime = Path(self.state["engine_evidence_root"]) / row["id"]'),
        ('"--lock-timeout", "5"]', '"--lock-timeout", "60"]'),
    ):
        if text.count(before) != 1:
            raise off.image.SafetyHalt('Frozen KVMem query startup anchor changed')
        text = text.replace(before, after, 1)
    return text


def guarded_controller(original):
    namespace = dict(original.fresh_refresh_engine.__globals__)
    exec(compile(source(original), __file__, 'exec'), namespace)
    guarded_off = off.guarded_controller(original)

    class ResourceQueryController(guarded_off):
        fresh_refresh_engine = namespace['fresh_refresh_engine']

        def __init__(self, *values):
            super().__init__(*values)
            self.state['resource_query_startup_wrapper_sha256'] = off.image.owned.digest(Path(__file__))

    return ResourceQueryController
