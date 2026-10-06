"""Original native eight, bound only to the freshly restored query producer."""
from pathlib import Path
import types

import native_control_model_queue as original
import resource_window_resume_scope as scope

ORIGINAL_SHA = 'bbb27079bc0d8cb508dbc532e8fe725d807630a03c1efde3382b9c75afcf1c05'
scope.stable.pinned(original.__file__, ORIGINAL_SHA)
source = Path(original.__file__).read_text()
for before, after in (
    ('import stable_resource_deferred as deferred', 'import resource_window_deferred as deferred'),
    ("MODULE = 'native_control_model_queue'", "MODULE = 'resource_window_native'"),
    ("PARENT_SHA = '26426d1191712e87e40e1d91ac81403b68fa11bd48b8c88b82c5e90e2f0660e6'",
     'PARENT_SHA = owned.digest(deferred.__file__)'),
):
    source = scope.replace_once(source, before, after)
NAMESPACE = types.ModuleType('qualified_resource_window_native')
NAMESPACE.__file__ = __file__
exec(compile(source, original.__file__, 'exec'), NAMESPACE.__dict__)
NAMESPACE.HELPERS = tuple(dict.fromkeys((*original.HELPERS, *scope.OWN)))


def qualified(path):
    original.qualification(scope.qualification(path, 'native'))


NAMESPACE.qualification = qualified
for name, value in vars(NAMESPACE).items():
    if not name.startswith('__'):
        globals()[name] = value

if __name__ == '__main__':
    scope.check_audit(actual=True)
    NAMESPACE.main()
