"""Frozen condition arms after all eight original native tasks on the new route."""
from pathlib import Path
import types

import condition_schema_model_queue as original
import resource_window_resume_scope as scope

scope.stable.pinned(original.native.__file__, original.NATIVE_SHA)
source = Path(original.__file__).read_text()
for before, after in (
    ('import native_control_model_queue as native', 'import resource_window_native as native'),
    ("MODULE='condition_schema_model_queue'", "MODULE='resource_window_condition'"),
    ("NATIVE_SHA='bbb27079bc0d8cb508dbc532e8fe725d807630a03c1efde3382b9c75afcf1c05'",
     'NATIVE_SHA=owned.digest(native.__file__)'),
    ("if p['parent_declaration']!=protocol['required_original_native_queue_declaration']:",
     "if protocol['required_original_native_queue_declaration']!=ORIGINAL_NATIVE_PARENT or p['parent_declaration']['path']!=str(RESUMED_NATIVE_DECLARATION):"),
    ("parent_path=study.adapter.checked_ref(protocol['required_original_native_queue_declaration'])",
     'parent_path=RESUMED_NATIVE_DECLARATION'),
    ("if __name__=='__main__':", 'def main():'),
):
    source = scope.replace_once(source, before, after)
NAMESPACE = types.ModuleType('qualified_resource_window_condition')
NAMESPACE.__file__ = __file__
NAMESPACE.ORIGINAL_NATIVE_PARENT = scope.ref(scope.ORIGINAL_NATIVE_DECLARATION)
NAMESPACE.RESUMED_NATIVE_DECLARATION = scope.NATIVE_DECLARATION
exec(compile(source, original.__file__, 'exec'), NAMESPACE.__dict__)
NAMESPACE.HELPERS = tuple(dict.fromkeys((*original.HELPERS, *scope.OWN)))


def qualified(path):
    original.qualification(scope.qualification(path, 'condition'))


NAMESPACE.qualification = qualified
for name, value in vars(NAMESPACE).items():
    if not name.startswith('__'):
        globals()[name] = value

if __name__ == '__main__':
    scope.check_audit(actual=True)
    NAMESPACE.main()
