"""Reuse the frozen deferred controllers with a fresh, proven image parent."""
from pathlib import Path
from unittest.mock import patch

import stable_resource_deferred as original
import resource_window_resume_scope as scope


def qualified(path):
    original.qualification(scope.qualification(path, 'deferred'))


NAMESPACE = original.build_namespace()
NAMESPACE.__file__ = __file__
NAMESPACE.MODULE = 'resource_window_deferred'
NAMESPACE.FOUR_DECLARATION = scope.DECLARATION
NAMESPACE.HELPERS = tuple(dict.fromkeys((*original.HELPERS, *scope.OWN)))
NAMESPACE.qualification = qualified
for device in (NAMESPACE.BOUNDED, NAMESPACE.QUERY):
    device.__file__ = __file__
NAMESPACE.QUERY.BOUNDED_SHA = scope.owned.digest(__file__)
for name, value in vars(NAMESPACE).items():
    if not name.startswith('__'):
        globals()[name] = value


if __name__ == '__main__':
    scope.check_audit(actual=True)
    with patch.object(NAMESPACE.successors.q.old, 'IMAGE_VALIDATE',
                      original.successors.previous.previous.historical_image):
        NAMESPACE.main()
