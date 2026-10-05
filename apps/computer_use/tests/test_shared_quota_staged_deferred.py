"""Retain full inventories and source-copy bounds for the staged parent."""
import importlib.util
from pathlib import Path
import unittest

import shared_quota_staged_deferred as adapter

spec = importlib.util.spec_from_file_location("retained_staged_deferred_tests", Path(__file__).with_name("test_shared_quota_handoff_deferred.py"))
retained = importlib.util.module_from_spec(spec); spec.loader.exec_module(retained)
retained.adapter = retained.retained.adapter = adapter
retained.retained.bounded_tests.queue = adapter.BOUNDED
retained.retained.query_tests.queue = adapter.QUERY


class BoundedQueueTests(retained.BoundedQueueTests):
    pass


class QueryQueueTests(retained.QueryQueueTests):
    pass


class FullHandoffTests(retained.FullHandoffTests):
    pass


class RoutingTests(retained.RoutingTests):
    pass


# Keep the retained port tests, targeting the unchanged admission helper's
# own globals rather than assuming a particular number of routing wrappers.
_namespace = dict(vars(retained), adapter=adapter)
_file = Path(retained.__file__).read_text()
_source = _file[_file.index('class SpawnTests('):_file.index("if __name__")].replace(
    'adapter.successors.previous.port', 'adapter.admission.successors.previous.port')
exec(compile(_source, __file__, 'exec'), _namespace)


class SpawnTests(_namespace['SpawnTests']):
    pass


if __name__ == '__main__':
    unittest.main()
