import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import condition_schema_candidate as candidate


class PrivateCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = candidate.RUNTIME/'native-control-candidate-20261005-02/build.json'
        cls.temporary = tempfile.TemporaryDirectory(dir=candidate.RUNTIME, prefix='condition-schema-cpu-fixture-')
        cls.output = Path(cls.temporary.name)/'candidate'
        cls.build = candidate.build(cls.base, cls.output)
        cls.package = '_qualified_condition_schema_cpu_fixture'
        package = types.ModuleType(cls.package);package.__path__ = [str(cls.output/'framework/qw3_computer')]
        sys.modules[cls.package] = package
        spec = importlib.util.spec_from_file_location(cls.package+'.protocol', cls.output/'framework/qw3_computer/protocol.py')
        cls.protocol = importlib.util.module_from_spec(spec);sys.modules[spec.name] = cls.protocol;spec.loader.exec_module(cls.protocol)
        cls.schema = sys.modules[cls.package+'.tool_schema']

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == cls.package or name.startswith(cls.package+'.'):sys.modules.pop(name)
        cls.temporary.cleanup()

    def test_full_candidate_only_changes_converter_and_adds_helper(self):
        self.assertEqual(self.build['changed_package_files'], ['qw3_computer/protocol.py', 'qw3_computer/condition_schema_wire.py'])
        for name, digest in self.build['candidate_files_sha256'].items():
            self.assertEqual(candidate.digest(self.output/'framework'/name), digest)
            if name not in self.build['changed_package_files']:
                self.assertEqual(digest, self.build['base_candidate_files_sha256'][name])
        self.assertEqual(candidate.digest(self.base), candidate.BASE_BUILD_SHA)

    def request(self, platform, workflow):
        catalog = self.schema.tools_for_platform(platform, workflow) + (self.schema.WORKFLOW_TOOLS if workflow else [])
        return {'model': 'qw3-computer', 'stream': True, 'reasoning': {'effort': 'medium'},
            'input': 'Fixture only', 'tools': [{'type': 'function', 'name': 'mcp__computer__'+t['name'],
                'parameters': copy.deepcopy(t['inputSchema'])} for t in catalog]}

    def test_private_converter_repairs_three_platforms_and_leaves_plain_modes_unchanged(self):
        declared = self.schema.WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']
        for platform in ('windows', 'linux', 'android'):
            for workflow in (False, True):
                with self.subTest(platform=platform, workflow=workflow):
                    request = self.request(platform, workflow)
                    if workflow:request['tools'][1]['parameters']['properties']['step']['properties']['conditions']['items'] = {}
                    before = copy.deepcopy(request)
                    converted = self.protocol.convert_request(request, 'qw3', 16384, allow_thinking=True)
                    self.assertEqual(request, before)
                    actual = converted.body['tools'][1]['function']['parameters']
                    if workflow:self.assertEqual(actual['properties']['step']['properties']['conditions'], declared)
                    else:self.assertEqual(actual, request['tools'][1]['parameters'])
                    for index in [0] + ([2] if workflow else []):
                        self.assertEqual(converted.body['tools'][index]['function']['parameters'], request['tools'][index]['parameters'])
                    self.assertEqual(converted.body['max_tokens'], 16384)
                    self.assertTrue(converted.body['enable_thinking'])

    def test_converter_refuses_changed_nested_definition(self):
        request = self.request('linux', True)
        request['tools'][1]['parameters']['properties']['step']['properties']['conditions']['items'] = {'type': 'object'}
        with self.assertRaises(ValueError):self.protocol.convert_request(request, 'qw3', 16384, allow_thinking=True)

    def test_actual_preserved_android_normalization_restores_only_trusted_conditions(self):
        capture = candidate.RUNTIME/'condition-schema-transport-20261006-02/responses-raw-6.json'
        request = json.loads(capture.read_text())
        before = copy.deepcopy(request)
        converted = self.protocol.convert_request(request, 'qw3', 16384, allow_thinking=True)
        actual = next(t['function']['parameters'] for t in converted.body['tools'] if t['function']['name']=='mcp__computer__act')
        declared = self.schema.WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']
        self.assertEqual(actual['properties']['step']['properties']['conditions'], declared)
        self.assertEqual(request, before)
        damaged = copy.deepcopy(request)
        for tool in damaged['tools']:
            if tool.get('type') == 'namespace':
                for child in tool['tools']:
                    if tool['name']+'__'+child['name'] == 'mcp__computer__act':
                        child['parameters']['properties']['step']['properties']['conditions']['items']['oneOf'][2]['required'].remove('value')
        with self.assertRaises(ValueError):self.protocol.convert_request(damaged, 'qw3', 16384, allow_thinking=True)

    def test_wrong_source_and_output_outside_runtime_are_rejected(self):
        with self.assertRaises(ValueError):candidate.patch_protocol(b'changed converter')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'not-created'
            with self.assertRaises(ValueError):candidate.build(self.base, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':unittest.main()
