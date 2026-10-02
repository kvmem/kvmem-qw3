import copy
import json
from pathlib import Path
import tempfile
import unittest

from osworld_windows.site_platform import verify
from osworld_windows.source import sha256


class QualifiedSitePlatformTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        evidence = self.root / 'native.json'
        evidence.write_text(json.dumps(dict(passed=True, final_clean_restore=True, render={'passed': True})))
        routing = self.root / 'routing.conf'
        routing.write_text('reviewed offline routing')
        self.mount = dict(source=str(routing), destination='/etc/nginx/conf.d/default.conf', rw=False)
        self.name = 'qw3-osworld-windows-gateway-1'
        self.frozen = dict(kind='qualified_windows_private_site_platform', qualified=True,
            windows_qualification='native.json', files={'native.json': sha256(evidence), 'routing.conf': sha256(routing)},
            containers={self.name: dict(image='sha256:qualified', service='gateway', bind_mounts=[self.mount])})
        self.manifest = self.root / 'platform.json'
        self.manifest.write_text(json.dumps(self.frozen))
        self.container = dict(Name='/'+self.name, Image='sha256:qualified', State={'Running': True},
            Config={'Labels': {'com.docker.compose.project': 'qw3-osworld-windows', 'com.docker.compose.service': 'gateway'}},
            Mounts=[self.mount | dict(Type='bind', Source=self.mount['source'], Destination=self.mount['destination'], RW=False)])

    def check(self, containers=None):
        return verify(containers or [self.container], self.manifest, root=self.root)

    def test_known_runtime_passes_and_records_content_hash(self):
        result = self.check()
        self.assertTrue(result['qualified'])
        self.assertEqual(result['sha256'], sha256(self.manifest))

    def test_moving_image_tag_or_stopped_container_cannot_reuse_qualification(self):
        for update in [dict(Image='sha256:other'), dict(State={'Running': False})]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.check([self.container | update])

    def test_base_compose_routing_or_writable_mount_is_rejected(self):
        for key, value in [('Source', '/old/routing.conf'), ('RW', True)]:
            changed = copy.deepcopy(self.container)
            changed['Mounts'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check([changed])

    def test_modified_routing_or_render_receipt_is_rejected(self):
        for name in ['routing.conf', 'native.json']:
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(b'changed')
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check()
            path.write_bytes(original)

    def test_other_project_or_extra_duplicate_container_is_rejected(self):
        changed = copy.deepcopy(self.container)
        changed['Config']['Labels']['com.docker.compose.project'] = 'another-project'
        for values in [[changed], [self.container, self.container]]:
            with self.assertRaises(ValueError):
                self.check(values)

    def test_failed_browser_qualification_rejected_even_with_matching_hash(self):
        path = self.root / 'native.json'
        path.write_text(json.dumps(dict(passed=False, final_clean_restore=True, render={'passed': False})))
        self.frozen['files']['native.json'] = sha256(path)
        self.manifest.write_text(json.dumps(self.frozen))
        with self.assertRaises(ValueError):
            self.check()


if __name__ == '__main__':
    unittest.main()
