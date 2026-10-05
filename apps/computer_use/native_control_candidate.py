"""Build a separate native-control candidate from the qualified keyboard package."""
import argparse
import hashlib
import json
from pathlib import Path


BASE_BUILD_SHA256 = '0314d1d331746c5cade72f44b8d5d0e74183a148c94a31a774fcae91e6cd8b72'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Qualified native-keyboard source anchor changed')
    return source.replace(before, after, 1)


def patch_source(name, data):
    source = data.decode()
    if name == 'qw3_computer/linux_driver.py':
        source = once(source, 'from .device_protocol import capabilities\n',
            'from .device_protocol import capabilities\n'
            'from .field_identity import provider_identity_from_attrs\n'
            'from .native_control_evidence import atspi_button_script, button_metadata_from_attrs\n'
            'from .native_control_execution import control_dispatch_expectation, execute_native_control\n'
            'from .targeting import identify_controls\n')
        source = once(source, "                         'value':node.text if node.text is not None else a.get('value')})\n",
            "                         'value':node.text if node.text is not None else a.get('value'),\n"
            "                         'provider_identity':provider_identity_from_attrs(a),\n"
            "                         'native_button_action':button_metadata_from_attrs(a)})\n")
        source = once(source,
            "            result=await self.execute(foreground_tree_script(foreground['title']),read_timeout=4)\n",
            "            result=await self.execute(atspi_button_script(foreground_tree_script(foreground['title'])),read_timeout=4)\n")
        source = once(source, '    async def start(self):\n',
            "        self.capabilities=self.capabilities | {'actions':self.capabilities['actions']+['click_control']}\n\n"
            '    async def start(self):\n')
        source = once(source, "        self.expected_native_window=dispatch_expectation(observation)\n",
            "        self.expected_native_window=dispatch_expectation(observation)\n"
            "        self.expected_native_button=None\n"
            "        if action['kind']=='click_control':\n"
            "            bound=identify_controls(observation)\n"
            "            matches=[c for c in bound['accessibility']['controls'] if c.get('control_id')==action['control_id']]\n"
            "            if len(matches)!=1:raise DeviceRejected('Native control is not uniquely observed; no input sent')\n"
            "            self.expected_native_button=control_dispatch_expectation(bound,matches[0])\n"
            "            if self.expected_native_button is None:raise DeviceRejected('Positive native button evidence unavailable; no input sent')\n")
        source = once(source, "        kind=action['kind']\n",
            "        kind=action['kind']\n"
            "        if kind=='click_control':\n"
            "            status,receipt=await execute_native_control(self,self.expected_native_button)\n"
            "            if status=='rejected':raise DeviceRejected('Native button or context changed before invocation; no input sent')\n"
            "            return {'backend':'atspi-native-button','native_invocation':receipt,'business_completion_verified':False}\n")
    elif name == 'qw3_computer/desktop.py':
        source = once(source, 'from .targeting import TargetError, identify_controls\n',
            'from .targeting import TargetError, identify_controls\n'
            'from .native_control_execution import native_control_revalidation_evidence\n')
        anchor = "                revalidation={'source_observation_id':old['observation_id'],'fresh_observation_id':fresh['observation_id'],\n"
        source = once(source, anchor,
            "                native_control_evidence=native_control_revalidation_evidence(self.runtime,old,fresh,mapping['control']) if guarded_control and same_device and target_valid else None\n"
            "                native_control_valid=native_control_evidence is not None\n" + anchor)
        source = once(source, "                    'native_keyboard_evidence':native_keyboard_evidence,\n",
            "                    'native_keyboard_evidence':native_keyboard_evidence,\n"
            "                    'native_control_valid':native_control_valid,\n"
            "                    'native_control_evidence':native_control_evidence,\n")
        source = once(source,
            "                    'scope':('Exact native client pixels, process generation and raw focus; unchanged observed control evidence; immediate native dispatch gate required' if native_keyboard_valid else\n",
            "                    'scope':('Native button identity, action, target pixels and complete unchanged semantic controls; immediate native invocation gate' if native_control_valid else\n"
            "                             'Exact native client pixels, process generation and raw focus; unchanged observed control evidence; immediate native dispatch gate required' if native_keyboard_valid else\n")
        source = once(source, "                    allowed=target_valid and ((same_view or context_valid) if aged else True)\n",
            "                    allowed=target_valid and ((same_view or context_valid or native_control_valid) if aged else True)\n")
        source = once(source, "                dispatched=dispatched | {'desktop_id':fresh['desktop_id'],'observation_id':fresh['observation_id']}\n",
            "                dispatched=dispatched | {'desktop_id':fresh['desktop_id'],'observation_id':fresh['observation_id']}\n"
            "                if native_control_valid:\n"
            "                    dispatched=dispatched | {'action':{'kind':'click_control','control_id':native_control_evidence['fresh_control_id']}}\n"
            "                    mapping=mapping | {'dispatched_action':dispatched['action'],'dispatch_route':'atspi-native-button','native_evidence':native_control_evidence}\n")
        source = once(source, '                "x", "y", "width", "height", "resource_id"}}\n',
            '                "x", "y", "width", "height", "resource_id", "provider_identity", "native_button_action"}}\n')
    else:
        return data
    compile(source, 'private-native-control-' + name, 'exec')
    return source.encode()


def build_candidate(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    declaration = base / 'build.json'
    if digest(declaration.read_bytes()) != BASE_BUILD_SHA256:
        raise ValueError('Use the exact qualified native-keyboard build')
    report = json.loads(declaration.read_text())
    if output == base or output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError('Candidate must be separate from the immutable base')
    contents = {}
    for name, expected in report['candidate_files_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid base inventory path')
        data = (base / 'framework' / name).read_bytes()
        if digest(data) != expected:
            raise ValueError('Qualified base file changed: ' + name)
        contents[name] = patch_source(name, data)
    helpers = {}
    modules = ('field_identity', 'native_control_evidence', 'native_control_execution',
               'native_keyboard_execution', 'native_window_evidence')
    for name in ('field_identity.py', 'native_control_evidence.py', 'native_control_execution.py'):
        raw = Path(__file__).with_name(name).read_bytes()
        helpers[name] = digest(raw)
        source = raw.decode()
        for module in modules:
            source = source.replace('from ' + module + ' import ', 'from .' + module + ' import ')
        compile(source, 'private-native-control-helper-' + name, 'exec')
        contents['qw3_computer/' + name] = source.encode()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, data in contents.items():
        target = output / 'framework' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = digest(data)
    built = dict(kind='isolated_native_button_invocation_candidate',
                 base_build=str(declaration), base_build_sha256=BASE_BUILD_SHA256,
                 base_candidate_files_sha256=report['candidate_files_sha256'],
                 recipe_sha256=digest(Path(__file__).read_bytes()), helper_source_sha256=helpers,
                 candidate_files_sha256=hashes,
                 changed_package_files=[name for name in hashes if hashes[name] != report['candidate_files_sha256'].get(name)],
                 frozen_framework_changed=False, model_requests=0, device_inputs=0,
                 real_model_retest=False, qualified_for_production=False,
                 scope='Complete unchanged semantic evidence, unchanged exact button ROI and screenshot-bound native window/process/focus. Preserve public click_control and invoke the observed native object once. All visual mouse routes and outcome conditions remain unchanged.',
                 limitations=['Only complete foreground AT-SPI push-button evidence admits this route.',
                              'No name, coordinate or partial-tree identity fallback.',
                              'Invocation False and exceptions never imply confirmed non-delivery.',
                              'Windows, Android, schemas, execution budgets, graders and canonical entry bytes are unchanged.'])
    (output / 'build.json').write_text(json.dumps(built, indent=2) + '\n')
    return built


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();d=build_candidate(args.base,args.output)
    print(json.dumps({key:d[key] for key in ('kind','changed_package_files','real_model_retest')}))
