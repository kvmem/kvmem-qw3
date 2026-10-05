"""Build an independent SDK51 native keyboard candidate; never edit live code."""
import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    'qw3_computer/desktop.py': 'b504edbd0f178febcae1f95cc113d14ff4c4febba1a8b99cf1b02401c71f75f0',
    'qw3_computer/linux_driver.py': '949aab700b326dc9d340cdbfdfa73747a85b6828c84fdf9c7b28f497afc94ff4',
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Frozen SDK51 source anchor changed')
    return text.replace(old, new, 1)


def patch_file(name, original):
    if hashlib.sha256(original).hexdigest() != EXPECTED.get(name):
        raise ValueError('Use the exact frozen SDK51 source: ' + name)
    text = original.decode()
    if name.endswith('/linux_driver.py'):
        text = replace_once(text, 'from .device_protocol import capabilities\n',
            'from .device_protocol import capabilities\n'
            'from .native_window_adapter import observe_native_window\n'
            'from .native_keyboard_execution import dispatch_expectation, guarded_input_script, confirmed_native_rejection\n')
        text = replace_once(text, '    async def observe(self):\n',
            '    async def observe(self):\n'
            '        return await observe_native_window(self, LinuxDriver._observe_without_native_window)\n\n'
            '    async def _observe_without_native_window(self):\n')
        text = replace_once(text, "        self.expected_foreground=observation['foreground']\n",
            "        self.expected_foreground=observation['foreground']\n"
            "        self.expected_native_window=dispatch_expectation(observation)\n")
        text = replace_once(text, '        result=await self.execute(prefix+code)\n',
            "        source,nonce=guarded_input_script(prefix+code,getattr(self,'expected_native_window',None),kind) if kind!='open_app' else (prefix+code,None)\n"
            "        result=await self.execute(source)\n"
            "        rejection=confirmed_native_rejection(result,nonce)\n"
            "        if rejection is not None:\n"
            "            raise DeviceRejected('Native window, process or raw input focus changed before dispatch; no desktop input sent. Clipboard prepared: '+str(rejection['clipboard_prepared']))\n")
    else:
        text = replace_once(text, 'from .targeting import TargetError, identify_controls\n',
            'from .targeting import TargetError, identify_controls\n'
            'from .native_keyboard_execution import keyboard_revalidation_evidence\n')
        anchor = "                revalidation={'source_observation_id':old['observation_id'],'fresh_observation_id':fresh['observation_id'],\n"
        text = replace_once(text, anchor,
            "                native_keyboard_evidence=keyboard_revalidation_evidence(self.runtime,old,fresh,arguments['action']['kind']) if aged and same_device else None\n"
            "                native_keyboard_valid=native_keyboard_evidence is not None\n" + anchor)
        text = replace_once(text, "                    'foreground_context_unchanged':bool(keyboard_context_valid),\n",
            "                    'foreground_context_unchanged':bool(keyboard_context_valid),\n"
            "                    'native_keyboard_context_unchanged':native_keyboard_valid,\n"
            "                    'native_keyboard_evidence':native_keyboard_evidence,\n")
        text = replace_once(text, "                    'scope':('Explicit allowed app intent' if independent_app_intent else\n",
            "                    'scope':('Exact native client pixels, process generation and raw focus; unchanged observed control evidence; immediate native dispatch gate required' if native_keyboard_valid else\n"
            "                             'Explicit allowed app intent' if independent_app_intent else\n")
        text = replace_once(text,
            '                allowed=same_view or independent_app_intent or visual_context_valid or visual_window_valid or keyboard_context_valid\n',
            '                allowed=same_view or independent_app_intent or visual_context_valid or visual_window_valid or keyboard_context_valid or native_keyboard_valid\n')
    compile(text, 'isolated-native-keyboard-' + name, 'exec')
    return text.encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build_candidate(source_framework, acceptance, output, *, auxiliary_framework=None):
    source_framework, acceptance, output = map(lambda p:Path(p).resolve(), (source_framework, acceptance, output))
    declaration = json.loads(acceptance.read_text())
    files = declaration['framework_files']
    if any(files.get(name) != expected for name, expected in EXPECTED.items()):
        raise ValueError('Declare the exact SDK51 framework inventory')
    if output == source_framework or output.is_relative_to(source_framework) or source_framework.is_relative_to(output):
        raise ValueError('Keep the candidate outside the immutable framework')
    contents = {}
    for name, expected in files.items():
        path = Path(name)
        if (len(path.parts) != 2 or path.parts[0] not in {'qw3_computer','osworld_linux','osworld_windows'}
                or path.suffix not in {'.py','.html','.json','.tex'}):
            raise ValueError('Invalid frozen framework inventory path')
        source = source_framework / name
        if not source.is_file() and path.parts[0] != 'qw3_computer' and auxiliary_framework is not None:
            source = Path(auxiliary_framework).resolve() / name
        data = source.read_bytes()
        if digest(data) != expected:
            raise ValueError('Frozen framework changed: ' + name)
        contents[name] = data
    auxiliary = declaration.get('framework_auxiliary_files', {})
    for name, expected in auxiliary.items():
        if Path(name).name != name or Path(name).suffix != '.py':
            raise ValueError('Invalid auxiliary helper filename')
        data = (source_framework / name).read_bytes()
        if digest(data) != expected:
            raise ValueError('Frozen auxiliary helper changed: ' + name)
        contents[name] = data
    entry = {'device_gateway_entry.py':declaration['canonical_entry_sha256']}
    data = (source_framework / 'device_gateway_entry.py').read_bytes()
    if digest(data) != entry['device_gateway_entry.py']:
        raise ValueError('Frozen canonical gateway entry changed')
    contents['device_gateway_entry.py'] = data
    for name in EXPECTED:
        contents[name] = patch_file(name, contents[name])
    helpers = {}
    for name in ('native_window_evidence.py','native_window_adapter.py','native_keyboard_execution.py'):
        data = Path(__file__).with_name(name).read_bytes()
        helpers[name] = digest(data)
        text = data.decode()
        for module in ('native_window_adapter','native_window_evidence'):
            text = text.replace('from ' + module + ' import ', 'from .' + module + ' import ')
        compile(text, 'candidate-helper-' + name, 'exec')
        contents['qw3_computer/' + name] = text.encode()
    output.mkdir(parents=True, exist_ok=False)
    hashes, archives = {}, {}
    for name, data in contents.items():
        if name.startswith(('osworld_linux/','osworld_windows/')):
            # Runner root discovery depends on its original path. Archive exact
            # sources, and retain their original runtime location in protocols.
            target = output / 'source-archive' / name
            archives[name] = digest(data)
        else:
            target = output / 'framework' / name
            hashes[name] = digest(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    report = dict(kind='isolated_sdk51_native_keyboard_execution_candidate', source_framework=str(source_framework),
        source_acceptance=str(acceptance), source_acceptance_sha256=digest(acceptance.read_bytes()),
        source_files_sha256=files, source_auxiliary_files_sha256=auxiliary,
        source_canonical_entry_sha256=entry['device_gateway_entry.py'],
        auxiliary_source_framework=str(Path(auxiliary_framework).resolve()) if auxiliary_framework is not None else None,
        recipe_sha256=digest(Path(__file__).read_bytes()),helper_source_sha256=helpers,
        candidate_files_sha256=hashes, unchanged_runner_sources_sha256=archives,
        changed_package_files=[name for name in hashes if hashes[name] != (files | auxiliary | entry).get(name)],
        original_runners_not_relocated=True,model_requests=0,device_inputs=0,real_model_retest=False,
        qualified_for_production=False,frozen_framework_changed=False,
        scope='Additional aged key/type proof from screenshot-bracketed native window/process/raw-focus and exact entire client pixels. Native pre-dispatch gate in the same guest process. No clipboard paste retry, no age/budget change, no inferred missing controls or altered completion grades.',
        limitations=['Changes inside the client still reject this extra path, including the earlier Save-button case.',
                     'No additional click/control-click/drag/scroll admission is granted.',
                     'Original paths remain as before when native evidence is unavailable; Windows and Android source bytes are unchanged.'])
    (output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-framework', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--auxiliary-framework', type=Path)
    args = parser.parse_args()
    result = build_candidate(args.source_framework,args.acceptance,args.output,auxiliary_framework=args.auxiliary_framework)
    print(json.dumps({key:result[key] for key in ('kind','changed_package_files','real_model_retest')}))


if __name__ == '__main__':
    main()
