"""Build a private SDK51 browser-runtime revalidation candidate.

Frozen trial packages and protocols are never edited. Raw labels, completion
checks, age thresholds, receipts, budgets and driver input behavior are kept.
"""
import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {'desktop.py': 'b504edbd0f178febcae1f95cc113d14ff4c4febba1a8b99cf1b02401c71f75f0',
            'linux_driver.py': '949aab700b326dc9d340cdbfdfa73747a85b6828c84fdf9c7b28f497afc94ff4'}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Qualified SDK51 source anchor changed')
    return text.replace(old, new, 1)


def patch_file(name, original):
    if name not in EXPECTED or hashlib.sha256(original).hexdigest() != EXPECTED[name]:
        raise ValueError('Use the exact frozen SDK51 source: ' + name)
    text = original.decode()
    if name == 'linux_driver.py':
        text = replace_once(text, 'from .device_protocol import capabilities\n',
            'from .device_protocol import capabilities\nfrom .browser_runtime_metadata import native_browser_tree_script, metadata_from_attrs, PROVIDER\n')
        text = replace_once(text, "foreground_tree_script(foreground['title']),read_timeout=4",
            "native_browser_tree_script(foreground_tree_script(foreground['title']),foreground),read_timeout=4")
        text = replace_once(text, '        if len(controls)>80:break\n',
            "        if 'qw3-native-focused' in a:controls[-1]['native_focused']=a['qw3-native-focused']\n"
            "        runtime=metadata_from_attrs(a,controls[-1])\n"
            "        if runtime is not None:controls[-1]['browser_runtime_metadata']=runtime\n"
            '        if len(controls)>80:break\n')
        text = replace_once(text, "'provider':'official-osworld-create_atspi_node'", "'provider':PROVIDER")
    else:
        text = replace_once(text, 'from .targeting import TargetError, identify_controls\n',
            'from .targeting import TargetError, identify_controls\nfrom .browser_runtime_metadata import runtime_only_changes, valid_metadata\n')
        anchor = "                revalidation={'source_observation_id':old['observation_id'],'fresh_observation_id':fresh['observation_id'],\n"
        injected = '''                browser_runtime_evidence=runtime_only_changes(old,fresh) if aged and same_device else None
                browser_runtime_valid=False
                if browser_runtime_evidence and self.unchanged_foreground_window(old,fresh):
                    kind=arguments['action']['kind']
                    if kind in {'click','drag','scroll'}:
                        action=dispatched['action']
                        points=[(action['from_x'],action['from_y']),(action['to_x'],action['to_y'])] if kind=='drag' else [(action['x'],action['y'])]
                        browser_runtime_valid=self.unchanged_visual_foreground(old,fresh,points)
                    elif kind in {'key','type'}:
                        browser_runtime_valid=browser_runtime_evidence['positive_focus_object_unchanged']
                    elif guarded_control:
                        target=mapping['control']
                        native_target=valid_metadata(target,old['foreground'])
                        native_target_same=(native_target is not None and
                            {k:v for k,v in native_target.items() if k!='resource'}==browser_runtime_evidence['changes'][0]['source_identity'])
                        if target_valid or native_target_same:
                            browser_runtime_valid=self.unchanged_visual_foreground(old,fresh,[(target['x'],target['y']),
                                (target['x']+target['width']-1,target['y']+target['height']-1)])
                            if native_target_same and browser_runtime_valid:target_valid=True
'''
        text = replace_once(text, anchor, injected + anchor)
        text = replace_once(text, "                    'visual_foreground_context_unchanged':bool(visual_window_valid),\n",
            "                    'visual_foreground_context_unchanged':bool(visual_window_valid),\n"
            "                    'browser_runtime_metadata_context_unchanged':bool(browser_runtime_valid),\n"
            "                    'browser_runtime_metadata_evidence':browser_runtime_evidence,\n")
        text = replace_once(text, "                    'scope':('Explicit allowed app intent' if independent_app_intent else\n",
            "                    'scope':('Exact full-window pixels and native browser identity; only selected-tab runtime resource changed' if browser_runtime_valid else\n"
            "                             'Explicit allowed app intent' if independent_app_intent else\n")
        text = replace_once(text, '                allowed=same_view or independent_app_intent or visual_context_valid or visual_window_valid or keyboard_context_valid\n',
            '                allowed=same_view or independent_app_intent or visual_context_valid or visual_window_valid or keyboard_context_valid or browser_runtime_valid\n')
        text = replace_once(text, '                    allowed=target_valid and ((same_view or context_valid) if aged else True)\n',
            '                    allowed=target_valid and ((same_view or context_valid or browser_runtime_valid) if aged else True)\n')
    compile(text, 'isolated-browser-runtime-' + name, 'exec')
    return text.encode()


def build_candidate(source_package, acceptance, output, *, auxiliary_framework=None):
    source_package = Path(source_package).resolve()
    declared = json.loads(Path(acceptance).read_text()).get('framework_files', {})
    contents, auxiliaries = {}, {}
    if any(name.startswith('qw3_computer/') for name in declared):
        # The real acceptance inventories the entire framework, including the
        # unchanged task runners. Verify and retain those files too.
        for name, expected in declared.items():
            parts = Path(name).parts
            if (len(parts) != 2 or parts[0] not in {'qw3_computer', 'osworld_linux', 'osworld_windows'}
                    or Path(name).suffix not in {'.py', '.html', '.json', '.tex'}):
                raise ValueError('Invalid framework inventory path')
            source_path = source_package.parent / name
            if not source_path.is_file() and parts[0] != 'qw3_computer' and auxiliary_framework is not None:
                source_path = Path(auxiliary_framework) / name
            data = source_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Frozen framework changed: ' + name)
            if parts[0] == 'qw3_computer':
                contents[parts[1]] = data
            else:
                auxiliaries[name] = data
        files = {name.split('/', 1)[1]: digest for name, digest in declared.items()
                 if name.startswith('qw3_computer/')}
    else:
        files = declared
    if any(files.get(k) != v for k, v in EXPECTED.items()):
        raise ValueError('Declare exact SDK51 package inventory')
    for name, expected in files.items():
        if Path(name).name != name or Path(name).suffix not in {'.py', '.html', '.json', '.tex'}:
            raise ValueError('Invalid SDK51 package filename')
        data = (source_package / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Frozen package changed: ' + name)
        contents[name] = data
    for name in EXPECTED:
        contents[name] = patch_file(name, contents[name])
    helper = Path(__file__).with_name('browser_runtime_metadata.py')
    contents[helper.name] = helper.read_bytes()
    output = Path(output).resolve()
    if output == source_package or output.is_relative_to(source_package):
        raise ValueError('Keep candidate outside its frozen source')
    output.mkdir(parents=True, exist_ok=False)
    destination = output / 'framework/qw3_computer'
    destination.mkdir(parents=True)
    hashes = {}
    for name, data in contents.items():
        (destination / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    auxiliary_hashes = {}
    for name, data in auxiliaries.items():
        # Original runners compute their repository root from their path.
        # Archive exact sources for evidence; keep executing the pinned
        # original location rather than relocating them into the import path.
        target = output / 'source-archive/auxiliary' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        auxiliary_hashes[name] = hashlib.sha256(data).hexdigest()
    result = dict(kind='isolated_sdk51_native_browser_runtime_candidate',
        source_package=str(source_package), source_files_sha256=files,
        source_acceptance_sha256=hashlib.sha256(Path(acceptance).read_bytes()).hexdigest(),
        recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_files_sha256=hashes, changed_package_files=[n for n in hashes if hashes[n] != files.get(n)],
        unchanged_auxiliary_framework_files_sha256=auxiliary_hashes,
        auxiliary_source_root=str(Path(auxiliary_framework).resolve()) if auxiliary_framework is not None else None,
        auxiliary_sources_archived_without_relocating_runtime=True,
        model_calls=0, device_inputs=0, real_model_retest=False, native_qualified=False,
        qualified_for_production=False, frozen_packages_changed=False,
        raw_labels_and_completion_conditions_preserved=True,
        scope='Additional aged revalidation proof for native selected Chrome tab resource-only changes. Exact window pixels, source, task/device/session, focus and target bounds remain required. Keyboard needs positive native focus. No annotation means the original strict rule.')
    (output / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-package', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--auxiliary-framework', type=Path)
    a = parser.parse_args()
    r = build_candidate(a.source_package, a.acceptance, a.output, auxiliary_framework=a.auxiliary_framework)
    print(json.dumps({k: r[k] for k in ('kind', 'changed_package_files', 'real_model_retest')}))


if __name__ == '__main__':
    main()
