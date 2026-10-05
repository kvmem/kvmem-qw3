"""Build a private wire-repair candidate without modifying the native study."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'apps/computer_use/.runtime'
BASE_BUILD_SHA = 'a07266aed63d31f470aa2b5fbcd60fc8def54a519b907e1eeb8fb1a4c8343c8a'
PROTOCOL_SHA = '8692e6d2a85a3bde2732af15b553fd8153a2ab6966e9e832c4a060ef89b1ffee'
SCHEMA_SHA = '6d7e24c2460f98f40db457952fd4c7445cb56cf01384d34b56a687617bc0c277'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_protocol(data):
    if hashlib.sha256(data).hexdigest() != PROTOCOL_SHA:
        raise ValueError('Require the exact frozen native protocol converter')
    text = data.decode()
    old = 'from dataclasses import dataclass\n'
    new = old + ('from .condition_schema_wire import restore_inline_conditions\n'
                 'from .tool_schema import WORKFLOW_TOOLS\n')
    if text.count(old) != 1:
        raise ValueError('Protocol import anchor changed')
    text = text.replace(old, new)
    old = '        require(isinstance(parameters, dict), "tool parameters must be an object")\n'
    new = old + ('        parameters = restore_inline_conditions(wire_name, parameters,\n'
                 '            WORKFLOW_TOOLS[0]["inputSchema"]["properties"]["conditions"])\n')
    if text.count(old) != 1:
        raise ValueError('Protocol dispatch anchor changed')
    text = text.replace(old, new)
    compile(text, 'private-inline-condition-protocol', 'exec')
    return text.encode()


def build(base_build, output):
    base_build, output = Path(base_build).resolve(), Path(output).resolve()
    if digest(base_build) != BASE_BUILD_SHA:
        raise ValueError('Require the original declared native-button candidate')
    if not output.is_relative_to(RUNTIME) or output == base_build.parent or output.is_relative_to(base_build.parent):
        raise ValueError('Use a new goal-owned runtime outside the original candidate')
    manifest = json.loads(base_build.read_text())
    files = manifest['candidate_files_sha256']
    if files.get('qw3_computer/protocol.py') != PROTOCOL_SHA or files.get('qw3_computer/tool_schema.py') != SCHEMA_SHA:
        raise ValueError('Original protocol or condition declaration changed')
    contents = {}
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.suffix not in ('.py', '.html', '.json', '.tex'):
            raise ValueError('Invalid candidate inventory path')
        data = (base_build.parent / 'framework' / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Original candidate file changed: ' + name)
        contents[name] = data
    contents['qw3_computer/protocol.py'] = patch_protocol(contents['qw3_computer/protocol.py'])
    helper = Path(__file__).with_name('condition_schema_wire.py')
    contents['qw3_computer/condition_schema_wire.py'] = helper.read_bytes()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    hashes = {}
    for name, data in contents.items():
        target = output / 'framework' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    result = dict(kind='isolated_native_button_inline_condition_wire_candidate',
        base_build=str(base_build), base_build_sha256=digest(base_build), base_candidate_files_sha256=files,
        recipe_sha256=digest(__file__), helper_sha256=digest(helper), candidate_files_sha256=hashes,
        changed_package_files=[name for name in hashes if hashes[name] != files.get(name)],
        frozen_framework_changed=False, original_native_study_modified=False, model_calls=0,
        device_inputs=0, real_model_retest=False, qualified_for_production=False,
        scope='Restore only the trusted inline act condition schema after Codex transport; keep runtime predicates and all other behavior.')
    (output / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-build', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build(args.base_build, args.output)
    print(json.dumps(dict(changed_package_files=result['changed_package_files'], model_calls=0, device_inputs=0)))
