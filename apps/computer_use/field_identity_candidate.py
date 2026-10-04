"""Build a private field-reference experiment on the post-SDK51 candidate.

Existing label conditions retain their exact semantics. field_value is explicit
and available only for references minted from observed native object identity.
"""
import argparse
import hashlib
import json
from pathlib import Path

from post_sdk51_candidate import replace_once


EXPECTED = {
    "desktop.py": "353cab04e90a752d001e854997b95ed9546e9399bb77b168c40841186d753ce6",
    "linux_driver.py": "949aab700b326dc9d340cdbfdfa73747a85b6828c84fdf9c7b28f497afc94ff4",
    "execution.py": "de584fd72cd68f651baf607932737cc9048603e9c641bb7ee6cdb9c14377adcf",
    "tool_schema.py": "6d7e24c2460f98f40db457952fd4c7445cb56cf01384d34b56a687617bc0c277",
}


SCHEMA_EXTENSION = '''# Native-object references are completion selectors, never input targets.
check_fields['field_ref'] = {'type': 'string', 'pattern': '^fld_[a-f0-9]{32}$',
    'description': 'Copy a field_ref from a visible control. It tracks that native object within this task/device/window session even when its name/value changes. Never invent one.'}
check_fields['kind']['enum'].append('field_value')
WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']['items']['oneOf'].append({
    'type': 'object', 'properties': {'kind': {'const': 'field_value'},
        'field_ref': check_fields['field_ref'], 'value': check_fields['value']},
    'required': ['kind', 'field_ref', 'value'], 'additionalProperties': False})
WORKFLOW_TOOLS[0]['description'] += (
    ' For a control exposing field_ref, use {kind:field_value,field_ref:<copied reference>,value:<expected literal value>} '
    'to check its value after an edit. Names may change with the content: do not use the old value as an exact-name label. '
    'field_ref is only for completion conditions; click_control still requires the latest observation control_id. '
    'If a native reference is unavailable, do not guess it or substitute coordinates/resource IDs as identity.')

'''


def patch_package_file(name, original):
    if name not in EXPECTED or hashlib.sha256(original).hexdigest() != EXPECTED[name]:
        raise ValueError("Use the exact post-SDK51 source: " + name)
    text = original.decode()
    if name == "desktop.py":
        text = replace_once(text, 'from .targeting import TargetError, identify_controls\n',
                            'from .targeting import TargetError, identify_controls\nfrom .field_identity import identify_fields\n')
        text = replace_once(text, 'metadata = identify_controls({k: v for k, v in observation.items() if k != "image"})',
                            'metadata = identify_fields(identify_controls({k: v for k, v in observation.items() if k != "image"}), self.task_id)')
        text = replace_once(text, '"x", "y", "width", "height", "resource_id"}',
                            '"x", "y", "width", "height", "resource_id", "provider_identity"}')
        text = replace_once(text, '            if visible.get("value") in (None, "", visible.get("name")):',
                            '            if "field_ref" not in visible and visible.get("value") in (None, "", visible.get("name")):')
    elif name == "linux_driver.py":
        text = replace_once(text, 'from .device_protocol import capabilities\n',
                            'from .device_protocol import capabilities\nfrom .field_identity import atspi_identity_script, provider_identity_from_attrs\n')
        text = replace_once(text, "    return '''import sys,json\n", "    source = '''import sys,json\n")
        text = replace_once(text, '\n\ndef attrs(node):', '\n    return atspi_identity_script(source)\n\ndef attrs(node):')
        text = replace_once(text, '        if len(controls)>80:break\n',
                            '        identity=provider_identity_from_attrs(a)\n'
                            '        if identity is not None:controls[-1]["provider_identity"]=identity\n'
                            '        if a.get("editable")=="true":controls[-1]["editable"]=True\n'
                            '        if len(controls)>80:break\n')
        text = replace_once(text,
                            'self.session=aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=8))',
                            'self.session=aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=8),\n'
                            '            connector=aiohttp.TCPConnector(force_close=True))')
    elif name == "execution.py":
        text = replace_once(text, 'from .desktop import DesktopError, DesktopValidationError\n',
                            'from .desktop import DesktopError, DesktopValidationError\nfrom .field_identity import field_value_evidence, FIELD_PATTERN\n')
        text = replace_once(text, '"foreground_app", "screen_changed"}\n', '"foreground_app", "screen_changed", "field_value"}\n')
        text = replace_once(text, '{"kind", "text", "value", "control_kind"}', '{"kind", "text", "value", "control_kind", "field_ref"}')
        text = replace_once(text, '        required = {"kind"} if check["kind"] == "screen_changed" else {"kind", "text"}\n',
                            '        if check["kind"] == "field_value":\n'
                            '            if (set(check) != {"kind", "field_ref", "value"}\n'
                            '                    or not isinstance(check.get("field_ref"), str)\n'
                            '                    or not re.fullmatch(FIELD_PATTERN, check["field_ref"])\n'
                            '                    or not isinstance(check.get("value"), str) or len(check["value"]) > 512):\n'
                            '                raise DesktopError("field_value requires an observed field_ref and a literal value of at most512 characters")\n'
                            '            continue\n'
                            '        required = {"kind"} if check["kind"] == "screen_changed" else {"kind", "text"}\n')
        text = replace_once(text, '        kind, text = check["kind"], check.get("text", "")\n',
                            '        if check["kind"] == "field_value":\n'
                            '            evidence.append(field_value_evidence(check, observation))\n'
                            '            continue\n'
                            '        kind, text = check["kind"], check.get("text", "")\n')
    elif name == "tool_schema.py":
        text = replace_once(text, '# Public validation admits this only when Execution is enabled;',
                            SCHEMA_EXTENSION + '# Public validation admits this only when Execution is enabled;')
    compile(text, "isolated-field-identity-" + name, "exec")
    return text.encode()


def build_candidate(source_package, source_build, output):
    source_package = Path(source_package).resolve()
    source_build = Path(source_build)
    files = json.loads(source_build.read_text()).get("candidate_files_sha256", {})
    if any(files.get(name) != digest for name, digest in EXPECTED.items()):
        raise ValueError("Declare the post-SDK51 package inventory")
    contents = {}
    for name, digest in files.items():
        if Path(name).name != name or Path(name).suffix not in {".py", ".html", ".json", ".tex"}:
            raise ValueError("Invalid package filename")
        contents[name] = (source_package / name).read_bytes()
        if hashlib.sha256(contents[name]).hexdigest() != digest:
            raise ValueError("Source package changed: " + name)
    for name in EXPECTED:
        contents[name] = patch_package_file(name, contents[name])
    helper = Path(__file__).with_name("field_identity.py")
    contents["field_identity.py"] = helper.read_bytes()
    output = Path(output).resolve()
    if output == source_package or output.is_relative_to(source_package):
        raise ValueError("Keep the experiment outside its immutable source")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "framework/qw3_computer"
    destination.mkdir(parents=True)
    hashes = {}
    for name, data in contents.items():
        (destination / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {"kind": "isolated_post51_native_field_identity_candidate", "qualified_for_production": False,
        "native_windows_qualified": False, "real_model_retest": False, "model_calls": 0, "device_inputs": 0,
        "source_package": str(source_package), "source_files_sha256": files,
        "source_build_sha256": hashlib.sha256(source_build.read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(), "candidate_files_sha256": hashes,
        "changed_package_files": [name for name in hashes if hashes[name] != files.get(name)],
        "field_provider": "Linux AT-SPI bus owner/PID/native window and object paths",
        "scope": "Explicit field_value evidence from session-scoped native identity. Fresh Linux HTTP connections avoid idle socket reuse, with one POST and no input retry. Old exact-name conditions, reservations, budgets and grading retained. No Android/Windows reference inferred from coordinates/resource IDs."}
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_candidate(args.source_package, args.source_build, args.output)
    print(json.dumps({key: result[key] for key in ("kind", "changed_package_files", "real_model_retest")}))


if __name__ == "__main__":
    main()
