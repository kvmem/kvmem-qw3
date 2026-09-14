"""Ensure reversible private queries fit before enabling their mutation guard.

The existing capacity policy runs before snapshot/capture. Inside a private probe
it is correctly disabled, since reselecting there cannot be rolled back safely.
No query tokens, attention computations or replay are omitted.
"""
from pathlib import Path
import difflib
import json

ROOT = Path(__file__).resolve().parent
path = ROOT / "native_source/src/qwen_native_backend.cpp"
before = path.read_text()
after = before
if before.count('// Admit the whole reversible private query before its guard disables pressure handling.') == 2:
    print('already applied')
    raise SystemExit(0)

for begin_marker, end_marker, insertion_marker, reserve, timer in [
    ('                            if (!executor_->kvmem_begin_guided_query_probe(\n',
     '                            // This private turn is never registered',
     '                            query_guided_private_prompt_tokens =\n',
     'private_prompt.size() + options.kvmem_query_guided_thinking_max_tokens + '
     'options.kvmem_query_guided_query_max_tokens + 64u',
     'post_query_guided_state_s'),
    ('                if (!executor_->kvmem_begin_guided_query_probe(\n',
     '                std::string private_prompt;\n',
     '                private_prompt_tokens =\n',
     'private_prompt_ids.size() + options.kvmem_middecode_query_max_tokens + 64u',
     'kvmem_middecode_state_s'),
]:
    # Anchor to complete line starts so the shorter indent cannot match the
    # boundary path. Patching is atomic and repeated invocations are no-ops.
    begin_marker = '\n' + begin_marker
    end_marker = '\n' + end_marker
    insertion_marker = '\n' + insertion_marker
    begin = after.index(begin_marker) + 1
    end = after.index(end_marker, begin) + 1
    insert = after.index(insertion_marker) + 1
    if begin > insert:
        continue
    block = after[begin:end]
    after = after[:begin] + after[end:]
    insert = after.index(insertion_marker) + 1
    indent = begin_marker.split('if (')[0].lstrip('\n')
    addition = (indent + '// Admit the whole reversible private query before its guard disables pressure handling.\n' +
                indent + '{\n' + indent + '    const double capacity_start = wall_seconds();\n' +
                indent + '    executor_->kvmem_prepare_decode_window(static_cast<uint32_t>(\n' +
                indent + '        ' + reserve + '));\n' +
                indent + '    ' + timer + ' += wall_seconds() - capacity_start;\n' +
                indent + '}\n')
    after = after[:insert] + addition + block + after[insert:]

if after != before:
    path.write_text(after)
    (ROOT / 'reports/native_guided_capacity_patch.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='src/qwen_native_backend.cpp',
        tofile='src/qwen_native_backend.cpp')))
    manifest = ROOT / 'reports/native_source_manifest.json'
    data = json.loads(manifest.read_text())
    data.update(additional_guided_capacity_patch='reports/native_guided_capacity_patch.diff',
                guided_capacity_patch_validated=False)
    manifest.write_text(json.dumps(data, indent=2) + '\n')
print('patched' if after != before else 'already applied')
