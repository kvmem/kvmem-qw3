"""Freeze an isolated native build; do not edit or commit the user's source tree."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TARGET = ROOT / "native_source"


def replace_once(path, before, after):
    original = path.read_text()
    if original.count(before) != 1:
        raise RuntimeError(f"unexpected source for isolated patch: {path.name}")
    changed = original.replace(before, after)
    path.write_text(changed)
    return "".join(difflib.unified_diff(original.splitlines(keepends=True),
        changed.splitlines(keepends=True), fromfile=str(path.relative_to(TARGET)),
        tofile=str(path.relative_to(TARGET))))


def main():
    if TARGET.exists():
        raise SystemExit("native_source already exists; preserve the frozen copy")
    TARGET.mkdir()
    for name in ["src", "include"]:
        shutil.copytree(REPO / name, TARGET / name)
    shutil.copy2(REPO / "CMakeLists.txt", TARGET / "CMakeLists.txt")
    (TARGET / "third_party").mkdir()
    for name in ["json.hpp", "httplib.h"]:
        shutil.copy2(REPO / "third_party" / name, TARGET / "third_party" / name)
    shutil.copytree(REPO / "third_party/yyjson", TARGET / "third_party/yyjson")
    hashes = {str(p.relative_to(TARGET)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(TARGET.rglob("*")) if p.is_file()}
    diff = replace_once(TARGET / "src/model_source.cpp",
        '        if (method != "compressed-tensors") {\n'
        '            throw std::runtime_error("HF model is not compressed-tensors: " + method);\n'
        '        }',
        '        if (!method.empty() && method != "compressed-tensors") {\n'
        '            throw std::runtime_error("unsupported HF quantization method: " + method);\n'
        '        }\n'
        '        if (method.empty()) info_.format = "safetensors/unquantized";')
    # Keep the guided/replay path and its required recurrent + MTP prefix state.
    # Disable only speculative token proposals in the main generation loop for
    # a no-speculation comparison. Prefix maintenance cost stays in KVMem timing.
    diff += replace_once(TARGET / "src/qwen_native_backend.cpp",
        '        bool run_spec_mtp = spec_mtp && use_mtp_prefix && !dump;',
        '        bool run_spec_mtp = spec_mtp && use_mtp_prefix && !dump &&\n'
        '            !env_flag_enabled("QW3_BASELINE_NO_SPECULATIVE_DECODE");\n'
        '        if (env_flag_enabled("QW3_BASELINE_NO_SPECULATIVE_DECODE")) {\n'
        '            log("baseline: speculative token proposals disabled; guided/replay and prefix maintenance retained");\n'
        '        }')
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "native_adapter.patch").write_text(diff)
    # The bridge target uses the exact frozen server renderer and tool parser.
    with (TARGET / "CMakeLists.txt").open("a") as cmake:
        cmake.write('\ninclude("${CMAKE_CURRENT_SOURCE_DIR}/../bridge_target.cmake")\n')
    (reports / "native_source_manifest.json").write_text(json.dumps({
        "source_repo": str(REPO), "git_head": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "includes_current_uncommitted_source": True, "original_sha256": hashes,
        "patch": "native_adapter.patch", "unvalidated": True,
        "notes": "Main tree unchanged. BF16 loader and no-speculation path require parity checks before benchmarking."
    }, indent=2) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
