"""Load original attention modules without importing obsolete HF model patches.

No upstream implementation is rewritten or copied into an in-house algorithm.
The private package name avoids polluting an installed inf_llm package.
"""
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "InfLLM": ("https://github.com/thunlp/InfLLM.git", "12b70798f56e56ebb23c53c7018091a3f540a028"),
    "FlexiCache": ("https://github.com/NazmulTakbir/FlexiCache.git", "ab0f495b10fc6509c1637af35779b29121fb7d3f"),
}


def verify_source(name):
    path = ROOT / "upstream" / name
    sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if sha != PINS[name][1]:
        raise RuntimeError(f"{name}: unexpected source revision {sha}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError(f"{name}: tracked upstream source has modifications")
    return {"url": PINS[name][0], "commit": sha}


def infllm_module(suffix):
    prefix = "_kvbaseline_infllm"
    if prefix not in sys.modules:
        verify_source("InfLLM")
        source = ROOT / "upstream" / "InfLLM" / "inf_llm"
        # The package initializers import old transformers model APIs. Bypass
        # those two initializers only; attention implementation files stay exact.
        for package, directory in [(prefix, source), (prefix + ".attention", source / "attention")]:
            module = types.ModuleType(package)
            module.__path__ = [str(directory)]
            sys.modules[package] = module
    return importlib.import_module(prefix + ".attention." + suffix)


def enable_infllm_head_dim_256():
    """Adapt a shape guard and launch staging, preserving Triton kernel math.

    Qwen3.8 has head_dim=256; the original guard enumerates only up to 128.
    Validation must compare 256-d fused outputs and selection with the original
    Torch oracle. Generated source is explicit and inspectable by Triton JIT.
    """
    infllm_module("dot_production_attention")
    name = "_kvbaseline_infllm.attention.dot_production_attention.triton_impl"
    if name in sys.modules:
        return
    source = ROOT / "upstream/InfLLM/inf_llm/attention/dot_production_attention/triton_impl.py"
    text = source.read_text()
    before = "assert Lk in {16, 32, 64, 128}"
    after = "assert Lk in {16, 32, 64, 128, 256}"
    if text.count(before) != 1:
        raise RuntimeError("unexpected upstream head dimension guard")
    # Blackwell workstation SM shared memory is smaller than H100's. Use the
    # same kernel math with fewer pipeline stages for 256-dimensional heads.
    # All other head sizes retain the author's launch configuration.
    if text.count("num_stages=4") != 2:
        raise RuntimeError("unexpected upstream attention launch configuration")
    text = text.replace("num_stages=4", "num_stages=(1 if Lk == 256 else 4)")
    target = ROOT / "compat_generated/infllm/triton_impl.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.replace(before, after))
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
