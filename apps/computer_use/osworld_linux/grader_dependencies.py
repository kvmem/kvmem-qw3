"""Pin local decoding tools without changing official numerical scoring rules."""
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from osworld_windows.task051 import Decoder

XCF_IMAGE = "sha256:f49901b8ad84c5e151367d30f13cf9eb90921ef9e2bf4cb8ea20bc4d262346e6"


def configure(task, directory):
    if task.id != "051":
        return {}
    decoder = Decoder(XCF_IMAGE, directory / "xcf-decoder")
    private_temp = directory / "grader-tmp"
    private_temp.mkdir()
    # The original rule writes a fixed extracted_c1.png. Scope its temporary
    # output to this attempt instead of overwriting another host task's file.
    module = sys.modules[task.__class__.__module__]
    module.tempfile = SimpleNamespace(gettempdir=lambda: str(private_temp))

    def extract(source, name, destination):
        destination = Path(destination).resolve()
        if not destination.is_relative_to(private_temp.resolve()):
            raise ValueError("XCF evaluator extraction left its private directory")
        shutil.copyfile(decoder.layer(source, name), destination)
        return True

    task._ensure_imagemagick = lambda *_args: None
    task._extract_layer_to_png = extract
    return {"xcf_decoder_image": XCF_IMAGE, "network": "none", "source_mount": "read-only",
            "score_rules": "Original numerical/layer/SIFT rules unchanged",
            "adapter": "Pinned ImageMagick extraction with per-attempt temporary output"}
