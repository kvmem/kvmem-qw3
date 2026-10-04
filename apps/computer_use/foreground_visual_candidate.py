"""Build an isolated Linux visual-context experiment from qualified SDK49.

This never edits the source package or a running app. The candidate additionally
accepts an aged visual target inside a positively identified foreground window
whose entire stored screenshot crop and semantic evidence remain unchanged.
No task, grader, age threshold, coordinate rule or delivery receipt is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path


BASELINE_DESKTOP_SHA256 = "1adeb8e59f52a685f42e91b7bd6c26453ea8eda27a1fc7e4e368cfbe0575bb6f"

VISUAL_WINDOW_METHOD = '''    def unchanged_visual_foreground(self, old, fresh, points):
        """Every visual point must be inside the same proven unchanged window."""
        if any(observation.get('accessibility', {}).get('status') != 'ok'
               for observation in (old, fresh)):
            return False
        bounds = self.matched_foreground_bounds(old, fresh)
        if bounds is None or not points:
            return False
        if any(len(point) != 2 or any(type(value) is not int for value in point)
               or not (bounds[0] <= point[0] < bounds[2]
                       and bounds[1] <= point[1] < bounds[3]) for point in points):
            return False
        return self.unchanged_foreground_window(old, fresh)

'''


def patch_desktop_bytes(original):
    if hashlib.sha256(original).hexdigest() != BASELINE_DESKTOP_SHA256:
        raise ValueError("Use the exact qualified SDK49 Desktop source")
    text = original.decode("utf-8")
    changes = [
        ("    def unchanged_target_context(", VISUAL_WINDOW_METHOD + "    def unchanged_target_context("),
        ("                visual_context_valid=False\n", "                visual_context_valid=False\n"
         "                visual_window_valid=False\n"),
        ("                    context_valid=context_valid or visual_context_valid\n",
         "                    visual_window_valid=(not visual_context_valid\n"
         "                        and self.unchanged_visual_foreground(old,fresh,points))\n"
         "                    context_valid=context_valid or visual_context_valid or visual_window_valid\n"),
        ("                    'foreground_context_unchanged':bool(keyboard_context_valid),\n",
         "                    'foreground_context_unchanged':bool(keyboard_context_valid),\n"
         "                    'visual_foreground_context_unchanged':bool(visual_window_valid),\n"),
        ("                             'Exact unchanged content and semantic evidence; visual points inside content region' if visual_context_valid else\n",
         "                             'Exact full foreground window pixels, identity and semantic evidence; every visual point inside the window' if visual_window_valid else\n"
         "                             'Exact unchanged content and semantic evidence; visual points inside content region' if visual_context_valid else\n"),
        ("                allowed=same_view or independent_app_intent or visual_context_valid or keyboard_context_valid\n",
         "                allowed=same_view or independent_app_intent or visual_context_valid or visual_window_valid or keyboard_context_valid\n"),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError("Qualified source anchor changed")
        text = text.replace(before, after, 1)
    compile(text, "qualified-sdk49-visual-foreground-candidate.py", "exec")
    return text.encode("utf-8")


def build_candidate(source_package, qualification_protocol, output):
    source_package = Path(source_package).resolve()
    protocol = json.loads(Path(qualification_protocol).read_text())
    files = protocol.get("framework_sha256")
    if not isinstance(files, dict) or files.get("desktop.py") != BASELINE_DESKTOP_SHA256:
        raise ValueError("Declare the exact qualified SDK49 package files")
    original_files = {}
    for name, expected in files.items():
        if Path(name).name != name or Path(name).suffix not in {".py", ".html", ".json", ".tex"}:
            raise ValueError("Invalid package filename")
        contents = (source_package / name).read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError("Qualified package changed: " + name)
        original_files[name] = contents
    candidate_desktop = patch_desktop_bytes(original_files["desktop.py"])
    output = Path(output).resolve()
    if output == source_package or output.is_relative_to(source_package):
        raise ValueError("Keep the experiment outside its immutable source package")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "framework" / "qw3_computer"
    destination.mkdir(parents=True)
    candidate_files = {}
    for name, contents in original_files.items():
        if name == "desktop.py":
            contents = candidate_desktop
        (destination / name).write_bytes(contents)
        candidate_files[name] = hashlib.sha256(contents).hexdigest()
    manifest = {
        "kind": "isolated_sdk49_visual_foreground_experiment",
        "qualified_for_production": False,
        "real_model_retest": False,
        "native_windows_qualified": False,
        "input_sent_during_build": 0,
        "source_package": str(source_package),
        "source_protocol_sha256": hashlib.sha256(Path(qualification_protocol).read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_files_sha256": files,
        "candidate_files_sha256": candidate_files,
        "changed_package_files": [name for name in files if files[name] != candidate_files[name]],
        "scope": "Linux aged visual points; exact whole-window pixels and positive frame evidence. SDK49 control, keyboard, receipt and grader behavior retained.",
    }
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--qualification-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_candidate(args.source_package, args.qualification_protocol, args.output)
    print(json.dumps({key: result[key] for key in ("kind", "changed_package_files", "real_model_retest")}))


if __name__ == "__main__":
    main()
