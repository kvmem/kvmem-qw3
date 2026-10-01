"""Live private-site identity isolation and initialization qualification."""
import argparse
import json
from pathlib import Path
import uuid

from .sites import PORTS, Site


def qualify(output):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for app in PORTS:
        identity = "osworld-state-qualification-" + uuid.uuid4().hex[:16]
        site = Site(app, identity)
        desired = {"data": {"qualification": identity}, "note": "host-only qualification"}
        actual = site.initialize(desired)
        reused_rejected = False
        try:
            Site(app, identity).initialize({"data": {"qualification": "overwrite"}})
        except FileExistsError:
            reused_rejected = True
        after = site.state()
        # Simulate loss of the ordinary caller's registry, keeping the real
        # backend identity. Its non-fresh metadata/note must still block PUT.
        backend_reuse_rejected = False
        try:
            Site(app, identity, claims=output / "independent-claims").initialize({"data": {"qualification": "overwrite"}})
        except ValueError:
            backend_reuse_rejected = True
        final = site.state()
        case = {"app": app, "identity": identity, "fresh_initialized": actual["data"] == desired["data"],
                "local_reuse_rejected": reused_rejected, "backend_reuse_rejected": backend_reuse_rejected,
                "state_preserved": actual == after == final}
        case["passed"] = all(case[k] for k in ("fresh_initialized", "local_reuse_rejected", "backend_reuse_rejected", "state_preserved"))
        cases.append(case)
    report = {"kind": "private_site_identity_qualification", "formal_run": False,
              "passed": all(c["passed"] for c in cases), "cases": cases}
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if qualify(args.output) else 1)
