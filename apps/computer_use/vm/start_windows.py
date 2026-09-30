"""Create the task-owned Windows 11 evaluation VM; never replace existing VMs."""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess

IMAGE = "dockurr/windows@sha256:0cff9eb0e7aee9953e55bc682852ca4fdca233145a58ae1ec94f0b0c01a2ed30"
NAME = "qw3-computer-windows"
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".runtime" / "windows")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(runtime).free < 50 * 1024**3:
        raise SystemExit("Windows installation requires at least 50 GiB currently free")
    existing = subprocess.run(["docker", "container", "inspect", NAME], capture_output=True)
    if existing.returncode == 0:
        raise SystemExit(f"{NAME} already exists; inspect it instead of replacing it")
    for directory in ("storage", "shared"):
        (runtime / directory).mkdir(exist_ok=True)
    credentials = runtime / "windows.env"
    if not credentials.exists():
        with credentials.open("x") as output:
            credentials.chmod(0o600)
            output.write("USERNAME=Qw3Test\nPASSWORD=" + secrets.token_urlsafe(24) + "\n")
    direct_env = {key: value for key, value in os.environ.items() if key not in PROXY_KEYS}
    proxy_args = [value for key in PROXY_KEYS for value in ("-e", key + "=")]
    subprocess.run(["docker", "run", "-d", "--name", NAME,
                    "--label", "com.qw3.computer-use.owner=codex-goal-20260930",
                    "--device", "/dev/kvm", "--device", "/dev/net/tun",
                    "--cap-add", "NET_ADMIN", "--cpus", "6", "--memory", "20g",
                    "--stop-timeout", "120", "--env-file", str(credentials),
                    "-e", "VERSION=11e", "-e", "RAM_SIZE=8G", "-e", "CPU_CORES=4",
                    "-e", "DISK_SIZE=64G", "-e", "DISK_FMT=qcow2", "-e", "ALLOCATE=N",
                    "-e", "DISK_IO=threads", "-e", "DISK_CACHE=writeback",
                    "-e", "WIDTH=1280", "-e", "HEIGHT=720",
                    "-p", "127.0.0.1:18006:8006", "-p", "127.0.0.1:15900:5900",
                    "-p", "127.0.0.1:18082:5000",
                    "-v", str(runtime / "storage") + ":/storage",
                    "-v", str(runtime / "shared") + ":/shared",
                    "-v", str(Path(__file__).resolve().parent) + ":/oem:ro",
                    *proxy_args, IMAGE], check=True, env=direct_env)
    (runtime / "manifest.json").write_text(json.dumps({"container": NAME, "image": IMAGE,
        "version": "11e", "ram_gib": 8, "vcpus": 4, "disk_virtual_gib": 64,
        "viewer": "http://127.0.0.1:18006", "executor": "http://127.0.0.1:18082"}, indent=2))
    print("Windows installation started. Inspect: docker logs --tail 30 " + NAME)


if __name__ == "__main__":
    main()
