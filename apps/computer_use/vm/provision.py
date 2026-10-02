"""Run a reviewed setup script in the task-owned VM through its QEMU console.

This administrative helper is never exposed to the agent's Computer MCP.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import uuid

NAME = "qw3-computer-windows"
HMP = r'''
import socket,time
s=socket.socket(socket.AF_UNIX);s.settimeout(2);s.connect('/run/shm/monitor.sock');s.recv(4096)
def key(name):
 s.sendall(('sendkey '+name+' 50\n').encode());time.sleep(.15);s.recv(8192)
key('esc');key('meta_l-r');time.sleep(1.5);key('ctrl-a');time.sleep(.3)
mapping={' ':'spc','-':'minus',':':'shift-semicolon','\\':'backslash','.':'dot'}
for c in r'powershell -noprofile -executionpolicy bypass -file \\host.lan\data\admin-script.ps1':key(mapping.get(c,c))
key('ret');time.sleep(1);s.close()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    args = parser.parse_args()
    info = json.loads(subprocess.check_output(["docker", "inspect", NAME], text=True))[0]
    if info["Config"]["Labels"].get("com.qw3.computer-use.owner") != "codex-goal-20260930":
        raise SystemExit("VM owner does not match this task")
    shared = next(Path(m["Source"]) for m in info["Mounts"] if m["Destination"] == "/shared")
    # A Windows SMB client can retain the preceding contents of a path that the
    # Linux host overwrites. A unique path also preserves the exact setup script
    # and prevents a later invocation from replacing a script still executing.
    staged_name = "admin-" + uuid.uuid4().hex + ".ps1"
    shutil.copyfile(args.script, shared / staged_name)
    command = HMP.replace("admin-script.ps1", staged_name)
    subprocess.run(["docker", "exec", "-i", NAME, "python3", "-"], input=command, text=True, check=True)


if __name__ == "__main__":
    main()
