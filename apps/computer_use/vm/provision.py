"""Run a reviewed setup script in the task-owned VM through its QEMU console.

This administrative helper is never exposed to the agent's Computer MCP.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

NAME = "qw3-computer-windows"
HMP = r'''
import socket,time
s=socket.socket(socket.AF_UNIX);s.settimeout(2);s.connect('/run/shm/monitor.sock');s.recv(4096)
def key(name):
 s.sendall(('sendkey '+name+' 50\n').encode());time.sleep(.10);s.recv(8192)
key('esc');key('meta_l-r');time.sleep(.5);key('ctrl-a')
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
    shutil.copyfile(args.script, shared / "admin-script.ps1")
    subprocess.run(["docker", "exec", "-i", NAME, "python3", "-"], input=HMP, text=True, check=True)


if __name__ == "__main__":
    main()
