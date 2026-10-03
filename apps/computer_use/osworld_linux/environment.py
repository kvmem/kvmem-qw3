"""Pinned, loopback-only Docker provider for the original OSWorld environment."""
import json
import os
from pathlib import Path
import requests

from osworld_windows.source import ROOT

IMAGE='sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9'
DISK=ROOT/'third_party/OSWorld-V2/docker_vm_data/osworld-v2-ubuntu-x86.qcow2'
PORTS={'server':18300,'chromium':18301,'vnc':18302,'vlc':18303}


def create(directory,name):
    import desktop_env.desktop_env as module
    from desktop_env.providers.docker.provider import DockerProvider
    from desktop_env.providers.docker.manager import DockerVMManager

    class PinnedProvider(DockerProvider):
        def start_emulator(self,path_to_vm,headless,os_type):
            if Path(path_to_vm).resolve()!=DISK.resolve() or os_type!='Ubuntu':raise ValueError('Unexpected VM image')
            if self.container is not None:raise RuntimeError('VM already owned')
            # Docker refuses a colliding name/port; never removes an existing container.
            self.server_port=PORTS['server'];self.chromium_port=PORTS['chromium']
            self.vnc_port=PORTS['vnc'];self.vlc_port=PORTS['vlc']
            self.container=self.client.containers.run(IMAGE,name=name,labels={'qw3.goal':'cross-device-20261003'},
                environment={'DISK_SIZE':'32G','RAM_SIZE':'8G','CPU_CORES':'6'},cap_add=['NET_ADMIN'],devices=['/dev/kvm'],
                volumes={str(DISK):{'bind':'/System.qcow2','mode':'ro'}},
                ports={5000:('127.0.0.1',self.server_port),9222:('127.0.0.1',self.chromium_port),
                       8006:('127.0.0.1',self.vnc_port),8080:('127.0.0.1',self.vlc_port)},detach=True)
            (directory/'vm.json').write_text(json.dumps({'container_id':self.container.id,'name':name,'image':IMAGE,
                'disk':str(DISK),'disk_read_only':True,'ports':PORTS,'bind':'127.0.0.1'},indent=2))
            self._wait_for_vm_ready(timeout=300)

    original_factory=module.create_vm_manager_and_provider
    module.create_vm_manager_and_provider=lambda *args,**kwargs:(DockerVMManager(),PinnedProvider(None))
    # Task initialization is attempted once per recorded attempt. An uncertain
    # website setup must not be silently retried on another identity.
    module.MAX_RETRIES=1
    try:
        env=module.DesktopEnv(provider_name='docker',path_to_vm=str(DISK),headless=True,
            require_a11y_tree=False,enable_proxy=False,action_space='pyautogui',
            force_disable_vnc=True,force_disable_recording=True,cache_dir=str(directory/'env_cache'))
    finally:module.create_vm_manager_and_provider=original_factory
    domains=' '.join(app+'.qw3.test' for app in ['formcraft','teamchat','cloudcrm','calendar','mailhub'])
    # Setup-only fixed guest network mapping. No host /etc/hosts changes and no
    # arbitrary guest code is exposed to the model's tools.
    code="import subprocess; subprocess.run(['sudo','-S','sh','-c',"+repr('printf "172.17.0.1 '+domains+'\\n" >> /etc/hosts')+"],input='osworld-public-evaluation\\n',text=True,check=True)"
    response=requests.post(f'http://127.0.0.1:{PORTS["server"]}/execute',json={'command':['python','-c',code],'shell':False},timeout=30)
    response.raise_for_status();result=response.json()
    if result.get('returncode',0)!=0 or result.get('status')=='error':
        env.close();raise RuntimeError('Guest local website routing initialization failed')
    return env
