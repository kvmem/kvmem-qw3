"""OSWorld's PyAutoGUI/AT-SPI backend; exposes no Python or shell tool to agents."""
import ast
import asyncio
import hashlib
import io
import json
import re
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import aiohttp
from PIL import Image

from .device_gateway import DeviceRejected
from .device_protocol import capabilities


FOREGROUND_SCRIPT = '''import json,subprocess,re
root=subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True)
match=re.search(r'0x[0-9a-fA-F]+',root)
handle=match.group() if match else '0x0'
if handle=='0x0':
 print(json.dumps({'handle':handle,'process':'gnome-shell','title':'Desktop'}))
else:
 props=subprocess.check_output(['xprop','-id',handle,'_NET_WM_NAME','WM_NAME','WM_CLASS'],text=True)
 title='';process=''
 for line in props.splitlines():
  try:values=json.loads('['+line.split(' = ',1)[1]+']')
  except (ValueError,IndexError):values=[]
  if line.startswith('_NET_WM_NAME') and values:title=values[0]
  if line.startswith('WM_NAME') and not title and values:title=values[0]
  if line.startswith('WM_CLASS') and values:process=values[-1]
 print(json.dumps({'handle':handle,'process':process,'title':title}))
'''

APPS = {"notepad": ["gedit"], "edge": ["google-chrome"], "explorer": ["nautilus"],
        "writer": ["libreoffice","--writer"], "calc": ["libreoffice","--calc"],
        "impress": ["libreoffice","--impress"], "thunderbird": ["thunderbird"], "gimp": ["gimp"]}


def attrs(node):
    return {key.rsplit('}',1)[-1]:value for key,value in node.attrib.items()}


def pair(text):
    value = ast.literal_eval(text)
    if not isinstance(value,tuple) or len(value)!=2 or any(type(v) is not int for v in value):
        raise ValueError("Bad AT-SPI bounds")
    return value


def controls_from_xml(xml, foreground, width, height):
    if len(xml)>8*1024*1024:
        return {"status":"too_large","controls":[],"truncated":True}
    root=ET.fromstring(xml)
    # Only the current foreground frame. Do not click hidden/background trees.
    frames=[node for node in root.iter() if node.tag.rsplit('}',1)[-1] in {'frame','dialog'} and attrs(node).get('name')==foreground['title']]
    if len(frames)!=1:
        return {"status":"foreground_unmatched","controls":[],"truncated":False}
    controls=[]
    for node in frames[0].iter():
        a=attrs(node)
        if a.get('visible')!='true' or a.get('showing')!='true' or a.get('enabled')!='true':continue
        if 'password' in node.tag:continue
        try:
            x,y=pair(a['screencoord']);w,h=pair(a['size']);cx=x+w//2;cy=y+h//2
        except (KeyError,ValueError,SyntaxError):continue
        if not (w>0 and h>0 and 0<=cx<width and 0<=cy<height):continue
        name=a.get('name') or (node.text or '').strip()
        if not name:continue
        controls.append({'name':name[:1024],'kind':node.tag.rsplit('}',1)[-1],
                         'x':x,'y':y,'width':w,'height':h,'center_x':cx,'center_y':cy,
                         'value':node.text if node.text is not None else a.get('value')})
        if len(controls)>80:break
    return {'status':'ok','controls':controls[:80],'truncated':len(controls)>80}


class LinuxDriver:
    def __init__(self,endpoint,device_id):
        if urlparse(endpoint).hostname not in {'localhost','127.0.0.1'}:
            raise ValueError("OSWorld device service must be a local owned endpoint")
        self.endpoint=endpoint.rstrip('/')
        self.device_id=device_id
        self.capabilities=capabilities('linux') | {'open_apps':list(APPS),'transport':'OSWorld PyAutoGUI/AT-SPI',
            'app_processes': {'notepad':['Gedit'],'edge':['Google-chrome'],'explorer':['Org.gnome.Nautilus'],
                              'writer':['libreoffice-writer'],'calc':['libreoffice-calc'],
                              'impress':['libreoffice-impress'],'thunderbird':['Thunderbird'],'gimp':['Gimp']}}

    async def start(self):
        self.session=aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=8))

    async def execute(self,code):
        # One POST only: even OSWorld's default controller retry would be unsafe here.
        async with self.session.post(self.endpoint+'/execute',json={'command':['python','-c',code],'shell':False}) as r:
            result=await r.json()
            if r.status!=200 or result.get('returncode',0)!=0 or result.get('status')=='error':
                if 'QW3_PRE_DISPATCH_FOCUS_REJECT' in str(result.get('error','')):
                    raise DeviceRejected('Foreground changed immediately before dispatch; no input sent')
                raise RuntimeError('Guest execution failed: '+str(result.get('error',result.get('stderr','unknown'))))
            return result

    async def foreground(self):
        result=await self.execute(FOREGROUND_SCRIPT)
        return json.loads(result['output'].strip())

    async def observe(self):
        before=await self.foreground()
        async with self.session.get(self.endpoint+'/screenshot') as r:
            r.raise_for_status();png=await r.read()
        with Image.open(io.BytesIO(png)) as image:
            width,height=image.size
            # Exclude the desktop's clock, dock and cursor from foreground progress.
            region=image.crop((75,30,width,max(31,height-45))).convert('RGB')
            progress=hashlib.sha256(region.tobytes()).hexdigest()
        accessibility={'status':'unavailable','controls':[],'truncated':False}
        try:
            async with self.session.get(self.endpoint+'/accessibility',timeout=aiohttp.ClientTimeout(total=4)) as r:
                r.raise_for_status();tree=(await r.json())['AT']
            accessibility=controls_from_xml(tree,before,width,height)
        except (aiohttp.ClientError,TimeoutError,ET.ParseError,KeyError):pass
        after=await self.foreground()
        if before!=after:accessibility={'status':'foreground_changed','controls':[],'truncated':False}
        return {'png':png,'foreground':after,'accessibility':accessibility,'progress_sha256':progress}

    async def preflight(self,action,observation):
        if action['kind']=='open_app':
            if action['app'] not in APPS:raise DeviceRejected('Application is unsupported on Linux')
        elif await self.foreground()!=observation['foreground']:
            raise DeviceRejected('Foreground changed; observe again')
        self.expected_foreground=observation['foreground']

    async def dispatch(self,action):
        kind=action['kind']
        prefix='import pyautogui;pyautogui.FAILSAFE=False;'
        if kind=='click':
            code=f"pyautogui.click({action['x']},{action['y']},clicks={action['clicks']},interval=0.1,button={action['button']!r})"
        elif kind=='type':
            # A clipboard owner's inherited stdout/stderr can strand /execute.
            code=("import subprocess;subprocess.run(['xclip','-selection','clipboard'],"
                  f"input={action['text'].encode('utf-8')!r},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=3);"
                  "pyautogui.hotkey('ctrl','v')")
        elif kind=='key':
            keys=[{'CTRL':'ctrl','ALT':'alt','SHIFT':'shift','ESC':'esc','PAGEUP':'pageup','PAGEDOWN':'pagedown'}.get(k,k.lower()) for k in action['key'].split('+')]
            code='pyautogui.hotkey('+','.join(repr(k) for k in keys)+')'
        elif kind=='scroll':
            amount=action['steps']*(1 if action['direction']=='up' else -1)
            code=f"pyautogui.moveTo({action['x']},{action['y']});pyautogui.scroll({amount})"
        elif kind=='drag':
            key=action['modifier'].lower()
            # Use try/finally even on a guest exception, so a modifier is never left held.
            code=f"\npyautogui.moveTo({action['from_x']},{action['from_y']})\n"
            if key!='none':code+=f"pyautogui.keyDown({key!r})\n"
            code+=f"try:\n pyautogui.dragTo({action['to_x']},{action['to_y']},duration={action['duration_ms']/1000},button='left')\nfinally:\n pyautogui.mouseUp()\n"
            if key!='none':code+=f" pyautogui.keyUp({key!r})\n"
        elif kind=='open_app':
            code=f"import subprocess;subprocess.Popen({APPS[action['app']]!r},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
        else:raise DeviceRejected('Unsupported Linux action')
        if kind!='open_app':
            # Recheck in the same guest process immediately before sending input.
            guard=("import subprocess,re;_root=subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True);"
                   "_match=re.search(r'0x[0-9a-fA-F]+',_root);_handle=_match.group() if _match else '0x0';"
                   f"assert _handle=={self.expected_foreground['handle']!r},'QW3_PRE_DISPATCH_FOCUS_REJECT';")
            prefix=guard+prefix
        result=await self.execute(prefix+code)
        return {'backend':'pyautogui','returncode':result.get('returncode',0)}

    async def close(self):
        await self.session.close()
