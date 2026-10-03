"""Appium UiAutomator2 adapter: native visible UI actions, no agent ADB shell."""
import base64
import hashlib
import io
import re
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import aiohttp
from PIL import Image

from .device_gateway import DeviceRejected
from .device_protocol import capabilities


APPS = {"settings": "com.android.settings", "contacts": "com.google.android.contacts", "browser": "com.android.chrome"}
KEYS = {"ENTER": 66, "TAB": 61, "BACKSPACE": 67, "DELETE": 112, "SPACE": 62,
        "UP": 19, "DOWN": 20, "LEFT": 21, "RIGHT": 22, "HOME": 122, "END": 123,
        "ANDROID_BACK": 4, "ANDROID_HOME": 3, "ANDROID_RECENTS": 187,
        "CTRL+A": 29, "CTRL+C": 31, "CTRL+V": 50, "CTRL+X": 52}


def controls_from_xml(xml, width, height):
    if len(xml) > 8*1024*1024: return {"status":"too_large","controls":[],"truncated":True}
    root = ET.fromstring(xml)
    controls = []
    # Appium uses widget-class tags; raw UiAutomator dumps use <node>.
    for node in root.iter():
        a = node.attrib
        if a.get('enabled')!='true' or a.get('password')=='true' or a.get('displayed')=='false': continue
        match = re.fullmatch(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', a.get('bounds',''))
        if not match: continue
        x,y,right,bottom = map(int, match.groups());w=right-x;h=bottom-y;cx=x+w//2;cy=y+h//2
        if not (w>0 and h>0 and 0<=cx<width and 0<=cy<height): continue
        name = a.get('content-desc') or a.get('text')
        # Keep unlabeled input/gesture targets; passive layout IDs add noise.
        if not name and any(a.get(key)=='true' for key in ('clickable','focusable','scrollable')):
            name = a.get('resource-id')
        if not name: continue
        controls.append({'name':name,'value':a.get('text'),'kind':a.get('class',node.tag),'resource_id':a.get('resource-id'),
                         'x':x,'y':y,'width':w,'height':h,'center_x':cx,'center_y':cy,
                         'states':{key:a[key]=='true' for key in ('clickable','focusable','scrollable','checkable','checked','selected') if key in a}})
        if len(controls)>80:break
    return {'status':'ok','controls':controls[:80],'truncated':len(controls)>80}


class AndroidDriver:
    def __init__(self, endpoint, device_id, serial):
        if urlparse(endpoint).hostname not in {'localhost','127.0.0.1'}:
            raise ValueError('Appium must be a local owned endpoint')
        if not re.fullmatch(r'emulator-\d+', serial):
            raise ValueError('This first adapter accepts explicitly selected emulators only')
        self.endpoint=endpoint.rstrip('/');self.device_id=device_id;self.serial=serial
        self.capabilities=capabilities('android') | {'open_apps':list(APPS),'keys':list(KEYS),
            'transport':'Appium UiAutomator2','drag_modifiers':['none'], 'app_processes':{k:[v] for k,v in APPS.items()}}
        self.session_id=None

    async def request(self, method, route, body=None):
        async with self.session.request(method,self.endpoint+route,json=body) as r:
            raw=await r.json();value=raw.get('value')
            if r.status!=200 or isinstance(value,dict) and value.get('error'):
                raise RuntimeError('Appium request failed: '+str(value.get('message',value) if isinstance(value,dict) else value))
            return value

    async def start(self):
        self.session=aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=15))
        try:
            async with self.session.post(self.endpoint+'/session',json={'capabilities':{'alwaysMatch':{
                'platformName':'Android','appium:automationName':'UiAutomator2','appium:udid':self.serial,
                'appium:noReset':True,'appium:skipUnlock':True,'appium:newCommandTimeout':3600,
                'appium:autoLaunch':False,'appium:systemPort':18201}}},timeout=aiohttp.ClientTimeout(total=120)) as r:
                raw=await r.json()
                if r.status!=200:raise RuntimeError('Unable to start UiAutomator2: '+str(raw.get('value')))
                self.session_id=raw['value']['sessionId']
        except BaseException:
            await self.session.close();raise

    async def command(self, method, route, body=None):
        return await self.request(method,'/session/'+self.session_id+route,body)

    async def foreground(self):
        package=await self.command('GET','/appium/device/current_package')
        activity=await self.command('GET','/appium/device/current_activity')
        return {'handle':package+'/'+activity,'process':package,'title':activity}

    async def observe(self):
        before=await self.foreground()
        png=base64.b64decode(await self.command('GET','/screenshot'),validate=True)
        with Image.open(io.BytesIO(png)) as image:
            width,height=image.size
            region=image.crop((0,int(height*0.04),width,int(height*0.96))).convert('RGB')
            progress=hashlib.sha256(region.tobytes()).hexdigest()
        try:
            xml=await self.command('GET','/source')
            accessibility=controls_from_xml(xml,width,height)
        except (RuntimeError,ET.ParseError): accessibility={'status':'unavailable','controls':[],'truncated':False}
        after=await self.foreground()
        if after!=before:accessibility={'status':'foreground_changed','controls':[],'truncated':False}
        return {'png':png,'foreground':after,'accessibility':accessibility,'progress_sha256':progress,
                'rotation':90 if width>height else 0}

    async def preflight(self,action,observation):
        kind=action['kind']
        if kind=='open_app' and action['app'] not in APPS:raise DeviceRejected('Unsupported Android app')
        if kind=='key' and action['key'] not in KEYS:raise DeviceRejected('Unsupported Android key')
        if kind=='drag' and action['modifier']!='none':raise DeviceRejected('Android swipe does not support keyboard modifiers')
        if kind=='click' and action['button']!='left':raise DeviceRejected('Android supports taps, not right mouse clicks')
        if kind!='open_app' and await self.foreground()!=observation['foreground']:raise DeviceRejected('Foreground app changed; observe again')
        self.display=observation['width'],observation['height']

    async def gesture(self,points,duration=500):
        actions=[{'type':'pointerMove','duration':0,'x':points[0][0],'y':points[0][1]},
                 {'type':'pointerDown','button':0}]
        if len(points)>1:actions.append({'type':'pointerMove','duration':duration,'x':points[1][0],'y':points[1][1]})
        actions.append({'type':'pointerUp','button':0})
        return await self.command('POST','/actions',{'actions':[{'type':'pointer','id':'finger','parameters':{'pointerType':'touch'},'actions':actions}]})

    async def dispatch(self,action):
        kind=action['kind']
        if kind=='open_app':
            await self.command('POST','/appium/device/activate_app',{'appId':APPS[action['app']]})
        elif kind=='click':
            if action['clicks']==1:await self.gesture([(action['x'],action['y'])])
            else:
                await self.command('POST','/execute/sync',{'script':'mobile: doubleClickGesture','args':[{'x':action['x'],'y':action['y']}]})
        elif kind=='type':
            # UiAutomator2's Unicode keyboard input; literal text only, not a shell string.
            await self.command('POST','/execute/sync',{'script':'mobile: type','args':[{'text':action['text']}]})
        elif kind=='key':
            key=action['key'];body={'keycode':KEYS[key]}
            if key.startswith('CTRL+'):body['metastate']=4096
            await self.command('POST','/appium/device/press_keycode',body)
        elif kind=='drag':
            await self.gesture([(action['from_x'],action['from_y']),(action['to_x'],action['to_y'])],action['duration_ms'])
        elif kind=='scroll':
            w,h=self.display;x=action['x'];y=action['y'];distance=min(int(h*0.55),int(h*0.08*action['steps']))
            end_y=max(int(h*0.08),y-distance) if action['direction']=='down' else min(int(h*0.9),y+distance)
            if end_y==y:raise DeviceRejected('Scroll has no available swipe distance; select a point within the content pane')
            await self.gesture([(x,y),(x,end_y)],500)
        else:raise DeviceRejected('Unsupported Android action')
        return {'backend':'UiAutomator2','request_completed':True}

    async def close(self):
        try:
            if self.session_id:await self.command('DELETE','',None)
        finally:await self.session.close()
