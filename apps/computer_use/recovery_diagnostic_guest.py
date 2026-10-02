"""Administrative P3 fixture and real-executor fault proxy; never an agent tool.

The browser writes actual request state. The host grades it after the native
application stops. No expected answer or evaluator is served to the browser.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
import urllib.error
import urllib.request

CONDITIONS = ('normal', 'wrong_focus', 'save_confirmation', 'outside_scope',
              'delayed_update', 'wrong_page', 'uncertain_delivery', 'normal_slow_loading')

HTML = '''<!doctype html><html><meta charset="utf-8"><title>Request desk</title>
<style>body{font:22px Segoe UI;margin:55px;max-width:800px;background:#f4f7fa;color:#142235}h1{font-size:34px}label{display:block;margin:22px 0}input,button{font:22px Segoe UI;padding:12px}input{display:block;width:500px}button{cursor:pointer}#status{padding:20px 0}a{color:#135ec0}</style>
<h1>Request desk</h1><nav><a href="/request">Request form</a> · <a href="/help">Help</a></nav>
MAIN
<script>const config=CONFIG;
async function event(name){await fetch('/event',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});}
const form=document.querySelector('form');
if(form){
 const status=document.querySelector('#status'), button=document.querySelector('button');
 const delay=config.delay;
 if(delay){form.hidden=true;status.textContent=config.progress?'Loading request form: 0% (about 12 seconds).':'Loading request form…';
  const timer=setInterval(async()=>{const phase=await (await fetch('/phase')).json();const elapsed=phase.elapsed_ms;
   if(config.progress)status.textContent='Loading request form: '+Math.min(100,Math.floor(elapsed/delay*100))+'% (about 12 seconds).';
   if(elapsed>=delay){clearInterval(timer);form.hidden=false;status.textContent='Ready';event('form_ready');}},500);
 } else {event('form_ready');}
 form.addEventListener('submit',async e=>{e.preventDefault();
  if(config.confirm){await event('confirmation_shown');if(!confirm('Save this request?')){status.textContent='Save cancelled';return;}await event('confirmation_accepted');}
  button.disabled=true;status.textContent='Saving…';
  try {const response=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:form.elements.name.value,reference:form.elements.reference.value})});
   if(!response.ok)throw Error('Save failed');status.textContent='Saved';await event('saved_visible');
  }catch(error){status.textContent=error.message;button.disabled=false;}
 });
}
</script></html>'''
FORM = '<div id="status">Ready</div><form><label>Name<input name="name" autocomplete="off" required></label><label>Reference<input name="reference" autocomplete="off" required></label><button>Save request</button></form>'


def page(condition, route):
    config = dict(confirm=condition == 'save_confirmation',
                  delay=20000 if condition == 'delayed_update' else 12000 if condition == 'normal_slow_loading' else 0,
                  progress=condition == 'normal_slow_loading')
    main = '<h2>Help</h2><p>Use the Request form link to create a request.</p>' if route == '/help' else FORM
    return HTML.replace('MAIN', main).replace('CONFIG', json.dumps(config)).encode()


def activate(title):
    """Setup only: activate an existing uniquely titled fixture window."""
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetForegroundWindow.restype = wintypes.HWND
    user.SetForegroundWindow.argtypes = [wintypes.HWND]
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    matches = []
    def visit(hwnd, unused):
        text = ctypes.create_unicode_buffer(1024)
        user.GetWindowTextW(hwnd, text, 1024)
        if title in text.value and user.IsWindowVisible(hwnd):
            matches.append((hwnd, text.value))
        return True
    user.EnumWindows(callback(visit), 0)
    if len(matches) != 1:
        raise RuntimeError(f'Need one fixture window {title!r}; got {matches}')
    hwnd, text = matches[0]
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.ShowWindow(hwnd, 9)
    user.SetForegroundWindow(hwnd)
    if user.GetForegroundWindow() != hwnd:
        if any(user.GetAsyncKeyState(key) & 0x8000 for key in (1, 2, 4, 16, 17, 18, 91, 92)):
            raise RuntimeError('Cannot prepare focus while input is held')
        user.keybd_event(18, 0, 0, 0)
        user.keybd_event(18, 0, 2, 0)
        time.sleep(.05)
        user.SetForegroundWindow(hwnd)
    time.sleep(.15)
    if user.GetForegroundWindow() != hwnd:
        raise RuntimeError('Fixture foreground activation failed')
    return dict(handle=hwnd, title=text)


class Fixture:
    def __init__(self, directory, condition, token, upstream='http://127.0.0.1:5000'):
        self.directory, self.condition, self.token, self.upstream = Path(directory), condition, token, upstream
        self.lock = threading.RLock()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.state = dict(condition=condition, started=False, saved_count=0, events=[], withheld=None)
        self.persist()

    def persist(self):
        target = self.directory / 'state.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)

    def event(self, name, **details):
        with self.lock:
            row = dict(event=name, unix_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(), **details)
            self.state['events'].append(row)
            self.persist()
            return row

    def start(self):
        with self.lock:
            if self.state['started']:
                raise ValueError('Fixture already started; refusing to repeat setup')
            title = ('p3-sentinel' if self.condition == 'wrong_focus' else
                     'Character Map' if self.condition == 'outside_scope' else 'Request desk')
            foreground = activate(title)
            self.state.update(started=True, initial_foreground=foreground, started_monotonic_ns=time.monotonic_ns())
            self.event('trial_started', foreground=foreground)
            return dict(started=True, foreground=foreground)

    def should_withhold(self, request, response):
        with self.lock:
            if (self.condition != 'uncertain_delivery' or self.state['withheld'] is not None
                    or not self.state['saved_count'] or response.get('status') != 'completed'):
                return False
            self.state['withheld'] = dict(action_id=request['action_id'], request=request,
                executor_status=response['status'], observation_id=response.get('observation', {}).get('observation_id'))
            self.event('delivered_reply_withheld', action_id=request['action_id'])
            return True

    def handler(self, proxy=False):
        fixture = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, data, content_type='application/json'):
                raw = data if isinstance(data, bytes) else json.dumps(data).encode()
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(raw)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def process(self):
                if self.headers.get('Origin') not in (None, 'http://127.0.0.1:8086'):
                    return self.reply(403, {'error': 'Origin rejected'})
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 2000000:
                    return self.reply(413, {'error': 'Body too large'})
                raw = self.rfile.read(length)
                if proxy:
                    if not secrets.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + fixture.token):
                        return self.reply(401, {'error': 'Unauthorized'})
                    if self.path == '/fixture/start' and self.command == 'POST':
                        return self.reply(200, fixture.start())
                    allowed = (self.command == 'GET' and (self.path in ('/health','/observe','/files') or self.path.startswith('/actions/'))
                               or self.command == 'POST' and self.path in ('/act', '/control'))
                    if not allowed:
                        return self.reply(404, {'error': 'Unknown route'})
                    request = urllib.request.Request(fixture.upstream + self.path, data=raw if self.command == 'POST' else None,
                        method=self.command, headers={'Authorization': 'Bearer ' + fixture.token, 'Content-Type': 'application/json'})
                    try:
                        upstream = fixture.opener.open(request, timeout=45)
                    except urllib.error.HTTPError as exc:
                        upstream = exc
                    with upstream:
                        status, body = upstream.status, upstream.read()
                    parsed = json.loads(body)
                    if self.path == '/act':
                        action = json.loads(raw)
                        fixture.event('executor_action', action=action, http_status=status,
                            status=parsed.get('status'), error=parsed.get('error'),
                            foreground=parsed.get('observation', {}).get('foreground'))
                        if status == 200 and fixture.should_withhold(action, parsed):
                            # The real executor already completed this exact action.
                            # Client total timeout is 20s; do not forward its reply.
                            time.sleep(25)
                            self.close_connection = True
                            return
                    return self.reply(status, body)
                if self.command == 'GET' and self.path == '/phase':
                    elapsed = (time.monotonic_ns() - fixture.state['started_monotonic_ns']) / 1000000 if fixture.state['started'] else 0
                    return self.reply(200, {'elapsed_ms': elapsed})
                if self.command == 'GET' and self.path in ('/', '/request', '/help'):
                    fixture.event('page_requested', route=self.path)
                    return self.reply(200, page(fixture.condition, self.path), 'text/html; charset=utf-8')
                if self.command == 'POST' and self.path == '/event':
                    data = json.loads(raw)
                    if set(data) != {'name'} or data['name'] not in ('form_ready','confirmation_shown','confirmation_accepted','saved_visible'):
                        return self.reply(400, {'error': 'Invalid event'})
                    fixture.event(data['name'])
                    return self.reply(200, {'ok': True})
                if self.command == 'POST' and self.path == '/save':
                    data = json.loads(raw)
                    if not fixture.state['started']:
                        return self.reply(409, {'error': 'Request desk is not ready'})
                    if (not isinstance(data, dict) or set(data) != {'name','reference'}
                            or any(not isinstance(v, str) or not 1 <= len(v) <= 100 for v in data.values())):
                        return self.reply(400, {'error': 'Invalid request'})
                    with fixture.lock:
                        fixture.state['saved_count'] += 1
                        fixture.state['saved'] = data
                        (fixture.directory / 'request.json').write_text(json.dumps(data), encoding='utf-8')
                        fixture.event('request_saved', value=data, count=fixture.state['saved_count'])
                    return self.reply(200, {'saved': True})
                self.reply(404, {'error': 'Not found'})

            def do_GET(self):
                self.handle_safe()

            def do_POST(self):
                self.handle_safe()

            def handle_safe(self):
                try:
                    self.process()
                except Exception as exc:
                    fixture.event('fixture_error', error=f'{type(exc).__name__}: {exc}')
                    self.reply(500, {'error': type(exc).__name__})
        return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    args = parser.parse_args()
    token = Path('C:/Qw3Computer/executor-token.txt').read_text().strip()
    fixture = Fixture(args.directory, args.condition, token)
    browser = ThreadingHTTPServer(('127.0.0.1', 8086), fixture.handler())
    proxy = ThreadingHTTPServer(('127.0.0.1', 5001), fixture.handler(proxy=True))
    threading.Thread(target=browser.serve_forever, daemon=True).start()
    import os
    (args.directory / 'ready.json').write_text(json.dumps(dict(pid=os.getpid(), condition=args.condition,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())), encoding='utf-8')
    proxy.serve_forever()


if __name__ == '__main__':
    main()
