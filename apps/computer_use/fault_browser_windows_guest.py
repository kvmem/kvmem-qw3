"""Stdlib fixture transport for the same recovery page in the Windows VM.

The host stages the page; this server exposes no private files or expected
answer. State is independently collected after the agent reliably stops.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time


CONDITIONS={'slow_loading','focus_dialog','no_progress','normal'}
EVENTS={'loading_visible','form_ready','notice_shown','notice_closed','saving_pending_visible','saved_visible'}


def page(condition,source):
    if condition not in CONDITIONS:raise ValueError('Unknown condition')
    start='const page=`';end='`;\npersist();'
    if source.count(start)!=1 or source.count(end)!=1:raise ValueError('Recovery page template changed')
    body=source.split(start,1)[1].split(end,1)[0]
    if body.count('${JSON.stringify(condition)}')!=1:raise ValueError('Page condition marker changed')
    return body.replace('${JSON.stringify(condition)}',json.dumps(condition))


class Fixture:
    def __init__(self,directory,condition,html):
        if condition not in CONDITIONS:raise ValueError('Unknown condition')
        self.directory=Path(directory);self.condition=condition;self.html=html
        self.lock=threading.RLock()
        self.state={'condition':condition,'save_attempts':[],'saved_count':0,'events':[],'started_at':time.time()}
        self.persist()

    def persist(self):
        target=self.directory/'fault-state.json';temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state,ensure_ascii=False),encoding='utf-8');temporary.replace(target)

    def event(self,name):
        self.state['events'].append({'name':name,'at':time.time()});self.persist()

    def phase(self):
        path=self.directory/'arm.json'
        if not path.exists():return {'armed':False,'ready':self.condition!='slow_loading','percent':0}
        marker=json.loads(path.read_text(encoding='utf-8-sig'));elapsed=max(0,time.time()*1000-marker['armed_at_ms'])
        return {'armed':True,'ready':self.condition!='slow_loading' or elapsed>=90000,
                'percent':min(100,int(elapsed/900)),'armed_at_ms':marker['armed_at_ms']}

    def handler(self):
        fixture=self
        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup();self.connection.settimeout(15)

            def log_message(self,*args):pass

            def reply(self,status,body,kind='application/json'):
                raw=(body if isinstance(body,str) else json.dumps(body,ensure_ascii=False)).encode('utf-8')
                self.send_response(status);self.send_header('Content-Type',kind+'; charset=utf-8')
                self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(raw)

            def do_GET(self):
                with fixture.lock:
                    if self.path=='/':self.reply(200,fixture.html,'text/html')
                    elif self.path=='/phase':self.reply(200,fixture.phase())
                    else:self.reply(404,{'error':'Page not found'})

            def do_POST(self):
                if self.path not in {'/event','/save'}:return self.reply(404,{'error':'Page not found'})
                if not self.headers.get('Content-Type','').startswith('application/json'):
                    return self.reply(415,{'error':'Unsupported request'})
                try:length=int(self.headers.get('Content-Length','0'))
                except ValueError:return self.reply(400,{'error':'Malformed request'})
                if not 0<length<=4096:return self.reply(413,{'error':'Request too large'})
                try:value=json.loads(self.rfile.read(length))
                except (ValueError,UnicodeError):return self.reply(400,{'error':'Malformed request'})
                if not isinstance(value,dict):return self.reply(400,{'error':'Unsupported fields'})
                with fixture.lock:
                    if self.path=='/event':
                        if set(value)!={'name'} or not isinstance(value['name'],str) or value['name'] not in EVENTS:
                            return self.reply(400,{'error':'Unknown event'})
                        fixture.event(value['name']);return self.reply(200,{'accepted':True})
                    phase=fixture.phase()
                    if not phase['armed'] or not phase['ready']:return self.reply(409,{'error':'Form is not ready'})
                    if (set(value)!={'name','reference'} or
                            any(not isinstance(v,str) or not v.strip() or len(v)>100 for v in value.values())):
                        return self.reply(400,{'error':'Check both fields'})
                    fixture.state['save_attempts'].append({'at':time.time(),'values':value})
                    fixture.event('save_attempt_received')
                    if fixture.condition=='no_progress':return self.reply(202,{'status':'pending'})
                    with (fixture.directory/'submissions.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps({'submitted_at':time.time(),'values':value},ensure_ascii=False)+'\n')
                        stream.flush();os.fsync(stream.fileno())
                    fixture.state['saved_count']+=1;fixture.state['saved']=value;fixture.event('request_saved')
                    self.reply(200,{'status':'saved'})
        return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--page',type=Path,required=True)
    parser.add_argument('--condition',choices=sorted(CONDITIONS),required=True)
    parser.add_argument('--port',type=int,default=8086)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('Use a nonprivileged local port')
    fixture=Fixture(args.directory,args.condition,args.page.read_text(encoding='utf-8'))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),fixture.handler())
    (args.directory/'server-ready.json').write_text(json.dumps({'pid':os.getpid(),'port':server.server_port,
        'bind':'127.0.0.1','source':str(Path(__file__).resolve())}),encoding='utf-8')
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
