#!/usr/bin/env python3
"""Build an offline, synchronized observation replay from two closed archive runs.

This reconstructs retained screenshots, not continuous video. It never connects
to the desktop or model. All times use the host client's stored task clock.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(run, policy, output):
    task = read(run / 'app/task.json')
    result = read(run / 'task/result.json')
    suite = read(run / 'summary.json')
    engine = read(run / 'engine/manifest.json')
    if suite['status'] not in ('completed', 'failed') or task['status'] not in ('completed', 'stopped', 'failed'):
        raise ValueError('Replay requires a closed task and suite')
    if suite['kvmem'] != policy or engine['kvmem'] != policy:
        raise ValueError('Policy label does not match retained evidence')
    started = task['started_at']
    duration = result['elapsed_seconds']
    db = run / 'app/desktop/desktop.sqlite'
    before = digest(db)
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as connection:
        observations = list(connection.execute(
            'SELECT id,metadata,created FROM observations WHERE task=? ORDER BY created,id', (task['id'],)))
        actions = list(connection.execute(
            'SELECT id,request,status,result,created FROM actions WHERE task=? ORDER BY created,id', (task['id'],)))
    if digest(db) != before:
        raise ValueError('Source database changed during replay export')
    frames, events = [], []
    observed_at = {identifier: created for identifier, _, created in observations}
    target = output / policy
    target.mkdir()
    for identifier, raw, created in observations:
        metadata = json.loads(raw)
        sha = metadata['sha256']
        if not re.fullmatch('[a-f0-9]{64}', sha):
            raise ValueError('Unexpected screenshot digest')
        source = run / 'app/desktop/screenshots' / (sha + '.png')
        if digest(source) != sha:
            raise ValueError('Screenshot does not match observation digest')
        destination = target / source.name
        if not destination.exists():
            shutil.copyfile(source, destination)
        frames.append({'t': max(0, created - started), 'id': identifier,
                       'src': policy + '/' + source.name, 'sha256': sha})
    if not frames:
        raise ValueError('No retained observations')
    for identifier, raw, status, receipt, created in actions:
        request = json.loads(raw)
        record = {'t': max(0, created - started), 'id': identifier,
                  'action': request['action'], 'final_status': status, 'completed_at': None}
        if status == 'completed' and receipt:
            after = json.loads(receipt).get('observation', {}).get('observation_id')
            if after in observed_at:
                record['completed_at'] = max(0, observed_at[after] - started)
        events.append(record)
    inferences = []
    for path in (run / 'app/inference').glob('*/metrics.json'):
        metric = read(path)
        if 'elapsed_seconds' not in metric:
            continue
        inferences.append({'t': max(0, metric['started_at'] - started),
                           'end': max(0, metric['started_at'] + metric['elapsed_seconds'] - started),
                           'status': metric['status'], 'usage': metric.get('usage', {})})
    inferences.sort(key=lambda value: value['t'])
    evidence_paths = ['summary.json', 'task/result.json', 'app/task.json', 'engine/manifest.json']
    return {'policy': policy, 'duration': duration,
            'timeline_end': max(duration, frames[-1]['t']), 'passed': result['passed'],
            'error': result.get('error'), 'frames': frames, 'actions': events,
            'inferences': inferences, 'binary_sha256': engine['binary_sha256'],
            'snapshot': suite['snapshot'], 'task_timeout': suite['task_timeout'],
            'prompt_sha256': hashlib.sha256(task['prompt'].encode()).hexdigest(),
            'fixture_sha256': result['fixture_sha256'],
            'engine_profile': {key: engine[key] for key in
                               ('prefix_reuse', 'guided_refresh', 'visual_history')},
            'source': str(run), 'source_hashes': {name: digest(run / name) for name in evidence_paths},
            'database_sha256': before}


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KVMem 开关：观测帧双栏回放</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#192535;font:15px system-ui,sans-serif}
main{max-width:1600px;margin:auto;padding:24px}h1{font-size:25px;margin:0 0 10px}
.notice{padding:14px;background:#fff0cb;border:1px solid #e1c772;border-radius:8px;line-height:1.6}
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0}
button,select{font:inherit;padding:7px 12px;background:white;border:1px solid #bac4d1;border-radius:6px}
input{flex:1;min-width:200px}.clock{font-variant-numeric:tabular-nums;font-size:18px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:white;border:1px solid #d4dce6;border-radius:10px;overflow:hidden}
.head{display:flex;justify-content:space-between;padding:13px 16px;font-weight:650}.status{color:#596879}
.passed{color:#087c46}.failed{color:#b3392f}img{display:block;width:100%;aspect-ratio:1.6;object-fit:contain;background:#19212c}
.meta{padding:14px 16px;line-height:1.7}.action{font-family:ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;min-height:60px}
.small{font-size:12px;color:#647182;line-height:1.7}.footer{margin-top:16px}.links a{margin-right:15px}
@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:12px}}
</style><main>
<h1>KVMem 开关：36 页任务双栏回放</h1>
<div class="notice"><b>已有观测帧重建，非连续录屏。</b>没有截图的间隔保持上一帧，不能据此判断桌面没有变化。
两侧从各自任务启动时间对齐，保留真实等待时长；完成的一侧停留在最后观测画面。</div>
<div class="controls"><button id="play">播放</button><button id="reset">归零</button>
<select id="speed" aria-label="回放倍速"><option value="1">1× 原速</option><option value="4">4×</option><option value="16">16×</option></select>
<input id="seek" aria-label="共同时间轴" type="range" min="0" step="any"><span class="clock" id="clock"></span></div>
<div class="grid">
<section class="panel" id="on"><div class="head"><span>KVMem 开启</span><span class="status"></span></div><img alt="开启侧保留的桌面观测"><div class="meta"></div></section>
<section class="panel" id="off"><div class="head"><span>KVMem 关闭</span><span class="status"></span></div><img alt="关闭侧保留的桌面观测"><div class="meta"></div></section>
</div><div class="footer small">时间来自客户端保存的任务、观测和请求记录，未使用 Windows 来宾机的墙上时钟。
截图时间为客户端入库时间，不是精确的视频采集时刻。推理 token 仅显示已经结束的请求。
本轮开启侧包含前缀复用；两侧动作路径不同。该回放不构成纯检索加速比或可靠性统计。</div>
<p class="links"><a href="manifest.json">证据及来源清单</a><a href="timeline.json">完整回放时间轴</a></p>
</main><script>const DATA=__DATA__;
const maxTime=Math.max(DATA.on.timeline_end,DATA.off.timeline_end), seek=document.querySelector('#seek');
seek.max=maxTime;let now=Math.min(maxTime,Math.max(0,Number(new URLSearchParams(location.search).get('t'))||0)),playing=false,last=0;
const fmt=t=>`${Math.floor(t/60).toString().padStart(2,'0')}:${(t%60).toFixed(1).padStart(4,'0')}`;
const lastBefore=(xs,t,key='t')=>xs.filter(x=>x[key]<=t).at(-1);
function line(parent,text,cls=''){const p=document.createElement('div');p.className=cls;p.textContent=text;parent.append(p)}
function render(){seek.value=now;document.querySelector('#clock').textContent=fmt(now)+' / '+fmt(maxTime);
 for(const mode of ['on','off']){const run=DATA[mode],p=document.getElementById(mode),ended=now>=run.duration;
  const frame=lastBefore(run.frames,now),img=p.querySelector('img');
  if(frame){if(img.getAttribute('src')!==frame.src)img.src=frame.src}else{img.removeAttribute('src')}
  const status=p.querySelector('.status');status.textContent=ended?(run.passed?'已完成 · '+fmt(run.duration):'未通过 · '+fmt(run.duration)):'任务进行中';
  status.className='status '+(ended?(run.passed?'passed':'failed'):'');
  const meta=p.querySelector('.meta');meta.replaceChildren();
  line(meta,frame?'当前显示观测：'+fmt(frame.t)+'；距此观测 '+Math.max(0,now-frame.t).toFixed(1)+' 秒':'等待首张桌面观测');
  const completed=run.actions.filter(a=>a.completed_at!==null&&a.completed_at<=now).length;
  line(meta,'已确认完成动作：'+completed+' / '+run.actions.filter(a=>a.final_status==='completed').length);
  const action=lastBefore(run.actions,now);line(meta,action?'最近发起动作：'+JSON.stringify(action.action):'尚未发起动作','action');
  if(action)line(meta,'该动作在最终日志中的状态：'+action.final_status,'small');
  const done=run.inferences.filter(r=>r.end<=now&&r.status==='completed').at(-1);
  line(meta,done?'最近完成请求输入：'+(done.usage.input_tokens||0).toLocaleString()+' tokens':'尚无完成的推理请求');
  if(ended&&run.error)line(meta,'终止原因：'+run.error,'small');
 }}
document.querySelector('#play').onclick=()=>{if(now>=maxTime)now=0;playing=!playing;document.querySelector('#play').textContent=playing?'暂停':'播放';last=performance.now();render()};
seek.oninput=()=>{now=Number(seek.value);render()};document.querySelector('#reset').onclick=()=>{now=0;render()};
function tick(t){if(playing){now=Math.min(maxTime,now+(t-last)/1000*Number(document.querySelector('#speed').value));render();if(now>=maxTime){playing=false;document.querySelector('#play').textContent='播放'}}last=t;requestAnimationFrame(tick)}
render();requestAnimationFrame(tick);
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = {mode: collect(getattr(args, mode).resolve(), mode, output) for mode in ('on', 'off')}
    for field in ('binary_sha256', 'snapshot', 'task_timeout', 'prompt_sha256', 'fixture_sha256'):
        if data['on'][field] != data['off'][field]:
            raise ValueError('Paired replay mismatch: ' + field)
    serialized = json.dumps(data, ensure_ascii=False)
    (output / 'timeline.json').write_text(serialized + '\n')
    (output / 'index.html').write_text(HTML.replace('__DATA__', serialized.replace('<', '\\u003c')))
    manifest = {'kind': 'retained_observation_comparison_replay', 'continuous_video': False,
                'clock': 'Per-run client task.started_at; host observation insertion times',
                'sources': {mode: {key: value for key, value in run.items()
                            if key not in ('frames', 'actions', 'inferences')} for mode, run in data.items()},
                'frame_counts': {mode: len(run['frames']) for mode, run in data.items()},
                'builder_sha256': digest(Path(__file__)),
                'timeline_sha256': digest(output / 'timeline.json'), 'html_sha256': digest(output / 'index.html')}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(output / 'index.html')


if __name__ == '__main__':
    main()
