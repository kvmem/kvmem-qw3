"""Export only finalized native-trial videos and public outcome summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Windows 完整任务对照</title><style>body{font:16px system-ui;background:#f5f6fa;color:#203047;max-width:1600px;margin:25px auto;padding:0 22px}h1{font-size:26px}p{line-height:1.7}.note{color:#59677b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:white;border:1px solid #d8dfe8;border-radius:12px;padding:18px;min-width:0}button,select{font:inherit;padding:10px;margin:6px;border-radius:6px;border:1px solid #bcc7d4;background:white}video{width:100%;aspect-ratio:1.6;background:#172436}pre{font:inherit;white-space:pre-wrap;line-height:1.65}small{word-break:break-all}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>Windows 完整任务：连续录像对照</h1><p>真实本地 QW3 · 原生 Codex · GUI 操作 · 独立文件判分</p>
<p class="note">每个试次的版本与 thinking 状态分别列出。按任务开始对齐；动作轨迹可能不同，视频进度不代表单次推理加速比。失败和超时按原结果显示。</p>
<button id="play">从任务开始同时播放</button><button id="pause">暂停两侧</button><span id="error"></span>
<div class="grid"><section><select id="left"></select><pre id="leftFacts"></pre><video id="leftVideo" muted controls preload="metadata"></video><small id="leftIdentity"></small></section><section><select id="right"></select><pre id="rightFacts"></pre><video id="rightVideo" muted controls preload="metadata"></video><small id="rightIdentity"></small></section></div>
<p class="note">这里展示探索性复测，不是原十场景的正式 V0/V1 成绩，也不是稳定成功率。视频为连续 VNC 录制，未经内容重建。</p>
<script>
const el=id=>document.getElementById(id);let trials=[],generation=0;
function selected(side){return trials[Number(el(side).value)]}
function pause(){generation++;for(const side of ['left','right'])el(side+'Video').pause()}
function render(side){const t=selected(side);el(side+'Video').src=t.video;el(side+'Facts').textContent=t.run+'\\n'+(t.passed?'独立验收通过':'未通过独立验收')+' · '+t.outcome+'\\nthinking '+(t.thinking?'开启':'关闭')+' · KVMem '+t.kvmem+'\\n任务 '+t.task_seconds.toFixed(2)+' 秒 · 模型请求 '+t.model_requests+' 次 · 工具调用 '+t.tool_calls+' 次\\n验收项目 '+t.checks_passed+'/'+t.checks_total+' · 归一化坐标动作 '+t.normalized_actions+' 次';el(side+'Identity').textContent='客户端 SHA-256：'+t.bundle_sha256+' · 结果 SHA-256：'+t.result_sha256}
async function seek(v,time){if(v.readyState<1)await new Promise((ok,fail)=>{v.addEventListener('loadedmetadata',ok,{once:true});v.addEventListener('error',fail,{once:true})});const t=Math.min(time,Math.max(0,v.duration-.1));if(Math.abs(t-v.currentTime)<.02)return;await new Promise(ok=>{v.addEventListener('seeked',ok,{once:true});v.currentTime=t})}
el('pause').onclick=pause;el('play').onclick=async()=>{pause();const turn=generation;el('error').textContent='';try{await Promise.all(['left','right'].map(s=>seek(el(s+'Video'),selected(s).task_start_video_seconds)));if(turn!==generation)return;await Promise.all(['left','right'].map(s=>el(s+'Video').play()))}catch(e){if(turn===generation&&e.name!=='AbortError')el('error').textContent=e.message}};
fetch('trials.json').then(r=>{if(!r.ok)throw Error('无法读取试次');return r.json()}).then(data=>{trials=data;for(const side of ['left','right']){trials.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.textContent=t.run;el(side).append(o)});el(side).value=side==='left'?0:trials.length-1;el(side).onchange=()=>{pause();render(side)};render(side)}}).catch(e=>el('error').textContent=e.message);
</script></html>'''


def summarize(root):
    import sqlite3
    r=json.loads((root/'result.json').read_text());task=r['native_result']['task'];state=task['terminal_observation'];grade=r.get('independent_grade',{});checks=grade.get('checks',{})
    normalized=0
    for db in (root/'guest').rglob('desktop.sqlite'):
        conn=sqlite3.connect('file:'+str(db.resolve())+'?mode=ro',uri=True)
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coordinate_dispatches'").fetchone():normalized+=conn.execute('select count(*) from coordinate_dispatches').fetchone()[0]
        conn.close()
    return {'run':root.name,'task':r['task'],'passed':r.get('pipeline_passed') is True and grade.get('strict_success') is True,
            'outcome':task['outcome'],'thinking':state.get('thinking',False),'kvmem':r['kvmem'],
            'task_seconds':task['elapsed_seconds'],'tool_calls':state.get('tool_calls_used',0),
            'model_requests':r.get('inference_request_receipts',0),'normalized_actions':normalized,
            'checks_passed':sum(v is True for v in checks.values()),'checks_total':len(checks),
            'bundle_sha256':r['bundle_install_identity']['archive']['sha256'],
            'result_sha256':hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
            'task_start_video_seconds':max(0,state['started_at']-r['recording']['started_unix_ns']/1e9)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runs',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--port',type=int,default=18116);args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('Invalid local port')
    args.output.mkdir(parents=True,exist_ok=False);trials=[]
    for i,root in enumerate(args.runs):
        receipt=json.loads((root/'result.json').read_text());recording=receipt['recording']
        if recording.get('complete') is not True:raise ValueError('Video is not finalized')
        video=root/'video/screen.mp4'
        if hashlib.sha256(video.read_bytes()).hexdigest()!=recording['artifacts']['screen.mp4']['sha256']:raise ValueError('Final video hash mismatch')
        name=f'trial-{i}.mp4';shutil.copyfile(video,args.output/name);trials.append(summarize(root)|{'video':name})
    (args.output/'trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2)+'\n');(args.output/'index.html').write_text(HTML)
    (args.output/'serve.py').write_text("from pathlib import Path\nfrom aiohttp import web\nroot=Path(__file__).resolve().parent\napp=web.Application()\napp.router.add_static('/',root,show_index=False)\nweb.run_app(app,host='127.0.0.1',port="+str(args.port)+",print=None)\n")
    print(args.output/'index.html')


if __name__=='__main__':main()
