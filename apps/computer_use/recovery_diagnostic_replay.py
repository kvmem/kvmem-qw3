"""Export finalized recovery videos and public metrics to a loopback-only replay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Windows 恢复与定位 · 连续录像对照</title>
<style>body{font:16px system-ui;background:#f5f6fa;color:#203047;max-width:1600px;margin:24px auto;padding:0 20px}h1{font-size:25px}p{line-height:1.7}.note{color:#59677b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:white;border:1px solid #d8dfe8;border-radius:12px;padding:16px;min-width:0}button,select{font:inherit;padding:10px;margin:5px;border-radius:6px;border:1px solid #bcc7d4;background:white}video{width:100%;aspect-ratio:1.6;background:#172436}pre{font:inherit;white-space:pre-wrap;line-height:1.65}small{word-break:break-all}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>Windows 恢复与定位：连续录像对照</h1>
<p>真实本地 QW3 · thinking 开启 · DFlash2 · KVMem 关闭 · 顾问关闭</p>
<p class="note">两侧按任务开始对齐。完成情况由真实保存文件和操作回执判定；正确产物但不确定暂停单独标注。录像为连续 VNC 采集，未经截图重建。</p>
<button id="play">从任务开始同时播放</button><button id="pause">暂停两侧</button><span id="error" role="status"></span>
<div class="grid"><section><select id="left" aria-label="左侧试次"></select><pre id="leftFacts"></pre><video id="leftVideo" muted controls preload="metadata"></video><small id="leftIdentity"></small></section><section><select id="right" aria-label="右侧试次"></select><pre id="rightFacts"></pre><video id="rightVideo" muted controls preload="metadata"></video><small id="rightIdentity"></small></section></div>
<p class="note">八类短任务各重复两次，不能代表通用软件可靠性。candidate 试次使用 client31；final 试次使用包含控件定位与焦点恢复提示的 client33。耗时包含任务失败和暂停，不把单次差异视为稳定加速比例。一次基线基础设施中断保留在原始记录，不将其残缺录像当作完整回放。</p>
<script>
const el=id=>document.getElementById(id);let trials=[],generation=0;
function chosen(s){return trials[Number(el(s).value)]}
function pause(){generation++;for(const s of ['left','right'])el(s+'Video').pause()}
function render(s){const t=chosen(s),g=t.grade;el(s+'Video').src=t.video;
el(s+'Facts').textContent=t.attempt+'\\n'+t.condition+' · '+t.cohort+' · '+g.outcome+'\\n无人干预完成：'+(g.unassisted_success?'通过':'未通过')+' · 产物正确：'+(g.artifact_correct?'是':'否')+' · 不确定暂停：'+(g.explicit_uncertainty_handoff?'是':'否')+'\\n任务 '+g.elapsed_seconds.toFixed(2)+' 秒 · 工具调用 '+g.tool_calls+' 次 · 工具错误 '+t.tool_errors+' 次\\n执行器拒绝 '+g.rejected_calls+' 次 · 保存 '+g.save_count+' 次 · 控件编号点击 '+t.control_click_calls+' 次';
el(s+'Identity').textContent='客户端 SHA-256：'+t.bundle_sha256+' · 视频 SHA-256：'+t.video_sha256}
async function seek(v,time){if(v.readyState<1)await new Promise((ok,fail)=>{v.addEventListener('loadedmetadata',ok,{once:true});v.addEventListener('error',fail,{once:true})});let x=Math.min(time,Math.max(0,v.duration-.1));if(Math.abs(v.currentTime-x)<.02)return;await new Promise(ok=>{v.addEventListener('seeked',ok,{once:true});v.currentTime=x})}
el('pause').onclick=pause;el('play').onclick=async()=>{pause();let turn=generation;el('error').textContent='';try{await Promise.all(['left','right'].map(s=>seek(el(s+'Video'),chosen(s).task_start_video_seconds)));if(turn!==generation)return;await Promise.all(['left','right'].map(s=>el(s+'Video').play()))}catch(e){if(turn===generation&&e.name!=='AbortError')el('error').textContent=e.message}};
fetch('trials.json').then(r=>{if(!r.ok)throw Error('无法读取试次');return r.json()}).then(data=>{trials=data;if(!trials.length)throw Error('没有完整录像');let final=trials.findIndex(t=>t.attempt.startsWith('p3-final-'));let right=final>=0?final:Math.max(0,trials.findIndex(t=>t.cohort==='candidate'));for(const s of ['left','right']){trials.forEach((t,i)=>{let o=document.createElement('option');o.value=i;o.textContent=t.attempt+' · '+(t.grade.unassisted_success?'通过':t.grade.explicit_uncertainty_handoff?'不确定暂停':'未通过');el(s).append(o)});el(s).value=s==='left'?0:right;el(s).onchange=()=>{pause();render(s)};render(s)}}).catch(e=>el('error').textContent=e.message);
</script></html>'''


def export(audits, output, runtime, port):
    output.mkdir(parents=True, exist_ok=False)
    trials = []
    seen = set()
    for path in audits:
        audit = json.loads(path.read_text())
        if not audit.get('evidence_consistent'):
            raise ValueError('Only audited attempts may be exported')
        for trial in audit['trials']:
            name = trial['attempt']
            if name in seen or Path(name).name != name:
                raise ValueError('Unsafe or duplicate attempt identity')
            seen.add(name)
            root = runtime / name
            result = json.loads((root / 'result.json').read_text())
            video = root / 'video/screen.mp4'
            if not result['recording']['complete'] or hashlib.sha256(video.read_bytes()).hexdigest() != trial['video_sha256']:
                raise ValueError('Video is incomplete or changed after audit')
            public = {k: trial[k] for k in ('attempt', 'cohort', 'condition', 'grade', 'tool_errors', 'bundle_sha256', 'video_sha256')}
            public.update(video=f'trial-{len(trials)}.mp4', control_click_calls=trial['action_kinds'].count('click_control'),
                task_start_video_seconds=max(0, result['native_result']['task']['started_unix_ns']/1e9-result['recording']['started_unix_ns']/1e9))
            shutil.copyfile(video, output / public['video'])
            trials.append(public)
    (output / 'trials.json').write_text(json.dumps(trials, ensure_ascii=False, indent=2)+'\n')
    (output / 'index.html').write_text(HTML)
    (output / 'serve.py').write_text("from pathlib import Path\nfrom aiohttp import web\nroot=Path(__file__).resolve().parent\napp=web.Application()\napp.router.add_static('/',root,show_index=False)\nweb.run_app(app,host='127.0.0.1',port="+str(port)+",print=None)\n")
    return len(trials)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audits', type=Path, nargs='+', required=True)
    parser.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parent/'.runtime')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18123)
    args=parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Invalid loopback port')
    print('Exported',export(args.audits,args.output,args.runtime,args.port),'continuous recordings')


if __name__=='__main__':main()
