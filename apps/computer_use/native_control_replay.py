"""Publish audited native-control diagnostic videos with their measured scope."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>保存操作与故障恢复 · 验证回放</title>
<style>body{font:16px system-ui;max-width:1440px;margin:24px auto;padding:0 20px;background:#f4f6fa;color:#23334a}h1{font-size:26px}p{line-height:1.7}.note{color:#58677c}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:white;padding:18px;border:1px solid #dce2ed;border-radius:12px;min-width:0}video{width:100%;aspect-ratio:16/9;background:#172236}select,button{font:inherit;padding:10px;border:1px solid #bbc6d8;border-radius:6px;background:white;margin:5px 4px}pre{font:inherit;white-space:pre-wrap;line-height:1.7}#error{color:#a22}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>保存操作与故障恢复：真实虚拟机验证</h1>
<p>固定 Gedit 保存场景 · 工具层验证 · 未调用 QW3</p>
<p>第二次验证通过 12 项检查：原生保存按钮打开对话框，重复请求没有再次输入，焦点故障在输入前被拒绝，恢复后保存的文件内容正确。第一次保留了恢复步骤被重复保护阻止的失败。</p>
<p class="note">录屏每秒采样 2 帧，非截图重建；采样间隔不能用于精确延迟比较。默认从保存操作前开始播放。本页结果不计入模型任务或 OSWorld 成功率。</p>
<button id="play">同时播放两次验证</button><button id="pause">暂停</button><span id="error" role="status"></span>
<div class="grid"><section><select id="left" aria-label="左侧验证"></select><pre id="leftFacts"></pre><video id="leftVideo" muted controls preload="metadata"></video></section><section><select id="right" aria-label="右侧验证"></select><pre id="rightFacts"></pre><video id="rightVideo" muted controls preload="metadata"></video></section></div>
<p class="note">按钮之外的界面变化由原生对象身份、按钮状态和实际结果核验处理。原策略的原因隔离使用同一对保留画面做离线只读检查；不是第二次真实旧流程运行。</p>
<script>
const el=id=>document.getElementById(id);let trials=[],generation=0;
function selected(side){return trials[Number(el(side).value)]}
function pause(){generation++;for(const s of ['left','right'])el(s+'Video').pause()}
function render(s){const t=selected(s);el(s+'Video').src=t.video;el(s+'Facts').textContent=t.label+'\\n'+t.outcome+'\\n检查通过：'+t.checks_passed+' · 模型调用：0\\n'+t.note}
async function seek(video,value){if(video.readyState<1)await new Promise((ok,fail)=>{video.addEventListener('loadedmetadata',ok,{once:true});video.addEventListener('error',fail,{once:true})});let target=Math.min(value,Math.max(0,video.duration-.1));if(Math.abs(video.currentTime-target)<.02)return;await new Promise(ok=>{video.addEventListener('seeked',ok,{once:true});video.currentTime=target})}
el('pause').onclick=pause;el('play').onclick=async()=>{pause();const g=generation;el('error').textContent='';try{await Promise.all(['left','right'].map(s=>seek(el(s+'Video'),selected(s).start_seconds)));if(g!==generation)return;await Promise.all(['left','right'].map(s=>el(s+'Video').play()))}catch(e){if(g===generation&&e.name!=='AbortError')el('error').textContent='播放失败：'+e.message}};
fetch('summary.json').then(r=>{if(!r.ok)throw Error('无法读取回放');return r.json()}).then(data=>{trials=data.trials;for(const s of ['left','right']){trials.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.textContent=t.label;el(s).append(o)});el(s).value=s==='left'?0:1;el(s).onchange=()=>{pause();render(s)};render(s)}}).catch(e=>el('error').textContent=e.message);
</script></html>'''


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(audit_path,output,port):
    audit=json.loads(audit_path.read_text())
    if not audit['checks'] or not all(audit['checks'].values()):raise ValueError('Require independent completed audit')
    names=['native-control-delivery-live-01','native-control-delivery-live-02','native-control-interface-probe-live-01']
    labels=['第一次验证 · 恢复流程未完成','第二次验证 · 文件保存正确','原生接口探测 · 未激活按钮']
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name,label in zip(names,labels):
        evidence=audit['attempts'][name];result_path=Path(evidence['result']['path'])
        if digest(result_path)!=evidence['result']['sha256']:raise ValueError('Audited result changed')
        result=json.loads(result_path.read_text());video=Path(evidence['video']['path'])
        if digest(video)!=evidence['video']['sha256']:raise ValueError('Audited video changed')
        relative='trial-'+str(len(rows))+'.mp4';shutil.copyfile(video,output/relative)
        offset=0
        if result.get('inputs'):
            offset=max(0,result['inputs'][0]['started_at']-result['recording']['started_unix_ns']/1e9-3)
        rows.append(dict(label=label,video=relative,video_sha256=digest(video),start_seconds=offset,
            checks_passed=evidence['checks_passed'],outcome='工具验证通过' if result['passed'] else '工具验证未完成',
            note='2帧/秒采样录屏；固定场景，不代表模型任务成功率。',model_calls=0))
    summary=dict(kind='public_native_control_diagnostic_replay',contains_credentials=False,contains_private_requests=False,
                 independent_audit_sha256=digest(audit_path),trials=rows,model_calls=0,business_success_claimed=False)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'index.html').write_text(HTML)
    (output/'serve.py').write_text("from pathlib import Path\nfrom aiohttp import web\nroot=Path(__file__).resolve().parent\napp=web.Application()\napp.router.add_static('/',root,show_index=False)\nweb.run_app(app,host='127.0.0.1',port="+str(port)+",print=None)\n")
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18165)
    args=p.parse_args()
    if not 1024<=args.port<=65535:p.error('Choose a valid loopback port')
    print(json.dumps(dict(exported_videos=len(export(args.audit.resolve(),args.output.resolve(),args.port)['trials']))))
