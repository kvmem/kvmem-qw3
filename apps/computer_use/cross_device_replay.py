"""Export terminal cross-device attempts without private requests or credentials.

The export is a point-in-time snapshot. Missing videos and unavailable official
scores remain explicit rows; development pilots never become OSWorld scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>跨设备 Computer Use · 实测对照</title>
<style>
body{font:16px system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1600px;background:#f5f6fa;color:#203047}
h1{font-size:26px}p{line-height:1.7}.note,small{color:#59677b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
section{background:white;border:1px solid #d8dfe8;border-radius:12px;padding:18px;min-width:0}
select,button{font:inherit;padding:9px;margin:6px 8px 6px 0;border:1px solid #bcc7d4;border-radius:6px;background:white;max-width:100%}
video{width:100%;height:480px;object-fit:contain;background:#172436}pre{font:inherit;white-space:pre-wrap;line-height:1.65}
small{word-break:break-all}#error{color:#982a26}@media(max-width:850px){.grid{grid-template-columns:1fr}video{height:380px}}
</style>
<h1>跨设备 Computer Use：实测录像对照</h1>
<p id="summary"></p>
<p class="note">正式 OSWorld 与平台开发任务分别显示。基础设施失败或原评分不可用不计成模型失败；缺少录像的尝试仍保留。这里是导出时的快照，后续尝试不会自动追加。</p>
<select id="task" aria-label="测试场景"></select>
<button id="play">从录像开始同时播放</button><button id="pause">暂停两侧</button><span id="error" role="status"></span>
<div class="grid">
<section><select id="left" aria-label="左侧尝试"></select><pre id="leftFacts"></pre><video id="leftVideo" muted controls preload="metadata"></video><p id="leftMissing"></p><small id="leftIdentity"></small></section>
<section><select id="right" aria-label="右侧尝试"></select><pre id="rightFacts"></pre><video id="rightVideo" muted controls preload="metadata"></video><p id="rightMissing"></p><small id="rightIdentity"></small></section>
</div>
<p class="note">Linux 录像分别标记来源，时长不足的官方录像只能查看保留片段。Android 是实时 1Hz 采样录像，可能遗漏亚秒变化。两侧从录像起点播放，准备时间和操作轨迹可能不同；不能把进度条当成单次推理加速比。任务耗时、调用数与独立核验结果分别列出。</p>
<script>
const el=id=>document.getElementById(id);let rows=[],all=[],generation=0;
const chosen=side=>rows[Number(el(side).value)];
const yn=x=>x===true?'通过':x===false?'未通过':'不可用';
const sec=x=>typeof x==='number'?x.toFixed(2)+' 秒':'未取得';
function pause(){generation++;for(const side of ['left','right'])el(side+'Video').pause()}
function render(side){const t=chosen(side),v=el(side+'Video');v.pause();
 if(t.video){v.src=t.video;v.hidden=false;el(side+'Missing').textContent=''}else{v.removeAttribute('src');v.load();v.hidden=true;el(side+'Missing').textContent='未取得可用录像：'+t.media_status}
 const result=t.kind==='formal-osworld'?'官方评分：'+(t.original_score??'不可用')+' · 无接管通过：'+yn(t.unassisted_passed)+(t.secondary_local_score!=null?'\\n单列本地工作簿指标：'+t.secondary_local_score+'（不是官方分数）':''):'独立产物核验：'+yn(t.artifact_passed)+' · 完整验收：'+yn(t.accepted);
 el(side+'Facts').textContent=t.attempt+'\\n'+(t.kind==='formal-osworld'?'正式 OSWorld · Linux':t.kind==='software-product'?'软件配置验收 · Windows · 受控软件工具':'平台开发任务 · '+t.platform)+' · '+t.arm+'\\n'+result+'\\n任务状态：'+t.agent_status+' · 记录状态：'+t.outcome+'\\n任务耗时：'+sec(t.seconds)+' · 工具调用：'+t.tool_calls+' · 未知动作：'+t.uncertain_actions+'\\n输出预算：'+(t.model_output_tokens??'未记录')+' token · KVMem：'+t.kvmem+' · 视频：'+t.media_kind+'\\n录像时长：'+sec(t.video_duration_seconds)+' · '+t.time_coverage;
 if(t.fault_condition)el(side+'Facts').textContent+='\\n故障：'+t.fault_condition+' · 故障处理验收：'+yn(t.fault_accepted)+(t.fault_condition==='no_progress'?'（安全暂停；业务完成另计）':'');
 el(side+'Identity').textContent='原结果 SHA256：'+t.result_sha256+(t.video_sha256?' · 导出视频 SHA256：'+t.video_sha256:'');
}
function selectTask(){pause();rows=all.filter(t=>t.task===el('task').value);
 for(const side of ['left','right']){el(side).replaceChildren();rows.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.textContent=t.attempt;el(side).append(o)});
 el(side).value=side==='left'?0:rows.length-1;render(side)}}
async function rewind(v){if(v.readyState<1)await new Promise((ok,fail)=>{v.addEventListener('loadedmetadata',ok,{once:true});v.addEventListener('error',fail,{once:true})});
 if(v.currentTime<.02)return;await new Promise(ok=>{v.addEventListener('seeked',ok,{once:true});v.currentTime=0})}
el('pause').onclick=pause;el('play').onclick=async()=>{pause();const turn=generation;el('error').textContent='';
 const sides=['left','right'].filter(side=>chosen(side).video);
 if(!sides.length){el('error').textContent='当前尝试没有可播放录像';return}
 try{await Promise.all(sides.map(side=>rewind(el(side+'Video'))));if(turn!==generation)return;
 await Promise.all(sides.map(side=>el(side+'Video').play()))}catch(e){if(turn===generation&&e.name!=='AbortError')el('error').textContent='播放失败：'+e.message}};
for(const side of ['left','right'])el(side).onchange=()=>{pause();render(side)};
el('task').onchange=selectTask;
fetch('summary.json').then(r=>{if(!r.ok)throw Error('无法读取导出记录');return r.json()}).then(data=>{
 all=data.attempts;const s=data.formal_summary;
 el('summary').textContent='本地 QW3 · 顾问关闭 · 正式尝试 '+s.terminal_attempts+' 次，其中有效官方评分 '+s.available_original_scores+' 次、评分通过 '+s.original_passes+' 次 · 平台开发尝试 '+data.development_attempts+' 次 · 软件配置验收 '+(data.software_product_attempts||0)+' 次';
 for(const task of [...new Set(all.map(t=>t.task))]){const o=document.createElement('option');o.value=task;o.textContent=task;el('task').append(o)}
 if(all.length)selectTask();else el('error').textContent='没有终止记录';
}).catch(e=>el('error').textContent=e.message);
</script></html>'''


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def media_details(path, seconds):
    result={'video_duration_seconds':None,'time_coverage':'时长未核验'}
    try:
        import av
    except ImportError:
        result.update(time_coverage='视频解码依赖不可用，时长未核验，禁止延迟对比',media_decode_error='missing_dependency')
        return result
    try:
        with av.open(str(path)) as source:
            next(source.decode(video=0))
            duration=float(source.duration)/av.time_base if source.duration is not None else None
        result['video_duration_seconds']=duration
        if duration is not None and seconds is not None:
            result['time_coverage']='录像时长不足，禁止延迟对比' if duration+10<seconds else '时长足够，起止对齐仍未核验'
    except Exception as exc:
        result.update(time_coverage='视频解码失败',media_decode_error=type(exc).__name__)
    return result


def formal_rows(campaign):
    for job in read(campaign / 'plan.json')['jobs']:
        root=campaign/job['id']
        result_path=root/'result.json'
        if not result_path.exists():
            continue
        r=read(result_path)
        if not r.get('finished_at'):
            continue
        m=read(root/'manifest.json')
        state=read(root/'final_agent_state.json') if (root/'final_agent_state.json').exists() else {}
        yield root,dict(attempt=job['id'],task=m['task_id'],kind='formal-osworld',platform='linux',
            arm=m['arm'],outcome=r['status'],agent_status=r.get('agent_status'),
            original_score=r.get('score') if r['status']=='scored' else None,
            secondary_local_score=r.get('secondary_local_score'),
            secondary_local_passed=r.get('secondary_local_passed'),
            unassisted_passed=r.get('unassisted_passed'),seconds=r.get('seconds'),
            tool_calls=r.get('tool_calls'),uncertain_actions=r.get('uncertain_actions'),
            model_output_tokens=state.get('model_output_tokens'),kvmem='on' if m['kvmem'] else 'off',
            media_kind='官方连续桌面录屏',result_sha256=digest(result_path),video_bytes=r.get('video_bytes',0))


def pilot_row(root):
    r=read(root/'integration-result.json');m=read(root/'pilot-manifest.json')
    identity=root.parent.name if root.name=='pilot' else root.name
    fault=r.get('fault_condition')
    return root,dict(attempt=identity,task=m['case']+('__'+fault if fault else ''),kind='development-pilot',
        platform=m['case'].split('_',1)[0],arm='verified候选',
        outcome='accepted' if r.get('accepted') else 'not-accepted',agent_status=r.get('app_status'),
        artifact_passed=r.get('artifact_passed'),accepted=r.get('accepted'),
        seconds=r.get('seconds'),tool_calls=r.get('tool_calls'),uncertain_actions=r.get('uncertain_actions'),
        model_output_tokens=m.get('model_output_tokens'),kvmem=m.get('kvmem','未记录'),
        media_kind={'live':'实时1Hz采样','vnc':'真实时间戳VNC采样','official':'官方录像'}.get(m.get('video_mode'),m.get('video_mode')),
        result_sha256=digest(root/'integration-result.json'),video_bytes=r.get('video_bytes',0),
        fault_condition=fault,fault_response_passed=r.get('fault_response_passed'),fault_accepted=r.get('fault_accepted'))


def product_row(root):
    r=read(root/'result.json')
    if not r.get('finished_at'):raise ValueError('Cannot export a running Windows product trial')
    guest=r.get('guest_result',{});task=guest.get('task',{});state=guest.get('terminal_state',{})
    recording=r.get('recording',{});video=recording.get('artifacts',{}).get('screen.mp4',{})
    return root,dict(attempt=root.name,task=r['task'],kind='software-product',platform='windows',
        arm=r.get('workflow_mode'),outcome=r['status'],agent_status=state.get('status',task.get('outcome')),
        artifact_passed=guest.get('grade',{}).get('passed'),
        accepted=r.get('passed') is True and recording.get('complete') is True and guest.get('desktop_stopped') is True,
        seconds=task.get('elapsed_seconds'),tool_calls=state.get('tool_calls_used'),
        uncertain_actions=len(state.get('uncertain_actions',[])),model_output_tokens=r.get('model_output_tokens'),
        kvmem=r.get('engine',{}).get('kvmem','未记录'),media_kind='真实时间戳VNC采样',
        result_sha256=digest(root/'result.json'),video_bytes=video.get('bytes',0),
        expected_video_sha256=video.get('sha256'))


def export(campaign, pilots, output, port, products=()):
    rows=list(formal_rows(campaign)) if campaign else []
    rows.extend(pilot_row(root) for root in pilots)
    rows.extend(product_row(root) for root in products)
    output.mkdir(parents=True,exist_ok=False)
    exported=[]
    for index,(root,row) in enumerate(rows):
        source=root/'video/screen.mp4' if row['kind']=='software-product' else root/'continuous.mp4'
        row.update(video=None,video_sha256=None,media_status='未收集或收集失败',video_duration_seconds=None,time_coverage='未取得录像')
        if row['video_bytes'] and source.is_file():
            with source.open('rb') as file:
                header=file.read(12)
            if source.stat().st_size!=row['video_bytes'] or header[4:8]!=b'ftyp':
                row['media_status']='录像大小或文件格式不匹配'
            elif row.get('expected_video_sha256') and digest(source)!=row['expected_video_sha256']:
                row['media_status']='录像哈希与原始录制回执不匹配'
            else:
                name=f'trial-{index}.mp4';shutil.copyfile(source,output/name)
                row.update(video=name,video_sha256=digest(output/name),media_status='已收集')
                row.update(media_details(output/name,row['seconds']))
        exported.append(row)
    official=[r for r in exported if r['kind']=='formal-osworld']
    summary={'kind':'public-terminal-attempts-snapshot','exported_at':time.time(),
             'contains_private_requests':False,'contains_credentials':False,
             'formal_summary':{'terminal_attempts':len(official),
                 'available_original_scores':sum(r['original_score'] is not None for r in official),
                 'original_passes':sum(r['original_score']==1 for r in official),
                 'infrastructure_failures':sum(r['outcome']=='infrastructure-failed' for r in official),
                 'unavailable_original_evaluators':sum(r['outcome']=='evaluator-unavailable' for r in official)},
             'development_attempts':sum(row['kind']=='development-pilot' for row in exported),
             'software_product_attempts':sum(row['kind']=='software-product' for row in exported),'attempts':exported}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'index.html').write_text(HTML)
    # The local server exposes only copied public replay assets.
    (output/'serve.py').write_text('from pathlib import Path\nfrom aiohttp import web\n'
        "root=Path(__file__).resolve().parent\napp=web.Application()\napp.router.add_static('/',root,show_index=False)\n"
        f"web.run_app(app,host='127.0.0.1',port={port},print=None)\n")
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path)
    p.add_argument('--pilots',type=Path,nargs='*',default=[])
    p.add_argument('--products',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=18134)
    a=p.parse_args()
    if not a.campaign and not a.pilots and not a.products:p.error('Supply at least one campaign, pilot or product trial')
    if not 1024<=a.port<=65535:p.error('Invalid unprivileged local port')
    x=export(a.campaign,a.pilots,a.output,a.port,a.products)
    print({'formal':x['formal_summary'],'development_attempts':x['development_attempts'],'index':str(a.output/'index.html')})
