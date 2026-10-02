"""Build a self-contained, credential-free comparison of product-trial videos."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from software_product_report import summarize


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>本地 Python 配置 · 实测回放</title>
<style>
body{font:16px system-ui,sans-serif;background:#f5f6fa;color:#152134;margin:24px auto;padding:0 20px;max-width:1600px}
h1{font-size:26px}p{line-height:1.6}.note{color:#59677b}button,select{font:inherit;padding:9px 13px;border:1px solid #cad2df;border-radius:7px;background:white;margin:5px 8px 5px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.panel{background:white;border:1px solid #dfe4ed;border-radius:12px;padding:16px;min-width:0}
video{width:100%;aspect-ratio:1.6;background:#152134;border-radius:7px}.facts{white-space:pre-wrap;line-height:1.8}.error{color:#982a26}small{word-break:break-all;color:#617189}
@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style>
<h1>本地 Python 环境配置：失败、修复与复测</h1>
<p>真实 QW3 + Codex · Windows 虚拟机 · 受控软件工具 · KVMem 关闭 · 顾问关闭</p>
<p class="note">这里是连续 VNC 录像，包含准备等待。可按任务开始时间对齐播放；两侧走过的步骤不同，不能将画面进度或总时长直接视为模型加速比。右侧结果由新终端独立验收。</p>
<button id="play">从任务开始同时播放</button><button id="pause">暂停两侧</button>
<span class="error" id="error" role="status"></span>
<div class="grid">
 <section class="panel"><select id="left" aria-label="左侧试次"></select><div class="facts" id="leftFacts"></div><video id="leftVideo" controls preload="metadata" muted></video><p id="leftStages"></p><small id="leftIdentity"></small></section>
 <section class="panel"><select id="right" aria-label="右侧试次"></select><div class="facts" id="rightFacts"></div><video id="rightVideo" controls preload="metadata" muted></video><p id="rightStages"></p><small id="rightIdentity"></small></section>
</div>
<p class="note">这些是开发试次，全部失败保留。一次通过不代表稳定成功率，也不证明通用视觉或复杂规划能力。显存按宿主 GPU 设备采样，内存只统计引擎进程；尚未包含完整应用和虚拟机内存。</p>
<script>
const el=id=>document.getElementById(id);let trials=[];let generation=0;
const seconds=x=>x==null?'未取得':x.toFixed(2)+' 秒';
function chosen(side){return trials[Number(el(side).value)]}
function render(side){const t=chosen(side),v=el(side+'Video');v.pause();v.src=t.video;
 el(side+'Facts').textContent=(t.passed?'已通过独立验收':'未通过')+' · '+t.attempt+'\\n任务耗时：'+seconds(t.task_seconds)+' · 模型请求 '+t.inference.count+' 次 · 工具调用 '+t.tool_calls+' 次\\n工具错误 '+t.tool_errors+' 次，其中等待参数类型错误 '+t.wait_argument_type_errors+' 次';
 el(side+'Stages').textContent=t.stages.map(s=>s.name+'：'+s.status+'（'+seconds(s.seconds)+'）').join(' → ');
 el(side+'Identity').textContent='客户端 SHA-256：'+t.bundle_sha256+' · 原始结果 SHA-256：'+t.result_sha256;
}
function pause(){generation++;for(const side of ['left','right'])el(side+'Video').pause()}
async function seek(v,time){if(v.readyState<1)await new Promise((ok,fail)=>{v.addEventListener('loadedmetadata',ok,{once:true});v.addEventListener('error',fail,{once:true})});
 const target=Math.min(time,Math.max(0,v.duration-.1));if(Math.abs(v.currentTime-target)<.02)return;
 await new Promise(ok=>{v.addEventListener('seeked',ok,{once:true});v.currentTime=target})}
el('pause').onclick=pause;
el('play').onclick=async()=>{pause();const turn=generation;el('error').textContent='';try{
 await Promise.all(['left','right'].map(side=>seek(el(side+'Video'),chosen(side).task_start_video_seconds)));
 if(turn!==generation)return;await Promise.all(['left','right'].map(side=>el(side+'Video').play()));
 }catch(e){if(turn===generation&&e.name!=='AbortError')el('error').textContent='播放失败：'+e.message}};
fetch('trials.json').then(r=>{if(!r.ok)throw Error('无法读取试次数据');return r.json()}).then(data=>{
 trials=data;for(const side of ['left','right']){for(let i=0;i<trials.length;i++){const o=document.createElement('option');o.value=i;o.textContent=trials[i].attempt+(trials[i].passed?' · 通过':' · 未通过');el(side).append(o)}
 el(side).value=side==='left'?0:trials.length-1;el(side).onchange=()=>{pause();render(side)};render(side)}
}).catch(e=>el('error').textContent=e.message);
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trials = []
    for index, root in enumerate(args.runs):
        result = json.loads((root / "result.json").read_text())
        if result.get("recording", {}).get("complete") is not True:
            raise ValueError("Only finalized continuous recordings may enter the replay")
        trial = summarize(root)
        name = f"trial-{index}.mp4"
        source = root / "video/screen.mp4"
        if hashlib.sha256(source.read_bytes()).hexdigest() != result["recording"]["artifacts"]["screen.mp4"]["sha256"]:
            raise ValueError("Recorded video changed after the trial receipt was finalized")
        shutil.copyfile(source, args.output / name)
        trial.update(video=name, task_start_video_seconds=max(0,
            result["guest_result"]["terminal_state"]["started_at"] - result["recording"]["started_unix_ns"] / 1e9))
        trials.append(trial)
    (args.output / "trials.json").write_text(json.dumps(trials, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "index.html").write_text(HTML, encoding="utf-8")
    # Serve only the exported replay directory, never the broader .runtime tree.
    (args.output / "serve.py").write_text('''from pathlib import Path
from aiohttp import web
root=Path(__file__).resolve().parent
app=web.Application()
app.router.add_static('/',root,show_index=False)
web.run_app(app,host='127.0.0.1',port=18109,print=None)
''', encoding="utf-8")
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
