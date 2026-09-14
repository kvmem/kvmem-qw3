"""Publication-friendly figures; plot only completed unprofiled measurements."""
from pathlib import Path
import argparse
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
METHODS=['qw3','kvmem','infllm','infinigen']
LABELS={'qw3':'qw3','kvmem':'KVMem','infllm':'InfLLM (Qwen adapter)',
        'infinigen':'InfiniGen (Qwen adapter)'}
COLORS={'qw3':'#365f8b','kvmem':'#bc633a','infllm':'#46816a','infinigen':'#8365a3'}
LENGTHS=[8192,16384,32768,65536,131072]

def export(fig,name):
    fig.savefig(HERE/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(HERE/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steady-only',action='store_true');args=parser.parse_args()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    rows=list(csv.DictReader((HERE/'summary.csv').open()))
    measurements=list(csv.DictReader((HERE/'measurements.csv').open()))
    def row(scenario,method,n,mtp):
        found=[r for r in rows if r['scenario']==scenario and r['method']==method
               and int(r['history_tokens'])==n and int(r['mtp'])==mtp]
        assert len(found)==1,(scenario,method,n,mtp,len(found))
        assert int(found[0]['n'])==3,found[0]
        return found[0]
    for scenario,field,title in [
        ('steady_decode','decode_tokens_per_s','Steady decode: 127 tokens after first token'),
        ('new_query','effective_tokens_per_s','New query (provided question): 128 outputs / full request time')]:
        if args.steady_only and scenario!='steady_decode':continue
        fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,mtp in zip(axs,[0,4]):
            for method in METHODS:
                data=[row(scenario,method,n,mtp) for n in LENGTHS]
                med=np.array([float(r['median_'+field]) for r in data])
                lo=np.array([float(r['min_'+field]) for r in data])
                hi=np.array([float(r['max_'+field]) for r in data])
                ax.errorbar(range(5),med,yerr=[med-lo,hi-med],label=LABELS[method],
                            color=COLORS[method],marker='o',capsize=3,lw=2)
                if scenario=='new_query' and method=='kvmem':
                    selected=[i for i,n in enumerate(LENGTHS) if any(r['scenario']=='new_query' and r['method']=='kvmem'
                        and int(r['history_tokens'])==n and int(r['mtp'])==mtp and int(r['query_replay_log_count'] or 0)>0 for r in measurements)]
                    ax.scatter(selected,med[selected],marker='D',s=70,color=COLORS[method],zorder=5,label='KVMem selection + replay')
            ax.set(title=f'MTP = {mtp}',xticks=range(5),xticklabels=['8K','16K','32K','64K','128K'],
                   xlabel='History tokens (K = 1024)')
            ax.grid(alpha=.2)
        top=max(float(row(scenario,m,n,t)['max_'+field]) for m in METHODS for n in LENGTHS for t in [0,4])
        axs[0].set_ylim(0,top*1.08)
        axs[0].set_ylabel('Useful output tokens / s')
        axs[1].legend(fontsize=9)
        fig.suptitle(title)
        fig.text(.5,-.02,'Same Q8 weights / FP8 KV; different backends and active windows. Median of 3; bars show min–max.',ha='center',fontsize=9)
        fig.tight_layout();export(fig,scenario+'_throughput')
    if args.steady_only:return
    fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,mtp in zip(axs,[0,4]):
        for method in METHODS:
            y=[float(row('new_query',method,n,mtp)['median_query_ttft_s']) for n in LENGTHS]
            ax.plot(range(5),y,'o-',label=LABELS[method],color=COLORS[method])
        ax.set(title=f'MTP = {mtp}',xticks=range(5),xticklabels=['8K','16K','32K','64K','128K'],
               xlabel='History tokens (new query adds 40)');ax.grid(alpha=.2)
    top=max(float(row('new_query',m,n,t)['median_query_ttft_s']) for m in METHODS for n in LENGTHS for t in [0,4])
    axs[0].set_ylim(0,top*1.08)
    axs[0].set_ylabel('New-query TTFT (s)');axs[1].legend(fontsize=9)
    fig.suptitle('Provided-question TTFT, including actual working-set refreshes')
    fig.tight_layout();export(fig,'new_query_ttft')
    trace=HERE/'trace_analysis.json'
    if trace.exists():
        data=json.loads(trace.read_text())
        fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
        fields=['compute_only_s','copy_only_s','compute_copy_overlap_s','no_recorded_gpu_activity_s']
        labels=['Kernel / memset only','Copy only','Compute–copy overlap','No recorded GPU activity']
        for ax,mtp in zip(axs,[0,4]):
            bottom=np.zeros(4)
            for field,label in zip(fields,labels):
                vals=np.array([next(r[field] for r in data if r['method']==m and r['mtp']==mtp) for m in METHODS])
                ax.bar(range(4),vals,bottom=bottom,label=label);bottom+=vals
            ax.set(title=f'MTP = {mtp}',xticks=range(4),xticklabels=['qw3','KVMem','InfLLM','InfiniGen'])
            ax.grid(axis='y',alpha=.2)
        axs[0].set_ylabel('Instrumented request time (s)');axs[1].legend(fontsize=9)
        fig.suptitle('128K + new query: overlap-safe GPU activity breakdown')
        fig.text(.5,-.03,'Profiler timings are diagnostic. No recorded GPU activity is not a direct measure of CPU execution.',ha='center',fontsize=9)
        fig.tight_layout();export(fig,'new_query_gpu_activity')

if __name__=='__main__':main()
