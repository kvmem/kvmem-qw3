"""Summarize actual CUDA activity, accounting for overlap by interval unions."""
from pathlib import Path
import collections
import json
import sqlite3
from analyze import merge_intervals, write_csv

HERE=Path(__file__).resolve().parent


def inspect(path):
    conn=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    conn.row_factory=sqlite3.Row
    tables={r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    strings={r['id']:r['value'] for r in conn.execute('select * from StringIds')}
    ranges=[]
    for r in conn.execute('select * from NVTX_EVENTS'):
        r=dict(r);label=r.get('text') or strings.get(r.get('textId'),'')
        if label=='measured_decode_with_initial_reset' and r.get('end') is not None:
            ranges.append(r)
    assert len(ranges)==1, (path,len(ranges))
    start,end=ranges[0]['start'],ranges[0]['end'];duration=end-start
    def activities(table):
        if table not in tables:return []
        result=[]
        for r in conn.execute(f'SELECT * FROM {table} WHERE end > ? AND start < ?', (start,end)):
            row=dict(r);row['start']=max(start,row['start']);row['end']=min(end,row['end']);result.append(row)
        return result
    kernels=activities('CUPTI_ACTIVITY_KIND_KERNEL')
    copies=activities('CUPTI_ACTIVITY_KIND_MEMCPY')
    memset=activities('CUPTI_ACTIVITY_KIND_MEMSET')
    runtime=activities('CUPTI_ACTIVITY_KIND_RUNTIME')
    assert kernels and copies, (path,tables)
    def union(rows):return merge_intervals((r['start'],r['end']) for r in rows)
    def aggregate(rows,key):
        grouped=collections.defaultdict(list)
        for r in rows:grouped[key(r)].append(r)
        result=[]
        for label,items in grouped.items():
            spans=[r['end']-r['start'] for r in items]
            result.append(dict(label=label,n=len(items),sum_ms=sum(spans)/1e6,
                union_ms=union(items)/1e6,max_ms=max(spans)/1e6))
        return sorted(result,key=lambda r:r['sum_ms'],reverse=True)
    kinds={}
    if 'ENUM_CUDA_MEMCPY_OPER' in tables:
        kinds={r['id']:r['label'] for r in conn.execute('select * from ENUM_CUDA_MEMCPY_OPER')}
    copy_groups=aggregate(copies,lambda r:kinds.get(r['copyKind'],str(r['copyKind'])))
    for group in copy_groups:
        group['bytes']=sum(r['bytes'] for r in copies if kinds.get(r['copyKind'],str(r['copyKind']))==group['label'])
    data=json.loads(path.with_suffix('.json').read_text())
    measurement=data['measurements'][0];worker=data['worker_native_tid']
    main_tid=ranges[0]['globalTid'] & 0xFFFFFF
    role=lambda r: 'worker' if (r['globalTid'] & 0xFFFFFF)==worker else ('main' if (r['globalTid'] & 0xFFFFFF)==main_tid else 'other')
    result=dict(name=path.stem,prompt_tokens=measurement['prompt_tokens'],mtp=measurement['mtp'],mode=measurement['mode'],
        traced_decode_tps=measurement['decode_tokens_per_s'],traced_decode_s=measurement['decode_s'],
        nvtx_range_s=duration/1e9,worker_native_tid=worker,main_native_tid=main_tid,
        kernel_active_s=union(kernels)/1e9,memcpy_active_s=union(copies)/1e9,
        any_gpu_activity_s=union(kernels+copies+memset)/1e9,
        no_traced_gpu_activity_fraction=1-union(kernels+copies+memset)/duration,
        kernel_active_fraction=union(kernels)/duration,
        python_gc_events=len(data.get('python_gc_events',[])),
        python_gc_host_s=sum(r['host_s'] for r in data.get('python_gc_events',[])),
        top_kernels=aggregate(kernels,lambda r: strings.get(r.get('demangledName',r.get('shortName')),'unknown'))[:50],
        copies=copy_groups,runtime_calls=aggregate(runtime,lambda r:role(r)+': '+strings.get(r['nameId'],'unknown'))[:60],
        tables=sorted(tables),correct_output=measurement['exact_output_vs_prior_fresh_prefill'],
        scope='Nsight-instrumented 32-output diagnostic, range includes initial state reset and seed sampling. Activity unions handle stream overlap. No activity means no recorded kernel/copy/memset, not SM occupancy. Never substitute traced throughput for uninstrumented timing.')
    conn.close();return result


def main():
    results=[inspect(path) for path in sorted(HERE.glob('trace_*.sqlite'))]
    (HERE/'nsight_analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    write_csv('nsight_activity_summary.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results])
    write_csv('nsight_memcpy.csv',[dict(trace=r['name'],**x) for r in results for x in r['copies']])
    write_csv('nsight_top_kernels.csv',[dict(trace=r['name'],**x) for r in results for x in r['top_kernels']])
    write_csv('nsight_runtime_calls.csv',[dict(trace=r['name'],**x) for r in results for x in r['runtime_calls']])
    print(json.dumps([{k:v for k,v in r.items() if k not in ('top_kernels','runtime_calls','tables')} for r in results],indent=2))


if __name__=='__main__':main()
