"""Actual transfer traffic and overlap-safe activity breakdown of new queries."""
from pathlib import Path
import collections
import json
import sqlite3
from analyze import csv_out

HERE=Path(__file__).resolve().parent

def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)

def inspect(path):
    conn=sqlite3.connect(f'file:{path}?mode=ro',uri=True);conn.row_factory=sqlite3.Row
    tables={r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    strings={r['id']:r['value'] for r in conn.execute('select * from StringIds')}
    ranges=[]
    for r in conn.execute('select * from NVTX_EVENTS'):
        r=dict(r);name=r.get('text') or strings.get(r.get('textId'),'')
        if name=='measured_query' and r.get('end') is not None:ranges.append(r)
    assert len(ranges)==1,(path,len(ranges))
    start,end=ranges[0]['start'],ranges[0]['end'];duration=end-start
    def activities(table):
        if table not in tables:return []
        result=[]
        for r in conn.execute(f'SELECT * FROM {table} WHERE end > ? AND start < ?',(start,end)):
            x=dict(r);x['crosses_range_boundary']=x['start']<start or x['end']>end
            x['start']=max(x['start'],start);x['end']=min(x['end'],end);result.append(x)
        return result
    kernels=activities('CUPTI_ACTIVITY_KIND_KERNEL');copies=activities('CUPTI_ACTIVITY_KIND_MEMCPY')
    memset=activities('CUPTI_ACTIVITY_KIND_MEMSET');runtime=activities('CUPTI_ACTIVITY_KIND_RUNTIME')
    assert kernels,(path,tables)
    total=lambda rows:union((r['start'],r['end']) for r in rows)
    compute=total(kernels+memset);copy_time=total(copies);any_active=total(kernels+copies+memset)
    overlap=compute+copy_time-any_active
    copy_names={r['id']:r['label'] for r in conn.execute('select * from ENUM_CUDA_MEMCPY_OPER')} if 'ENUM_CUDA_MEMCPY_OPER' in tables else {}
    def aggregate(rows,key):
        groups=collections.defaultdict(list)
        for r in rows:groups[key(r)].append(r)
        output=[]
        for name,items in groups.items():
            output.append(dict(label=name,count=len(items),sum_ms=sum(r['end']-r['start'] for r in items)/1e6,
                union_ms=total(items)/1e6,bytes=sum(r.get('bytes',0) for r in items)))
        return sorted(output,key=lambda x:x['sum_ms'],reverse=True)
    suffix=path.stem.removeprefix('trace_query_');method,mtp=suffix.rsplit('_mtp',1)
    if method in ['qw3','kvmem']:
        data=json.loads((path.with_suffix('')/'result.json').read_text())
        measured=[r for r in data['results'] if r['instrumented']]
        assert len(measured)==1,(path,len(measured))
        r=measured[0]
    else:
        data=json.loads(path.with_suffix('.json').read_text());r=data['measurements'][0]
    result=dict(method=method,mtp=int(mtp),history_tokens=131072,outputs=128,trace=path.stem,
        nvtx_request_s=duration/1e9,instrumented_effective_tps=r['effective_tokens_per_s'],
        compute_only_s=(compute-overlap)/1e9,copy_only_s=(copy_time-overlap)/1e9,
        compute_copy_overlap_s=overlap/1e9,no_recorded_gpu_activity_s=(duration-any_active)/1e9,
        gpu_kernel_union_s=total(kernels)/1e9,gpu_any_active_fraction=any_active/duration,
        kernel_count=len(kernels),cuda_runtime_calls=len(runtime),
        boundary_crossing_copy_count=sum(r['crosses_range_boundary'] for r in copies),
        copy_groups=aggregate(copies,lambda x:copy_names.get(x['copyKind'],str(x['copyKind']))),
        kernels=aggregate(kernels,lambda x:strings.get(x.get('demangledName',x.get('shortName')),'unknown')),
        runtime=aggregate(runtime,lambda x:strings.get(x['nameId'],'unknown')),
        scope='Instrumented full new-query request. Exclusive activity categories sum to range duration. GPU kernels include model and retrieval. No recorded activity is not proven idle SM time or exclusively CPU execution. Never use instrumented speed as formal throughput.')
    assert abs(sum(result[k] for k in ['compute_only_s','copy_only_s','compute_copy_overlap_s','no_recorded_gpu_activity_s'])-result['nvtx_request_s'])<1e-7
    conn.close();return result

def main():
    results=[inspect(p) for p in sorted(HERE.glob('trace_query_*.sqlite'))]
    (HERE/'trace_analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    csv_out('trace_activity.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results])
    for label in ['copy_groups','kernels','runtime']:
        csv_out('trace_'+label+'.csv',[dict(method=r['method'],mtp=r['mtp'],**x) for r in results for x in r[label]])
    print(json.dumps([dict(method=r['method'],mtp=r['mtp'],request_s=r['nvtx_request_s'],gpu_active=r['gpu_any_active_fraction'],copies=r['copy_groups']) for r in results]))

if __name__=='__main__':main()
