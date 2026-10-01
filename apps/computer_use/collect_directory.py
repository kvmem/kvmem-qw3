"""Read-only, one-level task-folder export after the desktop has stopped.

Names are collected from the actual folder. No expected answer list is supplied
and subdirectories are listed without traversing them. File bytes use opaque
export names; original names and types remain in the directory receipt.
"""
import argparse,hashlib,json,os,stat,time
from pathlib import Path
from collect_artifacts import collect,identity,default_roots

MAX_FILES=64
MAX_TOTAL_BYTES=512*1024*1024


def is_link(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&0x400)


def listing(path):
    result=[]
    for item in sorted(path.iterdir(),key=lambda p:p.name):
        info=item.lstat()
        kind='reparse' if is_link(info) else ('file' if stat.S_ISREG(info.st_mode) else ('directory' if stat.S_ISDIR(info.st_mode) else 'special'))
        result.append((item.name,kind,identity(info)))
    return result


def collect_directory(source,output,*,allowed_roots=None):
    source,output=Path(source),Path(output)
    roots=[Path(p).resolve() for p in (default_roots() if allowed_roots is None else allowed_roots)]
    if not source.is_absolute() or not any(source.resolve().is_relative_to(root) and source.resolve()!=root for root in roots):
        raise ValueError('Folder must be an absolute task path beneath an allowed artifact root')
    if output.resolve().is_relative_to(source.resolve()):raise ValueError('Output must be outside the collected directory')
    output.mkdir(parents=True,exist_ok=False)
    report={'kind':'native_directory_artifact_collection','source':str(source),'complete':False,'infrastructure_ok':False,
            'read_only_sources':True,'recursive':False,'entries':[],'directory_status':'unknown'}
    started=time.monotonic()
    try:
        try:before=source.lstat()
        except FileNotFoundError:
            report.update(directory_status='missing',infrastructure_ok=True)
        else:
            if is_link(before):raise ValueError('Folder is a reparse point; no files were followed')
            if not stat.S_ISDIR(before.st_mode):
                report.update(directory_status='not_directory',infrastructure_ok=True)
            else:
                report['directory_status']='present'
                initial=listing(source)
                if len(initial)>MAX_FILES:raise ValueError('Folder entry limit exceeded')
                if any(kind not in {'file','directory'} for _,kind,_ in initial):raise ValueError('Unsupported or reparse folder member')
                files=[]
                for name,kind,info in initial:
                    member={'name':name,'type':kind}
                    if kind=='file':
                        member['export_name']=f'item-{len(files):03d}.bin'
                        files.append({'name':member['export_name'],'source':str(source/name)})
                    report['entries'].append(member)
                if sum(info[2] for _,kind,info in initial if kind=='file')>MAX_TOTAL_BYTES:raise ValueError('Total folder byte limit exceeded')
                if files:
                    receipt=collect({'files':files},output/'files',allowed_roots=roots)
                    report['files_receipt_sha256']=hashlib.sha256((output/'files/receipt.json').read_bytes()).hexdigest()
                    if not receipt['infrastructure_ok'] or any(f['status']!='copied' for f in receipt['files']):raise ValueError('Folder files changed or could not be collected')
                if identity(source.lstat())!=identity(before) or listing(source)!=initial:raise ValueError('Folder changed during collection')
                report['infrastructure_ok']=True
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        report.update(complete=True,elapsed_seconds=time.monotonic()-started)
        with (output/'directory-receipt.tmp').open('x',encoding='utf-8') as handle:
            json.dump(report,handle,indent=2,ensure_ascii=False);handle.flush();os.fsync(handle.fileno())
        (output/'directory-receipt.tmp').rename(output/'directory-receipt.json')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=collect_directory(args.source,args.output)
    print(json.dumps({key:report[key] for key in ['complete','infrastructure_ok','directory_status']}))
    raise SystemExit(0 if report['infrastructure_ok'] else 2)
