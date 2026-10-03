"""Lossless content-addressed request archive for completed private traces."""
import gzip
import hashlib
import json
from pathlib import Path


def digest(data): return hashlib.sha256(data).hexdigest()


def encode(value, blobs):
    if isinstance(value,str):
        if len(value)<1024:return {'s':value}
        data=value.encode();key=digest(data);path=blobs/(key+'.gz')
        if not path.exists():
            with path.open('xb') as file:file.write(gzip.compress(data,compresslevel=1))
        return {'blob':key}
    if isinstance(value,dict):return {'d':[[key,encode(member,blobs)] for key,member in value.items()]}
    if isinstance(value,list):return {'l':[encode(member,blobs) for member in value]}
    return {'p':value}


def decode(value,blobs):
    if set(value)=={'s'}:return value['s']
    if set(value)=={'p'}:return value['p']
    if set(value)=={'l'}:return [decode(v,blobs) for v in value['l']]
    if set(value)=={'d'}:return {key:decode(v,blobs) for key,v in value['d']}
    if set(value)=={'blob'}:
        key=value['blob']
        if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Bad archive digest')
        data=gzip.decompress((blobs/(key+'.gz')).read_bytes())
        if digest(data)!=key:raise ValueError('Corrupt trace blob')
        return data.decode()
    raise ValueError('Invalid archive node')


def read_request(trace):
    trace=Path(trace)
    if (trace/'request.json').exists():return (trace/'request.json').read_bytes()
    meta=json.loads((trace/'request.archive.json').read_text())
    data=json.dumps(decode(meta['tree'],trace.parent/'blobs'),ensure_ascii=False).encode()
    if digest(data)!=meta['sha256'] or len(data)!=meta['bytes']:raise ValueError('Archive request identity mismatch')
    return data


def archive_completed(inference):
    """Only metrics-complete requests: never touch an in-flight writer."""
    inference=Path(inference)
    if not inference.exists():return 0
    blobs=inference/'blobs';blobs.mkdir(exist_ok=True);count=0
    for trace in inference.iterdir():
        source=trace/'request.json'
        if not trace.is_dir() or not (trace/'metrics.json').is_file() or not source.is_file():continue
        original=source.read_bytes();tree=encode(json.loads(original),blobs)
        meta={'version':1,'bytes':len(original),'sha256':digest(original),'tree':tree}
        target=trace/'request.archive.json'
        if target.exists():raise ValueError('Refusing to replace existing request archive')
        temp=trace/'request.archive.tmp';temp.write_text(json.dumps(meta,ensure_ascii=False));temp.replace(target)
        restored=json.dumps(decode(tree,blobs),ensure_ascii=False).encode()
        if restored!=original:raise ValueError('Lossless trace roundtrip failed; original retained')
        source.unlink();count+=1
    return count
