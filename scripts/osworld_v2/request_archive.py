"""Lossless grouped storage for complete recorded native request bytes.

Compression runs after forwarding in the artifact worker. Each finalized group
is independently decodable; individual records have SHA-256 checksums. The
format affects saved artifacts only, never the client/server protocol.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

class GroupedRequestWriter:
    def __init__(self,directory,group_size=32):
        if group_size<=0:raise ValueError('group_size must be positive')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.binary=shutil.which('zstd')
        if self.binary is None:raise RuntimeError('Grouped request recording requires zstd')
        self.group_size=group_size;self.group=0;self.pending=[];self.index={};self.proc=None;self.file=None
        if ((self.directory/'request_index.json').exists() or list(self.directory.glob('requests_*.zst*')) or
            list(self.directory.glob('request_*.json.gz'))):
            raise FileExistsError('Preserve existing request archives before starting a new writer')

    def _start(self):
        self.path=self.directory/f'requests_{self.group:04d}.zst'
        self.partial=self.path.with_suffix('.zst.partial')
        self.file=self.partial.open('xb')
        try:
            self.proc=subprocess.Popen([self.binary,'-q','--single-thread','-1','--long=29','-c'],
                stdin=subprocess.PIPE,stdout=self.file,stderr=subprocess.PIPE)
        except BaseException:
            self.file.close();self.file=None;raise

    def write(self,name,raw):
        if Path(name).name!=name or name in self.index or any(x['name']==name for x in self.pending):
            raise ValueError('Invalid or duplicate logical request filename')
        if self.proc is None:self._start()
        record={'name':name,'length':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        header=json.dumps(record,separators=(',',':')).encode()
        self.proc.stdin.write(struct.pack('>Q',len(header)))
        self.proc.stdin.write(header);self.proc.stdin.write(raw)
        self.pending.append(record)
        if len(self.pending)>=self.group_size:self._finish()

    def _finish(self):
        if self.proc is None:return
        proc,self.proc=self.proc,None
        try:
            proc.stdin.close()
            code=proc.wait(timeout=30)
            error=proc.stderr.read().decode(errors='replace')
            if code:raise RuntimeError(f'zstd exited {code}: {error}')
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            proc.stderr.close();self.file.close();self.file=None
        self.partial.replace(self.path)
        for record in self.pending:self.index[record['name']]={'archive':self.path.name,**record}
        temporary=self.directory/'request_index.tmp'
        temporary.write_text(json.dumps({'format':'qw3-native-request-groups-v1','records':self.index},indent=2)+'\n')
        temporary.replace(self.directory/'request_index.json')
        self.pending=[];self.group+=1

    def close(self):self._finish()

def _read_exact(stream,length):
    result=stream.read(length)
    if len(result)!=length:raise RuntimeError('Truncated request archive')
    return result

def iter_group(path):
    """Yield verified (logical filename, original bytes) records in one group."""
    proc=subprocess.Popen(['zstd','-d','-q','-c','-M1024MB',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    finished=False
    try:
        while True:
            prefix=proc.stdout.read(8)
            if not prefix:break
            if len(prefix)!=8:raise RuntimeError('Truncated archive record header')
            length=struct.unpack('>Q',prefix)[0]
            if length>4096:raise RuntimeError('Invalid archive record header size')
            record=json.loads(_read_exact(proc.stdout,length))
            raw=_read_exact(proc.stdout,record['length'])
            if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise RuntimeError('Request checksum mismatch')
            yield record['name'],raw
        code=proc.wait(timeout=30);error=proc.stderr.read().decode(errors='replace')
        if code:raise RuntimeError(f'Archive decompression failed: {error}')
        finished=True
    finally:
        if not finished and proc.poll() is None:proc.terminate()
        proc.stdout.close();proc.stderr.close()
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()

def read_request(directory,request_id):
    """Read existing gzip recordings or the grouped lossless format."""
    directory=Path(directory);name=f'request_{request_id:04d}.json.gz'
    legacy=directory/name
    if legacy.exists():return gzip.decompress(legacy.read_bytes())
    index=json.loads((directory/'request_index.json').read_text())
    record=index['records'][name];archive=record['archive']
    if Path(archive).name!=archive:raise ValueError('Invalid request archive path')
    selected=None
    # Drain the group to validate its final frame checksum as well.
    for logical_name,raw in iter_group(directory/archive):
        if logical_name==name:selected=raw
    if selected is None:raise RuntimeError('Indexed request missing from its archive')
    if len(selected)!=record['length'] or hashlib.sha256(selected).hexdigest()!=record['sha256']:
        raise RuntimeError('Request index mismatch')
    return selected
