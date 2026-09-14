"""Create a compact plotting/review bundle, retaining hashes of excluded raw data."""
from pathlib import Path
import hashlib
import json
import time
import zipfile

HERE=Path(__file__).resolve().parent


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    assert json.loads((HERE/'final_audit.json').read_text())['status']=='complete'
    assert json.loads((HERE/'resume_status.json').read_text())['status']=='complete'
    metadata=['final_audit.json','analysis_status.json','experiment_manifest.json',
              'new_query.json','pause.json','resume_status.json','finish_status.json']
    selected=set(HERE.glob('*.csv'))|set(HERE.glob('*.md'))|set(HERE.glob('*.py'))
    selected|=set(HERE.glob('*.png'))|set(HERE.glob('*.pdf'))
    selected|={HERE/name for name in metadata}
    # The initial manifest is immutable provenance; final script hashes record
    # the later analysis fixes and guided-only request flag correction.
    raw=set(HERE.glob('external_*.json'))|set(HERE.glob('*/result.json'))
    raw|=set(HERE.glob('native/*/result.json'))|set(HERE.glob('query_*/rep_*/result.json'))
    raw|={HERE/'infllm_chunk_query.json'}
    manifest=dict(created_epoch=time.time(),
                  files={str(p.relative_to(HERE)):sha256(p) for p in sorted(selected)},
                  excluded_raw_results={str(p.relative_to(HERE)):sha256(p) for p in sorted(raw)},
                  note='CSV contains all repetitions. Large raw JSON, Nsight reports/SQLite, and interrupted OSWorld files stay at source paths, outside this plotting bundle. Initial experiment manifest is preserved; final script hashes are here.')
    index=HERE/'artifact_manifest.json';index.write_text(json.dumps(manifest,indent=2)+'\n')
    selected.add(index)
    out=HERE/'throughput_breakdown_data.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(selected):z.write(p,p.relative_to(HERE))
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,files=len(selected))))


if __name__=='__main__':main()
