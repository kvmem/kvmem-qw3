"""Streaming reader for full-visible attention traces; no task alignment guesses."""
from pathlib import Path
import json
import numpy as np


def expected_decode_queries(completion_tokens, max_tokens):
    """Dense no-client-stop loop: first output comes from prefill logits.

    Natural EOS needs a forward on the last visible output, so queries ==
    visible outputs. At the output cap the last output is not forwarded,
    so queries == visible outputs - 1. See native generate_internal's loop.
    """
    if max_tokens<=0 or not 0<=completion_tokens<=max_tokens:
        raise ValueError('invalid completion budget accounting')
    return completion_tokens-int(completion_tokens==max_tokens)


def read_request(path, start, stop):
    """Read an explicit request interval, including layer-major prefill rows."""
    grouped={};block_tokens=None;rows=0
    with Path(path).open('rb') as f:
        f.seek(start)
        while f.tell()<stop:
            line=f.readline()
            if not line:break
            r=json.loads(line)
            if r.get('kind')!='attention_mass' or r.get('mode')!='global':continue
            pos=int(r['position']);layer=int(r['layer']);bt=int(r['block_tokens'])
            if 'seq_len' in r and int(r['seq_len'])!=pos+1:
                raise ValueError('trace causal length differs from query position')
            if block_tokens is not None and bt!=block_tokens:raise ValueError('block size changed')
            block_tokens=bt;layers=grouped.setdefault(pos,{})
            if layer in layers:raise ValueError('duplicate layer/position in request interval')
            ids=np.asarray(r['block_ids']);values=np.asarray(r['mass'],dtype=float)
            if not np.isfinite(values).all() or (values < -1e-6).any():raise ValueError('invalid mass')
            keep=ids>=0;v=np.zeros(int(ids[keep].max())+1)
            if (ids[keep]>(pos//bt)).any():raise ValueError('future block in causal trace')
            v[ids[keep]]=np.maximum(values[keep],0)
            if v.sum()<=0:raise ValueError('zero mass')
            layers[layer]=v/v.sum();rows+=1
    if not grouped:raise ValueError('empty request trace')
    positions=sorted(grouped);expected=set(grouped[positions[0]])
    width=max(len(v) for layers in grouped.values() for v in layers.values())
    mass=np.zeros((len(positions),width),dtype=np.float32)
    for i,pos in enumerate(positions):
        if set(grouped[pos])!=expected:raise ValueError('inconsistent layer set')
        for v in grouped[pos].values():mass[i,:len(v)]+=v/len(expected)
    return dict(mass=mass,positions=np.asarray(positions,dtype=np.int64),layers=sorted(expected),
                block_tokens=block_tokens,byte_start=start,byte_end=stop,rows=rows)


def iter_segments(path, start=0, stop=None):
    """Yield dense layer-mean matrices, splitting on sample resets.

    Byte intervals permit unambiguous association of new requests. Legacy
    association is separately verified against prompt and completion counts.
    """
    samples, positions, layers, sample_id = [], [], {}, None
    first_offset = start
    current_position = None
    block_tokens = None
    layer_set = None
    row_count = 0

    def finish_sample():
        nonlocal layer_set
        if not layers:
            return
        seen = tuple(sorted(layers))
        if layer_set is None:
            layer_set = seen
        if seen != layer_set:
            raise ValueError(f'inconsistent layers at position {current_position}')
        width = max(len(v) for v in layers.values())
        avg = np.zeros(width, dtype=np.float64)
        for v in layers.values():
            avg[:len(v)] += v / len(layers)
        samples.append(avg.astype(np.float32))
        positions.append(current_position)

    def result(end_offset):
        width = max(map(len, samples))
        mass = np.zeros((len(samples), width), dtype=np.float32)
        for i, v in enumerate(samples):
            mass[i, :len(v)] = v
        return dict(mass=mass, positions=np.array(positions, dtype=np.int64),
                    layers=list(layer_set), block_tokens=block_tokens,
                    byte_start=first_offset, byte_end=end_offset, rows=row_count)

    with Path(path).open('rb') as f:
        f.seek(start)
        while stop is None or f.tell() < stop:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('kind') != 'attention_mass' or row.get('mode') != 'global':
                continue
            sid = int(row['sample'])
            if sample_id is not None and sid != sample_id:
                finish_sample()
                layers.clear()
                if sid < sample_id:
                    yield result(offset)
                    samples, positions, layer_set, row_count = [], [], None, 0
                    first_offset = offset
            sample_id = sid
            current_position = int(row['position'])
            block_tokens = int(row['block_tokens'])
            ids = np.asarray(row['block_ids'], dtype=np.int64)
            values = np.asarray(row['mass'], dtype=np.float64)
            if not np.isfinite(values).all() or (values < -1e-6).any():
                raise ValueError('invalid attention mass')
            if int(row['layer']) in layers:
                raise ValueError('duplicate layer in token sample')
            keep = ids >= 0
            v = np.zeros(int(ids[keep].max()) + 1, dtype=np.float64)
            v[ids[keep]] = np.maximum(values[keep], 0)
            if v.sum() <= 0:
                raise ValueError('zero attention mass')
            v /= v.sum()
            layers[int(row['layer'])] = v
            row_count += 1
        finish_sample()
        if samples:
            yield result(f.tell())


def save_segment(path, seg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mass=seg['mass'], positions=seg['positions'])
    return {k: v for k, v in seg.items() if k not in ('mass', 'positions')} | {
        'path': str(path.resolve()), 'samples': len(seg['positions']),
        'first_position': int(seg['positions'][0]),
        'last_position': int(seg['positions'][-1]),
        'blocks': int(seg['mass'].shape[1]),
    }
