"""Decode an existing native-pilot movie without accessing any device or model."""
import hashlib
import json
from pathlib import Path


def decode(path):
    import av
    path = Path(path).resolve()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    frames = 0
    with av.open(str(path)) as movie:
        for frame in movie.decode(video=0):
            if frames == 0:width,height = frame.width,frame.height
            if (frame.width,frame.height) != (width,height):raise ValueError('Video dimensions changed')
            frames += 1
    if frames == 0:raise ValueError('No decoded video frames')
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:raise ValueError('Video changed during decoding')
    return dict(path=str(path),sha256=before,frames=frames,width=width,height=height)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('path',type=Path)
    print(json.dumps(decode(parser.parse_args().path)))
