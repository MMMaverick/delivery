import io
import json
from pathlib import Path
import tarfile
import time

ROOT = Path('/mnt/workspace/Data/nuScenes_mini')


class GrowingFile(io.RawIOBase):
    def __init__(self, path):
        self.stream = path.open('rb')

    def read(self, size=-1):
        chunks = []
        remaining = size
        deadline = time.monotonic()+180
        while remaining > 0:
            chunk = self.stream.read(remaining)
            if chunk:
                chunks.append(chunk)
                remaining -= len(chunk)
                deadline = time.monotonic()+180
            elif time.monotonic() > deadline:
                raise TimeoutError('Download stopped before sample extraction finished')
            else:
                time.sleep(1)
        return b''.join(chunks)


def main():
    counts = {'metadata':0,'images':0}
    source = ROOT/'v1.0-mini.tgz'
    with tarfile.open(fileobj=GrowingFile(source), mode='r|gz') as archive:
        for member in archive:
            path = Path(member.name)
            if path.parts[0] == 'sweeps':
                break
            wanted = path.parts[0] == 'v1.0-mini' or (
                path.parts[0] == 'samples' and len(path.parts)>1 and path.parts[1].startswith('CAM_'))
            if not wanted or not member.isfile():
                continue
            destination = ROOT/path
            if not destination.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Unsafe archive path')
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
            counts['metadata' if path.parts[0]=='v1.0-mini' else 'images'] += 1
            if counts['images']%200==0:
                print(counts,flush=True)
    (ROOT/'extraction.json').write_text(json.dumps(dict(source_url='https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz',**counts),indent=2))
    print('Camera samples and metadata extracted',counts,flush=True)


if __name__ == '__main__':
    main()
