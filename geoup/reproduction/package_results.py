import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT.parent/'geoup_results_delivery_20261118'
PART_BYTES = 512 * 1024 * 1024


class Parts:
    def __init__(self, name):
        self.name = name
        self.index = 0
        self.size = 0
        self.handle = None

    def write(self, data):
        remaining = memoryview(data)
        while remaining:
            if self.handle is None:
                self.handle = (DESTINATION/f'{self.name}.tar.gz.part{self.index:03d}').open('wb')
            count = min(len(remaining), PART_BYTES-self.size)
            self.handle.write(remaining[:count])
            remaining = remaining[count:]
            self.size += count
            if self.size == PART_BYTES:
                self.handle.close()
                self.handle = None
                self.index += 1
                self.size = 0
        return len(data)

    def close(self):
        if self.handle is not None:
            self.handle.close()


def main():
    DESTINATION.mkdir(exist_ok=True)
    status = DESTINATION/'status.json'
    source_files = []
    groups = [('reproduction', [ROOT/'local', ROOT/'LICENSE', ROOT/'README.md']),
              *[(directory.name, [directory]) for directory in sorted((ROOT/'outputs').iterdir()) if directory.is_dir()]]
    for name, paths in groups:
        status.write_text(json.dumps(dict(state='packaging', group=name)))
        sink = Parts(name)
        try:
            with tarfile.open(fileobj=sink, mode='w|gz', compresslevel=1) as archive:
                for path in paths:
                    candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
                    for candidate in candidates:
                        if not candidate.is_file() or '__pycache__' in candidate.parts:
                            continue
                        digest = hashlib.file_digest(candidate.open('rb'), 'sha256').hexdigest()
                        relative = str(candidate.relative_to(ROOT))
                        source_files.append(dict(path=relative, bytes=candidate.stat().st_size, sha256=digest, group=name))
                        archive.add(candidate, arcname=relative, recursive=False)
        finally:
            sink.close()
        print('PACKED', name, flush=True)
    (DESTINATION/'source_manifest.json').write_text(json.dumps(source_files, indent=2))
    checksums = []
    for path in sorted(DESTINATION.glob('*.part*')):
        with path.open('rb') as handle:
            checksums.append(f'{hashlib.file_digest(handle, "sha256").hexdigest()}  {path.name}\n')
    (DESTINATION/'SHA256SUMS').write_text(''.join(checksums))
    status.write_text(json.dumps(dict(state='complete', source_files=len(source_files), completed=time.time())))


if __name__ == '__main__':
    main()
