import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='local/172055_validated.yaml')
    arguments = parser.parse_args()
    config_path = ROOT/arguments.config
    config = yaml.safe_load(config_path.read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment['LD_LIBRARY_PATH'] = '/mnt/workspace/Box_3D/runtime/envs/mono-detect-original-2dbox-full-gpu/lib:/usr/local/PPU_SDK/targets/x86_64-linux/lib:'+environment.get('LD_LIBRARY_PATH','')
    environment.update(PYTHONFAULTHANDLER='1', MPLCONFIGDIR='/tmp/geoup-mpl',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (output/'inference.log').open('w') as stream:
        result = subprocess.run([sys.executable,'-u',str(ROOT/'local/infer_local.py'),'--config',str(config_path)],
            cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        status_path = output/'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update(state='failed', returncode=result.returncode, log=str(output/'inference.log'))
        status_path.write_text(json.dumps(status,indent=2))
        raise SystemExit(f'Inference failed ({result.returncode}); see {output/"inference.log"}')
    print((output/'results.json').read_text())


if __name__ == '__main__':
    main()
