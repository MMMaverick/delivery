import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='local/172055_occ_fullscene.yaml')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/args.config).read_text())
    output = Path(config['output'])
    deadline = time.monotonic()+7200
    while time.monotonic()<deadline:
        try:
            status = json.loads((output/'status.json').read_text())
        except OSError as error:
            if error.errno not in [2,11,116]:
                raise
            time.sleep(5)
            continue
        if status['state'] == 'complete':
            break
        if status['state'] not in ['running']:
            raise RuntimeError(status)
        time.sleep(5)
    else:
        raise TimeoutError('Inference not complete after two hours')
    environment = dict(os.environ)
    environment['LD_LIBRARY_PATH'] = '/mnt/workspace/Box_3D/runtime/envs/mono-detect-original-2dbox-full-gpu/lib:/usr/local/PPU_SDK/targets/x86_64-linux/lib:'+environment.get('LD_LIBRARY_PATH','')
    environment.update(MPLCONFIGDIR='/tmp/geoup-mpl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for script in ['export_occ_cloud.py','render_video.py','render_occ_overview.py','render_pointcloud_video.py']:
        print(f'Running {script}',flush=True)
        subprocess.run([sys.executable,str(ROOT/'local'/script),'--config',args.config],
                       cwd=ROOT,env=environment,check=True)
    rows = json.loads((output/'results.json').read_text())['frames']
    manifest = json.loads((Path(config['input_run'])/'manifest.json').read_text())
    if config['frames'] == 'all' and len(rows)!=manifest['frames']:
        raise ValueError('Full scene frame count mismatch')
    for row in rows:
        frame = row['frame']
        with np.load(output/f'{frame:06d}_occ.npz') as data:
            assert len(data['labels'])==row['occupied_voxels']
            assert np.array_equal(data['grid'][tuple(data['locations'].T)],data['labels'])
        assert (output/f'{frame:06d}_occ.ply').is_file()
    metrics = dict(frames=len(rows),cameras=len(manifest['videos']),
                   sampled_duration_seconds=rows[-1]['timestamp_seconds']-rows[0]['timestamp_seconds'],
                   first_inference_seconds=rows[0]['inference_seconds'],
                   steady_mean_inference_seconds=float(np.mean([row['inference_seconds'] for row in rows[1:]])),
                   occupied_voxels_min=min(row['occupied_voxels'] for row in rows),
                   occupied_voxels_max=max(row['occupied_voxels'] for row in rows),
                   full_view=config.get('image_fit')=='letterbox',all_grids_validated=True,
                   videos_decoded=True,ply_files=len(rows))
    (output/'final_validation.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == '__main__':
    main()
