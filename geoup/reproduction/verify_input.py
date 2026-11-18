import json
from pathlib import Path
import pickle
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local.infer_local import prepare_frame


def main():
    config = yaml.safe_load((ROOT/'local/172055.yaml').read_text())
    config['device'] = 'cpu'
    run = Path(config['input_run'])
    manifest = json.loads((run/'manifest.json').read_text())
    infos = pickle.loads((run/'result_interpolation_optimized_first101.pkl').read_bytes())['infos']
    annotations = sorted((Path(manifest['source'])/'format_output/annotations/NV').glob('*.json'))
    first = json.loads(annotations[config['frames'][0]].read_text())
    start = int(first['cams'][manifest['videos'][0]['camera']]['timestamp'])
    rows = []
    for frame in config['frames']:
        annotation = json.loads(annotations[frame].read_text())
        data, metadata, images, projections = prepare_frame(config,manifest,infos[frame],annotation,start)
        assert len(images) == 7 and data['img'].shape == (1,1,7,3,224,672)
        errors = []
        for camera_index, item in enumerate(manifest['videos']):
            intrinsic = np.asarray(infos[frame]['sensors']['cams'][item['camera']]['cam_intrinsic'])
            rotation = data['extrinsics'][0,0,camera_index,:3,:3].numpy()
            assert np.allclose(rotation@rotation.T, np.eye(3), atol=1e-4)
            assert np.isclose(np.linalg.det(rotation), 1., atol=1e-4)
            transform = np.asarray(metadata['ida_mat'][camera_index])
            ray = np.array([.2, -.1, 1.])
            pixel = intrinsic@ray
            expected = transform@pixel
            current = data['intrinsics'][0,0,camera_index,:3,:3].numpy()@ray
            errors.append(float(np.max(np.abs(current-expected))))
        assert max(errors) < 1e-3
        rows.append(dict(frame=frame, cameras=len(images), image_shape=list(data['img'].shape),
                         maximum_pixel_transform_error=max(errors),
                         camera_timestamp_spread_seconds=float(data['img_timestamp'].max()-data['img_timestamp'].min())))
    result = dict(passed=True, frames=rows, tests='image dimensions, SE3 extrinsics, resize/crop intrinsic consistency')
    output = Path(config['output'])
    (output/'input_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
