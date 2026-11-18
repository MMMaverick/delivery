import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def main():
    previous = Path('/mnt/workspace/Box_3D/monocular-3d-detection-v1-restored/outputs/mesh_matched2d_expanded_20261116/172055/run')
    manifest = json.loads((previous/'manifest.json').read_text())
    old = pickle.loads((previous/'result_interpolation_optimized_first101.pkl').read_bytes())['infos']
    annotations = sorted((Path(manifest['source'])/'format_output/annotations/NV').glob('*.json'))
    infos = []
    maximum_pose_error, maximum_intrinsic_error, maximum_extrinsic_error = 0.,0.,0.
    for index,path in enumerate(annotations):
        annotation = json.loads(path.read_text())
        world = np.eye(4)
        world[:3,:3] = Rotation.from_quat(annotation['ego2global_rotation']).as_matrix()
        world[:3,3] = annotation['ego2global_translation']
        cameras = {}
        for item in manifest['videos']:
            name = item['camera']
            sensor = annotation['cams'][name]
            image = Path(manifest['source'])/'camera'/name/f"{sensor['timestamp']}.jpg"
            if not image.is_file():
                raise FileNotFoundError(image)
            cameras[name] = {key:sensor[key] for key in ['cam_intrinsic','extrinsic']}
            if index<len(old):
                previous_sensor = old[index]['sensors']['cams'][name]
                maximum_intrinsic_error = max(maximum_intrinsic_error,float(np.abs(np.asarray(sensor['cam_intrinsic'])-previous_sensor['cam_intrinsic']).max()))
                maximum_extrinsic_error = max(maximum_extrinsic_error,float(np.abs(np.asarray(sensor['extrinsic'])-previous_sensor['extrinsic']).max()))
        if index<len(old):
            maximum_pose_error = max(maximum_pose_error,float(np.abs(world-old[index]['ego2global_transformation_matrix_camera_reference_offset_optimized']).max()))
        infos.append(dict(track_id=annotation['track_id'],sensors=dict(cams=cameras),
                          ego2global_transformation_matrix_camera_reference_offset_optimized=world))
    if max(maximum_pose_error,maximum_intrinsic_error,maximum_extrinsic_error)>1e-5:
        raise ValueError('Raw metadata differs from previous validated inputs')
    run = ROOT/'local_inputs/172055_full'
    run.mkdir(parents=True,exist_ok=True)
    manifest['frames'] = len(infos)
    manifest['input_3d_annotations_used'] = False
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (run/'result_interpolation_optimized_first101.pkl').write_bytes(pickle.dumps(dict(infos=infos)))
    report = dict(frames=len(infos),cameras=len(manifest['videos']),previous_frames_checked=len(old),
                  maximum_pose_error=maximum_pose_error,maximum_intrinsic_error=maximum_intrinsic_error,
                  maximum_extrinsic_error=maximum_extrinsic_error,all_images_present=True)
    (run/'validation.json').write_text(json.dumps(report,indent=2))
    print(report)


if __name__ == '__main__':
    main()
