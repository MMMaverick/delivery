import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']


def matrix(record):
    quaternion = np.asarray(record['rotation'],dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    scalar, first, second, third = quaternion
    rotation = np.array([
        [1-2*(second**2+third**2),2*(first*second-third*scalar),2*(first*third+second*scalar)],
        [2*(first*second+third*scalar),1-2*(first**2+third**2),2*(second*third-first*scalar)],
        [2*(first*third-second*scalar),2*(second*third+first*scalar),1-2*(first**2+second**2)]])
    result = np.eye(4)
    result[:3,:3] = rotation
    result[:3,3] = record['translation']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='/mnt/workspace/Data/nuScenes_mini')
    parser.add_argument('--scene',default='scene-0103')
    args = parser.parse_args()
    data = Path(args.data)
    tables = {name:{row['token']:row for row in json.loads((data/'v1.0-mini'/f'{name}.json').read_text())}
              for name in ['scene','sample','sample_data','calibrated_sensor','ego_pose','sensor']}
    scene = next(row for row in tables['scene'].values() if row['name']==args.scene)
    sample_data = {}
    for row in tables['sample_data'].values():
        if row['is_key_frame']:
            calibration = tables['calibrated_sensor'][row['calibrated_sensor_token']]
            channel = tables['sensor'][calibration['sensor_token']]['channel']
            sample_data[(row['sample_token'],channel)] = row
    run = ROOT/'local_inputs'/args.scene
    source = run/'source'
    annotations = source/'format_output/annotations/NV'
    annotations.mkdir(parents=True,exist_ok=True)
    infos = []
    token = scene['first_sample_token']
    while token:
        sample = tables['sample'][token]
        lidar = sample_data[(token,'LIDAR_TOP')]
        lidar_to_ego = matrix(tables['calibrated_sensor'][lidar['calibrated_sensor_token']])
        ego_to_world = matrix(tables['ego_pose'][lidar['ego_pose_token']])
        lidar_to_world = ego_to_world@lidar_to_ego
        info = dict(track_id=token,sensors=dict(cams={}),
                    ego2global_transformation_matrix_camera_reference_offset_optimized=lidar_to_world,
                    occ_ego2reference=np.linalg.inv(lidar_to_ego),occ_ego2global=ego_to_world,
                    reference_timestamp_ns=int(lidar['timestamp'])*1000)
        annotation = dict(track_id=token,cams={})
        for camera in CAMERAS:
            row = sample_data[(token,camera)]
            calibration = tables['calibrated_sensor'][row['calibrated_sensor_token']]
            camera_to_world = matrix(tables['ego_pose'][row['ego_pose_token']])@matrix(calibration)
            info['sensors']['cams'][camera] = dict(cam_intrinsic=calibration['camera_intrinsic'],
                camera_to_world_optimized=camera_to_world,extrinsic=np.linalg.inv(camera_to_world)@lidar_to_world)
            timestamp = int(row['timestamp'])*1000
            annotation['cams'][camera] = dict(timestamp=timestamp)
            original = data/row['filename']
            if not original.is_file():
                raise FileNotFoundError(original)
            target = source/'camera'/camera/f'{timestamp}.jpg'
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                target.symlink_to(original)
        (annotations/f'{len(infos):06d}.json').write_text(json.dumps(annotation))
        infos.append(info)
        token = sample['next']
    videos = []
    for camera in CAMERAS:
        with Image.open(next((source/'camera'/camera).glob('*.jpg'))) as image:
            width,height = image.size
        videos.append(dict(camera=camera,width=width,height=height))
    manifest = dict(source=str(source),videos=videos,dataset='nuScenes v1.0-mini',
                    scene=scene,frames=len(infos),bbox_reference='LIDAR_TOP',occ_reference='ego_at_lidar_timestamp',
                    input_3d_annotations_used=False)
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (run/'result_interpolation_optimized_first101.pkl').write_bytes(pickle.dumps(dict(infos=infos)))
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
