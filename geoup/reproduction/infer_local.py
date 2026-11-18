import argparse
from collections import OrderedDict
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local.bootstrap import register_detection_depth


def write(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare_frame(config, manifest, info, annotation, start_timestamp):
    height, width = config['image_height'], config['image_width']
    source = Path(manifest['source'])
    world = np.asarray(info['ego2global_transformation_matrix_camera_reference_offset_optimized'], dtype=np.float64)
    images, displays, intrinsics, extrinsics, paths, transforms, timestamps = [], [], [], [], [], [], []
    for item in manifest['videos']:
        camera = item['camera']
        raw = annotation['cams'][camera]
        image_path = source/'camera'/camera/f"{raw['timestamp']}.jpg"
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            original_width, original_height = image.size
            if (original_width, original_height) != (item['width'], item['height']):
                raise ValueError(f'Image size disagrees with calibration manifest: {camera}')
            mode = config.get('image_fit', 'bottom_crop')
            if mode == 'letterbox':
                scale = min(height/original_height, width/original_width)
                resized_width, resized_height = round(original_width*scale), round(original_height*scale)
                left, top = (width-resized_width)//2, (height-resized_height)//2
                canvas = Image.new('RGB', (width,height), (124,116,104))
                canvas.paste(image.resize((resized_width,resized_height)), (left,top))
                image = canvas
                transform = np.array([[resized_width/original_width,0.,left],
                                      [0.,resized_height/original_height,top],[0.,0.,1.]])
            elif mode == 'bottom_crop':
                scale = max(height/original_height, width/original_width)
                resized_width, resized_height = int(original_width*scale), int(original_height*scale)
                crop_left = int(max(0, resized_width-width)/2)
                crop_top = resized_height-height
                image = image.resize((resized_width, resized_height)).crop((crop_left, crop_top, crop_left+width, crop_top+height))
                transform = np.array([[scale,0.,-crop_left],[0.,scale,-crop_top],[0.,0.,1.]])
            else:
                raise ValueError(f'Unknown image_fit: {mode}')
            rgb = np.asarray(image).copy()
        sensor = info['sensors']['cams'][camera]
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = transform@np.asarray(sensor['cam_intrinsic'])
        if sensor.get('camera_to_world_optimized') is not None:
            external = np.linalg.inv(np.asarray(sensor['camera_to_world_optimized']))@world
        else:
            external = np.asarray(sensor['extrinsic'], dtype=np.float64)
        if external.shape != (4, 4) or not np.isfinite(external).all():
            raise ValueError('Invalid extrinsic')
        images.append(((rgb.astype(np.float32)-[123.675,116.28,103.53])/[58.395,57.12,57.375]).transpose(2,0,1))
        displays.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        intrinsics.append(intrinsic)
        extrinsics.append(external)
        paths.append(str(image_path))
        transforms.append(transform.tolist())
        timestamps.append((int(raw['timestamp'])-start_timestamp)/1e9)

    def tensor(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=config['device'])

    intrinsic_tensor = tensor(intrinsics)
    extrinsic_tensor = tensor(extrinsics)
    world_tensor = tensor(world)
    data = dict(img=tensor(images)[None,None], intrinsics=intrinsic_tensor[None,None],
        extrinsics=extrinsic_tensor[None,None], lidar2img=(intrinsic_tensor@extrinsic_tensor)[None,None],
        cam_extrinsics_global=(extrinsic_tensor@torch.linalg.inv(world_tensor))[None,None],
        ego_pose=world_tensor[None], ego_pose_inv=torch.linalg.inv(world_tensor)[None],
        timestamp=tensor([timestamps[0]]), img_timestamp=tensor(timestamps)[None,None])
    if 'reference_timestamp_ns' in info:
        data['timestamp'] = tensor([(info['reference_timestamp_ns']-start_timestamp)/1e9])
    metadata = dict(scene_token=source.name, filename=paths, img_shape=[(height,width,3)]*len(paths),
        pad_shape=[(height,width,3)]*len(paths), ori_shape=[(item['height'],item['width'],3) for item in manifest['videos']],
        box_type_3d=None, ida_mat=transforms)
    return data, metadata, displays, np.asarray(intrinsics)@np.asarray(extrinsics)


def render(output, frame, result, displays, projections, manifest, names, threshold):
    boxes = result['boxes_3d']
    scores = result['scores_3d'].float().cpu().numpy()
    labels = result['labels_3d'].cpu().numpy()
    corners = boxes.corners.float().cpu().numpy()
    depth = result['depth_pred'].float().cpu().numpy()
    if not np.isfinite(boxes.tensor.float().cpu().numpy()).all() or not np.isfinite(depth).all():
        raise ValueError('Nonfinite model output')
    camera_outputs = {key:result[key].detach().float().cpu().numpy()
                      for key in ['cam_pose_pred','cam_extrinsics'] if key in result}
    np.savez_compressed(output/f'{frame:06d}.npz', boxes=boxes.tensor.float().cpu().numpy(),
        corners=corners, scores=scores, labels=labels, depth_m=depth, projections=projections, **camera_outputs)
    panels, depth_panels = [], []
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for camera_index, (image, projection, item) in enumerate(zip(displays, projections, manifest['videos'], strict=True)):
        image = image.copy()
        for box_index in np.flatnonzero(scores >= threshold):
            homogeneous = np.column_stack([corners[box_index], np.ones(8)])
            projected = (projection@homogeneous.T).T
            visible = projected[:,2] > .1
            pixels = projected[:,:2]/np.maximum(projected[:,2:3], .1)
            pixels = np.clip(pixels, -100000, 100000).astype(np.int32)
            for first, second in edges:
                if visible[first] and visible[second]:
                    cv2.line(image, tuple(pixels[first]), tuple(pixels[second]), (30,230,50), 1, cv2.LINE_AA)
            inside = visible & (pixels[:,0]>=0) & (pixels[:,0]<image.shape[1]) & (pixels[:,1]>=0) & (pixels[:,1]<image.shape[0])
            if inside.any():
                anchor = pixels[np.flatnonzero(inside)[0]]
                cv2.putText(image, f'{names[int(labels[box_index])]} {scores[box_index]:.2f}', tuple(anchor),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (0,240,240), 1)
        camera_depth = np.squeeze(depth[camera_index])
        colored = cv2.applyColorMap((np.clip(camera_depth/90,0,1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored = cv2.resize(colored, (image.shape[1],image.shape[0]))
        cv2.putText(image, item['camera'], (8,18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        cv2.putText(colored, item['camera']+' depth 0-90m', (8,18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        panels.append(image)
        depth_panels.append(colored)
    for collection, label in [(panels,'boxes'),(depth_panels,'depth')]:
        if len(collection)%2:
            collection.append(np.zeros_like(collection[0]))
        mosaic = np.concatenate([np.concatenate(collection[index:index+2], axis=1) for index in range(0,len(collection),2)])
        cv2.imwrite(str(output/f'{frame:06d}_{label}.jpg'), mosaic)
    return dict(frame=frame, boxes=len(scores), boxes_above_threshold=int((scores>=threshold).sum()),
        degenerate_detection=bool(len(scores)>1 and np.ptp(scores)==0),
        score_max=float(scores.max()) if len(scores) else 0., depth_min_m=float(depth.min()),
        depth_median_m=float(np.median(depth)), depth_max_m=float(depth.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='local/172055_validated.yaml')
    arguments = parser.parse_args()
    config = yaml.safe_load((ROOT/arguments.config).read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    write(output/'status.json', dict(state='running', stage='imports'))
    try:
        with_occ = 'occ' in config['tasks']
        if set(config['tasks']) - {'bbox','depth','camera','occ'}:
            raise ValueError('Unsupported tasks')
        if config['precision'] not in ['float32','bfloat16']:
            raise ValueError('Unsupported precision')
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU unavailable; run outside sandbox with PPU runtime')
        register_detection_depth(with_occ=with_occ)
        from mmcv import Config
        from mmdet3d.models import build_model
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        model_config = Config.fromfile(str(ROOT/config['model_config']))
        architecture = copy.deepcopy(model_config.model)
        if not with_occ:
            architecture.occ_head = None
            architecture.occ_neck = None
        for name in ['img_backbone','depth_head','camera_head']:
            architecture[name]['init_cfg'] = None
        camera_count = len(json.loads((Path(config['input_run'])/'manifest.json').read_text())['videos'])
        if with_occ:
            architecture.occ_head.transformer.num_views = camera_count
        architecture.img_backbone.seq_info = dict(seq_length=4,batch_size=camera_count)
        model_config.model = architecture
        write(output/'status.json', dict(state='running', stage='build_model'))
        with patch('torch.nn.init.trunc_normal_', lambda tensor, *args, **kwargs: tensor):
            model = build_model(architecture)
        write(output/'status.json', dict(state='running', stage='load_checkpoint'))
        checkpoint = torch.load(config['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        state = checkpoint['state_dict']
        expected = model.state_dict()
        alias = 'pts_bbox_head.real_pc_range'
        if alias in state and alias not in expected:
            if not torch.equal(state[alias], state['pts_bbox_head.pc_range']):
                raise ValueError('Saved dynamic range is not the pc_range alias')
            state.pop(alias)
        missing = sorted(set(expected)-set(state))
        unexpected = sorted(set(state)-set(expected))
        write(output/'checkpoint_keys.json', dict(missing=missing, unexpected=unexpected,
            unexpected_outside_disabled_heads=[key for key in unexpected if not key.startswith(('occ_head.','occ_neck.'))]))
        if missing or any(not key.startswith(('occ_head.','occ_neck.')) for key in unexpected):
            raise RuntimeError(f'Unexpected checkpoint mismatch: missing={missing[:12]} unexpected={unexpected[:12]}')
        selected = OrderedDict((key,state[key]) for key in expected)
        if hasattr(state, '_metadata'):
            selected._metadata = state._metadata
        write(output/'status.json', dict(state='running', stage='load_state_dict'))
        model.load_state_dict(selected, strict=True, assign=True)
        write(output/'checkpoint_validation.json', dict(expected_keys=len(expected), loaded_keys=len(selected),
            ignored_keys=len(unexpected), ignored_prefixes=[] if with_occ else ['occ_head','occ_neck'], missing_keys=missing,
            random_unloaded_parameters=False))
        del selected, checkpoint, state
        write(output/'status.json', dict(state='running', stage='move_to_gpu'))
        model.to(config['device']).eval()
        if with_occ:
            if config.get('occ_backend') != 'pytorch_sampling_cpu_sparse':
                raise ValueError('Explicit OCC backend required on PPU')
            from local.occ_compat import configure
            configure(model)
        diagnostics = []
        handles = []
        if config.get('diagnostics'):
            def tensors(value):
                if torch.is_tensor(value):
                    yield value
                elif isinstance(value, (list, tuple)):
                    for entry in value:
                        yield from tensors(entry)
                elif isinstance(value, dict):
                    for entry in value.values():
                        yield from tensors(entry)

            def hook(name):
                def record(module, inputs, result):
                    values = []
                    for value in tensors(result):
                        if value.is_floating_point() and value.numel():
                            value = value.detach().float()
                            values.append(dict(shape=list(value.shape), finite_fraction=float(torch.isfinite(value).float().mean()),
                                maximum_absolute=float(torch.nan_to_num(value).abs().max()),
                                std=float(torch.nan_to_num(value).std()) if value.numel()>1 else 0.))
                    diagnostics.append(dict(module=name, outputs=values))
                return record

            for name, module in model.named_modules():
                if name.startswith('pts_bbox_head') or name in ['img_neck','img_backbone','depth_head']:
                    handles.append(module.register_forward_hook(hook(name)))
        run = Path(config['input_run'])
        manifest = json.loads((run/'manifest.json').read_text())
        infos = pickle.loads((run/'result_interpolation_optimized_first101.pkl').read_bytes())['infos']
        annotations = sorted((Path(manifest['source'])/'format_output/annotations/NV').glob('*.json'))
        if config['frames'] == 'all':
            if len(infos) != len(annotations):
                raise ValueError('Full scene requires metadata for every annotation frame')
            config['frames'] = list(range(len(infos)))
        if config['frames'] != sorted(set(config['frames'])):
            raise ValueError('Streaming frames must be strictly increasing')
        first_annotation = json.loads(annotations[config['frames'][0]].read_text())
        first_timestamp = int(first_annotation['cams'][manifest['videos'][0]['camera']]['timestamp'])
        summaries = []
        (output/'effective_model_config.py').write_text(model_config.pretty_text)
        write(output/'input_config.json', config)
        for frame in config['frames']:
            write(output/'status.json', dict(state='running', stage='inference', frame=frame))
            annotation = json.loads(annotations[frame].read_text())
            if annotation['track_id'] != infos[frame]['track_id']:
                raise ValueError('Input frame alignment mismatch')
            data, metadata, displays, projections = prepare_frame(config,manifest,infos[frame],annotation,first_timestamp)
            metadata['box_type_3d'] = LiDARInstance3DBoxes
            if with_occ:
                occ_ego2reference = torch.as_tensor(infos[frame].get('occ_ego2reference',np.eye(4)),
                                                   device=config['device'],dtype=torch.float32)
                occ_world = torch.as_tensor(infos[frame].get('occ_ego2global',
                    infos[frame]['ego2global_transformation_matrix_camera_reference_offset_optimized']),
                    device=config['device'],dtype=torch.float32)
                metadata.update(ego2img=data['lidar2img'][0,0]@occ_ego2reference,
                                ego2occ=torch.eye(4, device=config['device']),
                                ego2global=occ_world)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            precision = torch.autocast('cuda',dtype=torch.bfloat16) if config['precision']=='bfloat16' else nullcontext()
            with torch.inference_mode(), precision:
                result = model.simple_test([metadata], **data)[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            summary = render(output,frame,result,displays,projections,manifest,model_config.class_names,config['score_threshold'])
            summary['inference_seconds'] = seconds
            summary['timestamp_seconds'] = (int(annotation['cams'][manifest['videos'][0]['camera']]['timestamp'])-first_timestamp)/1e9
            if with_occ:
                from local.occ_compat import export
                summary.update(export(output, frame, result['occ_pred']))
            summaries.append(summary)
            if diagnostics:
                write(output/'diagnostics.json', diagnostics)
            write(output/'results.json', dict(frames=summaries, tasks=config['tasks'], cameras=len(manifest['videos']),
                scope='Local-domain smoke inference, not benchmark accuracy',
                coordinate_frame=manifest.get('bbox_reference','Input reference ego frame; not world coordinates'),
                input_3d_boxes_used=False, lidar_or_radar_input_used=False))
            print(json.dumps(summary),flush=True)
        quality_valid = not any(row['degenerate_detection'] for row in summaries)
        write(output/'status.json', dict(state='complete' if quality_valid else 'complete_with_warnings',
            frames=len(summaries), tasks=config['tasks'], detection_not_degenerate=quality_valid))
    except BaseException as error:
        write(output/'status.json', dict(state='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
