import types

import torch
from spconv.pytorch.conv import SparseConvolution


def cpu_sparse_forward(module, inputs):
    result = module._conv_forward(False, inputs, module.weight, None)
    if module.bias is not None:
        result = result.replace_feature(result.features + module.bias)
    return result


def configure(model):
    def validate(module, inputs, outputs):
        scores = outputs['all_cls_scores'][-1]
        points = outputs['all_refine_pts'][-1]
        if not torch.isfinite(scores).all() or not torch.isfinite(points).all():
            raise ValueError('Nonfinite OCC logits or points')
        if scores.numel() and scores.float().std() == 0:
            raise ValueError('Constant OCC logits')

    model.occ_head.register_forward_hook(validate)
    for module in model.occ_head.modules():
        if module.__class__.__name__ == 'PFNLayer':
            module.use_torch_scatter_reduce = True
        if module.__class__.__name__ == 'Densifier':
            module.sparse_cpu = True
            module.cls_branch.cpu()
            for convolution in module.cls_branch.modules():
                if isinstance(convolution, SparseConvolution):
                    convolution.forward = types.MethodType(cpu_sparse_forward, convolution)


def export(output, frame, prediction):
    import cv2
    import numpy as np

    locations = np.asarray(prediction['occ_loc'], dtype=np.int64).reshape(-1, 3)
    labels = np.asarray(prediction['sem_pred'], dtype=np.uint8).reshape(-1)
    grid = np.full((200, 200, 16), 17, dtype=np.uint8)
    if len(locations):
        if not ((locations >= 0).all() and (locations < [200, 200, 16]).all()):
            raise ValueError('Occupancy outside official grid')
        grid[tuple(locations.T)] = labels
    np.savez_compressed(output / f'{frame:06d}_occ.npz', locations=locations,
                        labels=labels, grid=grid, voxel_size=.4,
                        point_cloud_range=np.array([-40, -40, -1, 40, 40, 5.4]))
    palette = np.array([[100,100,100],[255,120,50],[220,50,220],[255,200,0],
                        [30,80,255],[100,180,240],[180,80,200],[60,30,255],
                        [0,100,255],[100,200,255],[40,180,255],[100,100,100],
                        [150,150,150],[200,180,140],[80,130,100],[180,120,100],
                        [40,180,40],[15,15,15]], dtype=np.uint8)
    heights = np.where(grid != 17, np.arange(16)[None,None,:], -1).argmax(axis=2)
    bev = np.take_along_axis(grid, heights[:,:,None], axis=2)[:,:,0]
    image = cv2.resize(palette[bev][::-1], (800,800), interpolation=cv2.INTER_NEAREST)
    cv2.putText(image, f'OCC frame {frame}: {len(labels)} voxels; 80m x 80m',
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
    cv2.imwrite(str(output / f'{frame:06d}_occ.jpg'), image)
    return dict(occupied_voxels=len(labels), class_counts=np.bincount(labels, minlength=17).tolist())
