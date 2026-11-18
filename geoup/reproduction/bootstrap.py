import importlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def register_detection_depth(with_occ=False):
    for name in ['projects.mmdet3d_plugin', 'projects.mmdet3d_plugin.models',
                 'projects.mmdet3d_plugin.models.utils', 'projects.mmdet3d_plugin.models.dense_heads',
                 'projects.mmdet3d_plugin.models.backbones', 'projects.mmdet3d_plugin.models.necks',
                 'projects.mmdet3d_plugin.models.detectors']:
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT/Path(*name.split('.')))]
        module.__package__ = name
        sys.modules[name] = module
    for name in ['core.bbox.assigners.hungarian_assigner_3d', 'core.bbox.coders.nms_free_coder',
                 'core.bbox.match_costs', 'models.utils.petr_transformer',
                 'models.utils.positional_encoding', 'models.backbones.aggregator_vggt',
                 'models.necks.vggt_detection_neck', 'models.dense_heads.focal_head',
                 'models.dense_heads.petr_head_vggt', 'models.dense_heads.camera_head',
                 'models.dense_heads.dpt_head_pseudo', 'models.detectors.petr3d']:
        importlib.import_module('projects.mmdet3d_plugin.'+name)
    if with_occ:
        for name in ['models.utils.opus_v2_transformer','models.necks.vggt_occ_neck','models.dense_heads.opus_v2_head']:
            importlib.import_module('projects.mmdet3d_plugin.'+name)
