# 实验结果交付

## GeoUP：本地原始数据与 nuScenes

见 [交付说明](geoup/README.md)。包含全部八组实验的配置、复现脚本、环境清单以及逐文件校验清单。

- 本地完整场景：`172055_occ_fullscene`，348 帧。
- nuScenes：`nuscenes_scene0103_occ`，scene-0103，40 个关键帧，属于 val / mini_val。
- 保留早期诊断结果，具体结论见 `geoup/reproduction/` 下各实验说明。

完整结果约 3.16 GiB，已按最多 512 MiB 分卷打包，全部上传至 [GeoUP 完整结果 Release](https://github.com/MMMaverick/delivery/releases/tag/geoup-results-20261118)。共 14 个结果分卷和 3 个清单/验证附件。

文件校验：`geoup/manifests/SHA256SUMS`、`geoup/manifests/source_manifest.json`。
