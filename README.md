# 实验结果交付

## GeoUP：本地原始数据与 nuScenes

见 [交付说明](geoup/README.md)。包含全部八组实验的配置、复现脚本、环境清单以及逐文件校验清单。

- 本地完整场景：`172055_occ_fullscene`，348 帧。
- nuScenes：`nuscenes_scene0103_occ`，scene-0103，40 个关键帧，属于 val / mini_val。
- 保留早期诊断结果，具体结论见 `geoup/reproduction/` 下各实验说明。

完整结果约 3.16 GiB，已按最多 512 MiB 分卷打包，计划通过本仓库 Release 发布。**当前仅说明、配置、代码和清单入库，完整二进制结果尚未上传。**

文件校验：`geoup/manifests/SHA256SUMS`、`geoup/manifests/source_manifest.json`。
