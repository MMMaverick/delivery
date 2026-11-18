# 本地完整场景 OCC 与三维点云视频

场景：2026_08_20_17_20_55。配置 `local/172055_occ_fullscene.yaml`，输出目录 `outputs/172055_occ_fullscene/`。

本次覆盖原始 NV 标注列表中的全部 348 个时刻、七路相机，逐帧连续推理，不做抽帧。旧实验的优化 PKL 只有 101 帧；本次通过 `local/prepare_local_fullclip.py` 从原始标注文件提取相机标定、时间戳及自车位姿，建立完整输入索引。已逐项核对前 101 帧，与旧输入的内参、外参、位姿最大误差全部为 0，所有 348×7 张图像存在。原始标注中的 2D/3D 框和类别不作为模型输入，track_id 只用于原有输入索引对齐检查。

首末采样跨度 59.100008 秒。各相机目录实际图像数不同（453–593 张）；本次处理全部具有原始 NV 同步索引的 348 个时刻，不是各路所有原始曝光的并集。中间缺少同步索引的时间段没有插值生成预测，视频在这类间隔保持前一帧直至下一真实采样时刻，画面时间戳可用于识别这些间隔。

七路图像使用 448×672 等比缩放、居中补边，保留完整视野，同步更新投影内参。权重、查询数、OCC 体素范围和尺寸、解码层数及时间历史均沿用已有配置。官方模型的原始裁剪与当前补边有分布差异，本实验不是本地精度评估。

已完成全部 348 帧：首帧联合前向 8.47 秒，后 347 帧平均 5.38 秒；每帧保留 18042–40657 个占据体素。348 份 PLY 全部导出，所有稀疏语义标签与稠密网格对应一致，五个 MP4 均完成编码及解码检查。上述时间仅为联合前向，未包含读图、模型加载、文件导出和视频编码。

结果：

- `pointcloud.mp4`：完整时间序列的三维语义点云预览，固定斜视角、坐标轴、语义颜色及时间戳。点云是每帧预测占据体素中心，0.4m 分辨率，不是传感器实测点云。
- `overview.mp4`：七路相机检测投影与 OCC 俯视图同屏。
- `occ.mp4` / `boxes.mp4` / `depth.mp4`：完整 OCC、3D 检测和深度视频。
- `*_occ.ply`：每帧独立语义点云，RGB＋semantic_label，可在 CloudCompare / MeshLab 中查看。
- `*_occ.npz`：原始语义体素；`results.json`：逐帧时间及计数。
- `final_validation.json`：全片帧数、时长、推理统计、输出一致性与视频验收。该文件仅在所有导出与检查完成后生成。

视频使用真实采样时间间隔。三维点云视频每帧显示当前自车坐标系下的预测，背景随自车运动是正常现象。没有把移动车辆跨帧直接累积，也没有对预测强行平滑。缺少预测的区域不能当作确认空闲。

复现：

```bash
cd /mnt/workspace/Box_3D/GeoUP
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/prepare_local_fullclip.py
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/run.py --config local/172055_occ_fullscene.yaml
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/finalize_scene.py --config local/172055_occ_fullscene.yaml
```

推理需沙箱外设备访问。后处理会导出逐帧 PLY，生成全部视频并解码验证。旧 12 帧本地实验、nuScenes 对照实验及原工程流程均保留。
