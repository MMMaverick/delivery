# 本地 GeoUP OCC 推理

配置：`local/172055_occ.yaml`。输入为 172055 clip 的 7 路相机、相机标定和自车位姿；不输入点云、雷达或目标 3D 标注。

运行（需要沙箱外 PPU 设备访问）：

```bash
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/run.py --config local/172055_occ.yaml
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/render_video.py --config local/172055_occ.yaml
```

加载官方 stage-2/nuscenes.pth 全部 1794 个非别名权重键，无遗漏参数。保留官方 4800 queries、5 层解码、8 帧历史、200×200×16 网格和 0.4m 体素；仅相机数量改为本数据的 7 路。起始历史按原实现重复首帧。主路径使用 FP32，避免本环境全局 BF16 出现非有限数值。

本机是 PPU：骨干、注意力、采样和体素特征在 PPU 上；MSMV 使用仓库自带 PyTorch grid_sample 参考实现，PFN 分组最大值使用 torch.scatter_reduce(amax)，两层稀疏卷积使用真实 CPU spconv。CPU spconv 不支持推理融合 bias，故卷积后单独加同一学习 bias。SparseConv3d 和 SubMConv3d 都与稠密 conv3d 在输出有效坐标上比对通过，微型测试最大绝对误差 5.96e-8。没有用空算子、随机权重或降低查询数替代 OCC。

兼容路径由 `tasks: [..., occ]` 和 `occ_backend: pytorch_sampling_cpu_sparse` 显式开启；原检测配置不启用 OCC。本次仅验证推理，不支持据此宣称训练后端可用。

结果在 `outputs/172055_occ/`：

- `results.json`：每帧联合 bbox/depth/camera/OCC 前向时间和体素类别计数。
- `checkpoint_validation.json`：实际权重装载核验。
- `*_occ.npz`：稀疏 locations/labels、稠密 grid、范围和体素尺寸。体素中心为 `[-40,-40,-1] + (locations+0.5)*0.4`，坐标相对输入参考自车坐标系。
- `*_occ.jpg`、`occ.mp4`：最高被预测占据体素的语义俯视图，覆盖 80m×80m；图上方为 +x，右方为 +y。不是 RGB 相机重投影图。
- `boxes.mp4`、`depth.mp4`：同一次前向的检测和深度。

类别 0–16 顺序：others, barrier, bicycle, bus, car, construction_vehicle, motorcycle, pedestrian, traffic_cone, trailer, truck, driveable_surface, other_flat, sidewalk, terrain, manmade, vegetation。17 为无通过阈值的预测（图中黑色），不能当成已确认可通行区域。保留模型原 score_thr；未做本地标注 mIoU 评估。

连续 12 帧验收完成：每帧 19682–26076 个占据体素，首帧联合前向 5.70s，后 11 帧平均 2.70s。头部最终输出有限且 logits 非常量，12 帧稀疏标签与稠密网格逐项一致，三个 MP4 均通过解码检查。时间包含 CPU 稀疏卷积及设备拷贝，不包含模型加载、图像读取、导出和视频编码；不是单独 OCC 延迟。实际记录见 results.json/status.json。
