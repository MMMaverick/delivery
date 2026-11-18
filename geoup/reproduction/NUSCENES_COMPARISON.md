# nuScenes 同域场景对照

已从 nuScenes 官方公开地址下载 v1.0-mini：
`https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz`。
数据包存放 `/mnt/workspace/Data/nuScenes_mini/v1.0-mini.tgz`，公开包大小 4168148189 字节。只提取 metadata 和 samples 中的 2424 张相机图像，不需要解压 sweeps。数据目录下 extraction.json 记录来源。

选择 scene-0103，完整 40 个关键帧、六相机，首末相机时间跨度 19.4 秒。已从公开 nuscenes-devkit 1.2.0 包中的 splits.py 核对：该场景同时属于 val 和 mini_val。原始描述为：Many peds right, wait for turning car, long bike rack left, cyclist。

## 设置与几何

- 配置 `local/nuscenes_scene0103_occ.yaml`。
- 官方 stage-2/nuscenes.pth，1794 个非别名权重全部加载；主路径 FP32。
- 使用发布配置的 224×672 底部裁剪，六相机顺序与官方 converter 相同。这里刻意不采用本地数据的完整视野补边，避免把新的输入适配引入同域基线。
- 模型、网格、查询、8 帧历史和 PPU/CPU 混合兼容算子与此前相同。
- 从原始 calibrated_sensor、ego_pose、sample_data 生成输入；相机异步曝光的自车运动通过各自 camera-to-world 变换补偿。
- 检测参考系为 LIDAR_TOP，OCC 参考系为该 LiDAR 采样时刻的 ego。这是官方数据接口的两套参考系；OCC 投影额外右乘 ego2lidar，不把两者混为一谈。
- 不输入 sample_annotation、LiDAR 点云或雷达；使用 LiDAR 传感器标定与时间戳仅用于对齐官方参考系。

## 实测结果

40 帧全部完成，首帧联合 bbox/depth/camera/OCC 前向 5.25 秒，后 39 帧平均 2.15 秒（不含输入读取、模型加载与可视化导出）。每帧保留 32473–56002 个占据体素。最终输出有限且 logits 非常量。

视频和点云中可见道路、路边建筑、植被、车辆及行人；还有自行车等类别预测。三维预览中可见沿道路延伸的建筑立面和车辆体积。检测投影可识别多辆车及行人，同时仍能看到重复、误检等现象。

这只是官方同域样例的定性对照，不是官方精度复现：未下载 Occ3D 体素真值，未计算 mIoU/RayIoU。该片段与本地高速/宽道路片段的目标密度、类别和距离不同，不能凭占据点数、检测框数或类别数判断谁精度更高；也不能把两次不同相机数量与分辨率的耗时当作严格速度对比。

## 结果文件

目录：`outputs/nuscenes_scene0103_occ/`。

- `overview.mp4`：左侧六相机检测投影，右侧语义 OCC 俯视图，按真实采样时间播放。
- `occ.mp4` / `boxes.mp4` / `depth.mp4`：分别播放三种输出。
- `000000_occ.ply` 至 `000039_occ.ply`：每帧预测占据体素中心点云，ego 坐标，米制，0.4m 体素，带 RGB 与 semantic_label。
- `pointcloud_preview.png`：最后一帧三维预览。
- `*_occ.npz`：稀疏标签、位置以及稠密语义网格。
- `results.json` / `checkpoint_validation.json` / `validation.json`：实际统计、权重核验和输出/坐标变换检查。

复现：

```bash
python local/prepare_nuscenes.py --scene scene-0103
export LD_LIBRARY_PATH=/mnt/workspace/Box_3D/runtime/envs/mono-detect-original-2dbox-full-gpu/lib:/usr/local/PPU_SDK/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/geoup-mpl
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/run.py --config local/nuscenes_scene0103_occ.yaml
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/render_video.py --config local/nuscenes_scene0103_occ.yaml
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/export_occ_cloud.py --config local/nuscenes_scene0103_occ.yaml
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/render_occ_overview.py --config local/nuscenes_scene0103_occ.yaml
```

推理需要沙箱外 PPU 访问。既有本地数据配置及输出未覆盖。
