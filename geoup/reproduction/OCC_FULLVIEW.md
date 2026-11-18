# 完整视野输入与 OCC 点云

配置：`local/172055_occ_fullview.yaml`；结果：`outputs/172055_occ_fullview/`。保留原裁剪实验及旧配置。

七路相机均等比缩放至宽 672，再在 448×672 画布居中补边。3840×2160 输入变为 672×378，上下各补 35；1920×1280 变为 672×448，不补边。没有裁掉原图。内参用实际整数缩放后的 sx、sy 和补边偏移更新，投影与输入使用同一个变换。图像高宽均为 DINO patch size 14 的整数倍。

官方模型原本使用底部裁剪；完整视野输入是本地推理适配，不能据此认定精度已提升，需要后续 GT 评测。图像像素数增加一倍，推理开销相应增加。模型权重、OCC 网格、查询数、解码层数和历史长度不变。

连续 12 帧来自原始 348 帧 clip 的开头，不是全长视频。视频按实际相机时间间隔播放，首末采样跨度约 1.2 秒。

产物：

- `input_fullview.jpg`：七路实际模型输入预览。
- `input_fullview_validation.json`：每路原始尺寸、完整保留检查和内参变换矩阵。
- `occ.mp4`：OCC 俯视图；`boxes.mp4` / `depth.mp4`：完整视野检测/深度视频。
- `000000_occ.ply` 至 `000011_occ.ply`：每帧语义占据点云，可用 CloudCompare / MeshLab 打开，带 RGB 与 semantic_label。
- `pointcloud_preview.png`：最后一帧点云三维预览。
- `pointcloud_classes.json`：点云语义类别表。

点云是网络预测的已占据体素中心，不是测量点云，也不是 RGB 深度反投影。单位米、输入参考自车坐标系，体素边长 0.4m，范围 x/y 为 [-40,40]、z 为 [-1,5.4]。未预测区域不导出；黑色区域不能解释为确认空闲。各帧独立导出，没有直接叠加动态车辆以免形成多帧拖影。

运行与导出：

```bash
export LD_LIBRARY_PATH=/mnt/workspace/Box_3D/runtime/envs/mono-detect-original-2dbox-full-gpu/lib:/usr/local/PPU_SDK/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/geoup-mpl
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/run.py --config local/172055_occ_fullview.yaml
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/render_video.py --config local/172055_occ_fullview.yaml
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/export_occ_cloud.py --config local/172055_occ_fullview.yaml
```

推理需沙箱外 PPU 访问；点云和视频导出不需要设备。
