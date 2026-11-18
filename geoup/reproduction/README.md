# GeoUP本地PPU环境与七相机推理

代码：`/mnt/workspace/Box_3D/GeoUP`。

独立Python环境：`/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu`。现有SAM、SAM3与mesh环境未修改。该环境复用系统PPU PyTorch和已有基础库，因此不是脱离本机的完全独立发行环境。

## 运行

```bash
cd /mnt/workspace/Box_3D/GeoUP
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/run.py
/mnt/workspace/Box_3D/runtime/venvs/geoup-ppu/bin/python local/render_video.py
```

GPU推理需要可访问PPU的执行环境。`local/run.py`设置运行库路径，捕获子进程错误并记录`status.json`，不需要修改全局环境变量。

默认配置`local/172055_validated.yaml`：172055片段连续12帧、全部7相机、224×672输入、4帧流式历史、FP32主路径，FlashAttention内部仍遵循原代码的低精度实现；不是所有算子均为FP32。类别与权重仍为官方nuScenes定义。阈值0.25只用于可视化，不删除原始300候选。

默认输出：`outputs/172055_validated`。

12帧验收已完成：每帧超过0.25的候选为8/7/9/7/5/6/4/4/6/5/5/6，共72个观测，不是72辆独立车辆。首帧前向4.31秒，后11帧前向中位1.704秒，范围1.699～1.712秒；不包含模型加载、图像读取和结果导出。全部保存数组有限、框尺寸为正、分数不退化，两个视频均由ffmpeg完整解码检查。验收记录见`acceptance.json`。

- `results.json`：逐帧候选数、阈值通过数、最高分、深度范围、含GPU同步的前向时间。
- `000000.npz`等：原始300个3D候选、类别、分数、角点、七相机米制深度、投影矩阵与相机预测。
- `000000_boxes.jpg`、`000000_depth.jpg`等：七相机拼图。
- `boxes.mp4`、`depth.mp4`：慢放预览，显示5个输入帧/秒；原始采样约10Hz，不是实时速度展示。
- `checkpoint_validation.json`、`checkpoint_keys.json`：启用模块参数严格加载记录。
- `effective_model_config.py`、`input_config.json`：实际模型与输入配置。

## 任务范围

当前本地入口启用3D检测、深度和相机头，不启用OCC。OCC依赖spconv、torch_scatter与MSMV采样等额外算子，尚未完成本机PPU适配；不能把当前结果称为GeoUP所有任务跑通。没有修改OCC权重或用空结果冒充OCC预测。

`local/bootstrap.py`仅注册已启用的真实模块，避免检测/深度推理强制导入未安装的OCC和多数据集训练依赖。原Petr3D只增加OCC可选时的空分支，已有启用OCC的配置保持原行为。

## 输入适配

输入run：`/mnt/workspace/Box_3D/monocular-3d-detection-v1-restored/outputs/mesh_matched2d_expanded_20261116/172055/run`。

原始图像来自其manifest所指向的`Data0918/2026_08_20_17_20_55`，与现有mesh实验使用的数据一致。只读取图像、内外参、ego位姿和时间戳，不把已有GT框、track特征、雷达或LiDAR输入网络。已有ID列表只用于核对帧对齐。

外参采用输入PKL的reference ego→camera，world矩阵采用同一reference ego→global；若存在优化后的camera→world，则从它与reference world矩阵计算外参。送入GeoUP的`cam_extrinsics_global`为world→camera，与官方数据集定义一致。

各相机按实际尺寸等比缩放、底部裁剪至224×672，同步更新内参；保留原模型的RGB标准化参数。不同于直接使用nuScenes固定1600×900尺寸，避免本数据的3840×2160与1920×1280相机标定失配。输出3D框位于传入的reference ego坐标系，不是world坐标。

`local/verify_input.py`已验证首3帧：七相机同步差约6.2ms，SE3旋转与resize/crop投影一致，最大像素变换误差约1.22e-5。此检查不是检测精度评估，也不证明输入已完成镜头去畸变；模型跨域效果仍需GT评估。

## 环境与权重

本机是PPU，使用Python3.12、厂商PyTorch2.6.0、torchvision0.21.0、FlashAttention2.5.6，区别于官方Python3.9/PyTorch2.7.1/CUDA11.8环境。不覆盖系统PyTorch。

本地依赖记录见`environment-freeze.txt`。关键Python包为MMCV1.7.2、MMDetection2.28.2、MMSegmentation0.30.0；MMDetection3D使用官方v1.0.0rc6源码路径。

源码版本：

- GeoUP：`cd38f61fefb35e9b5529a860cb87c82669dea7b7`。
- MMDetection3D：`47285b3f1e9dba358e98fcd12e523cfd0769c876`。
- MMCV：`4c01b026f0afa5a91a5f54aea313788da1e40f95`。

MMCV真实CPU扩展已针对当前PyTorch编译，支持模块导入与CPU算子；没有假装安装NVIDIA CUDA算子。已启用的检测/深度路径主要在PPU上通过PyTorch和FlashAttention执行，不依赖这些MMCV CUDA算子。MMCV vendor路径通过本环境的`geoup_mmcv.pth`优先加载。MMDetection3D按官方GeoUP说明将MMCV上限放宽至1.7.2。MMCV构建改用`os.sched_getaffinity`查询CPU数，避免沙箱PID命名空间导致psutil报错。

官方权重：`/mnt/workspace/Box_3D/models/geoup/stage-2/nuscenes.pth`，5414510240字节。

SHA256：`ed318ead350614fbb48935dec8bd866090a4b532f70329ace1f1bdf0cfa1f05a`，已与官方LFS发布值核对一致。官方站连接超时后通过hf-mirror下载同一权重，没有用其他模型代替。

启用模块1426个state_dict键严格加载，368个OCC头/neck键明确不加载。权重中的`real_pc_range`是运行时生成的`pc_range`别名，先验证两者逐值相同再去除重复键；保留OrderedDict的模块版本元数据，防止旧版MMDetection错误重命名头参数。

构建阶段跳过会被权重完整覆盖的截断正态随机初始化，避免本机该CPU初始化函数偶发段错误；全部启用参数由官方权重严格赋值后才推理。没有保留未加载的随机参数，也没有更改训练好的参数。

## 数值问题与已保留的对照

最初对整个模型开启BF16 autocast，位置编码出现非有限数，随后检测注意力变成NaN；原代码`nan_to_num`把它清零，导致300候选分数和尺寸相同。`outputs/172055_smoke`及`outputs/172055_diagnostic`保留这批异常结果，不应采用。

改回FP32主路径后，三帧得到8/7/9个分数超过0.25的候选，最高分0.710/0.797/0.811，检测头中间输出非有限数检查为0。对照保留在`outputs/172055_fp32`，此轮带大量诊断hook，不能用于纯模型速度基准。默认验收配置保持FP32，不降低阈值掩盖问题。

这只说明本数据推理链路和数值检查通过，不等于达到官方nuScenes指标。车辆框仍有错位、重复与漏检，尚未做该域微调或完整GT评估；深度也未用独立GT验证。不要直接替换已有生产流程。
