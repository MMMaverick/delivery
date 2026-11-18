# GeoUP 本地数据与 nuScenes 推理结果

完整交付包含 GeoUP `outputs/` 中全部八组实验结果，包含早期诊断/失败对照，不仅限最终成功结果。`reproduction` 包含运行脚本、配置、环境清单及实验说明。

重点结果：

- `172055_occ_fullscene`：本地完整场景，点云、OCC、检测、深度及视频。
- `172055_occ_fullview`：保留完整相机视野的本地测试。
- `nuscenes_scene0103_occ`：nuScenes mini scene0103 推理结果。
- 其他包保留初始 smoke、FP32、validated、diagnostic 和 OCC 实验。

每组压缩包按最多 512 MiB 分卷，适合上传 GitHub Release。下载全部分卷后，在空目录解压：

```bash
sha256sum -c SHA256SUMS
cat reproduction.tar.gz.part* | tar -xz
cat 172055_occ_fullscene.tar.gz.part* | tar -xz
cat nuscenes_scene0103_occ.tar.gz.part* | tar -xz
```

其他组按同样方式解压。压缩包保留 `local/`、`outputs/<实验名>/` 相对路径。`source_manifest.json` 提供每个原始文件的 SHA256 和字节数；`SHA256SUMS` 校验分卷。

不包含模型权重、运行环境二进制和原始输入数据集。配置中现有绝对路径需根据部署位置调整。预测及可视化不等于独立 GT 精度评测；训练集/验证集及完整视野适配说明见 reproduction 中 `local/NUSCENES_COMPARISON.md`、`local/OCC_FULLVIEW.md`、`local/LOCAL_FULLSCENE.md`。

仓库用于索引、说明和配置；完整结果分卷已上传至 [GitHub Release](https://github.com/MMMaverick/delivery/releases/tag/geoup-results-20261118)，避免将 GB 级产物写入 Git 历史。下载全部 14 个分卷及校验清单后，按上述命令验证和解压。
