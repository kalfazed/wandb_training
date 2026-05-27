# wandb_training — 一个用 NuScenes 风格数据学 PyTorch / Lightning 的小项目

我们在一份 **只有 2 帧 LIDAR + 标注** 的 J6Gen2 mini 数据集上，写一个能真正训练的 **3D 物体检测器**（CenterPoint 风格的 anchor-free BEV 检测，先做单类 `car`）。整个项目按一系列**渐进式练习**组织：第一关用纯 PyTorch 写出训练循环，之后**每一关把工程层（训练循环、设备搬运、AMP、checkpoint、logging…）逐步替换成 PyTorch Lightning**，让你直观体会 Lightning 帮你做了哪些事。

## 数据集

默认路径（可在每个 `train.py` 里改 `Config.data_root`）：

```
/mnt/data_archive/test/j6gen2/e0305816-afe6-4c89-9b5d-1b8aaab1f8b1
```

NuScenes 格式：`annotation/*.json` + `data/LIDAR_CONCAT/*.pcd.bin`。该 sample 约有 **191 个 LIDAR sweep**（每帧 ~24 万点）。练习脚本默认对点云做随机下采样（`--max-points 60000`），并可用 `--max-frames 8` 做快速 smoke test。

LIDAR_CONCAT 的标定外参是 identity，所以 LIDAR 坐标系 = ego 坐标系。GT 框在 `sample_annotation.json` 里以 **global 坐标** 存储，我们用对应 sample_data 的 `ego_pose` 把它变换到 ego/LIDAR 帧。

## 安装

```bash
pip install -r requirements.txt   # 需要 PyTorch ≥ 2.0 + NumPy
```

若你已有 conda 环境（例如带 CUDA 的 `fs-perc-train`），可直接用该环境的 Python 运行，不必新建 venv。

## 项目结构

```
wandb_training/
├── nusc_det/                # 共享核心（dataset / model / loss / 几何 / BEV）
│   ├── io.py
│   ├── geometry.py
│   ├── dataset.py
│   ├── voxelize.py
│   ├── targets.py
│   ├── model.py
│   └── losses.py
└── exercises/
    ├── 01_pure_pytorch/     # ← 当前这一关
    │   └── train.py
    ├── 02_lightning_module/
    ├── 03_lightning_datamodule/
    ├── 04_callbacks_wandb/
    ├── 05_trainer_flags/
    ├── 06_custom_callback/
    ├── 07_load_predict_resume/     (load / predict / resume)
    └── 08_hydra_omegaconf/         (Hydra + OmegaConf 配置驱动)
```

后续每一关只是新建一个 `exercises/0N_xxx/train.py`，**复用 `nusc_det/` 里的 dataset/model/loss**，让你看到"研究代码不动，工程代码替换"的过程。

## 怎么跑第 01 关

```bash
# 快速 smoke test（8 帧，~10 秒）
python exercises/01_pure_pytorch/train.py --max-frames 8 --epochs 30

# 全量 191 帧（较慢，建议 GPU）
python exercises/01_pure_pytorch/train.py --epochs 50
```

期望日志（smoke test，值大致量级）：

```
[data] 8 lidar frames available
  frame 0: points=237643  cars= 20  file=00000.pcd.bin
  ...
[bev] grid: 250 x 250 (voxel_size=0.4m)
[model] BEVDetector  params=0.10M  device=cuda
[epoch    0] loss=7.3015  hm=6.7666  reg=5.3485  ...
[epoch   29] loss=0.5853  hm=0.4672  reg=1.1803  ...
[done] saved checkpoint -> runs/01_pure_pytorch/final.pt
```

如果 loss 不下降，多半是几何 / BEV 栅格化 / target 渲染对不上格点，先看 `nusc_det/targets.py` 和 `voxelize.py` 的注释；它们有意写得直白方便你打断点。

## 各练习的"diff 主题"

| # | 主题 | 你将看到的 diff |
|---|---|---|
| **01** | 纯 PyTorch 训练循环 | 显式 `for epoch / for batch / loss.backward / optimizer.step`，手动 `.to(device)` |
| 02 | 重构为 `LightningModule` | 删掉显式循环，把 step 拆成 `training_step` / `configure_optimizers` |
| 03 | 抽出 `LightningDataModule` | dataset 逻辑独立 |
| 04 | `WandbLogger` + `ModelCheckpoint` + `LearningRateMonitor` | wandb 上看 loss 曲线 + BEV heatmap 可视化 |
| 05 | Trainer flag 一行切换 `precision="bf16-mixed"` / `accumulate_grad_batches` | 体会"工程优化在框架层免费完成" |
| 06 | 自定义 Callback：dump 预测 BEV heatmap | 理解 hook 系统 |
| 07 | Checkpoint: load / predict / resume | `load_from_checkpoint`, `trainer.predict`, `trainer.fit(ckpt_path=...)` |
| 08 | Hydra + OmegaConf 配置驱动 | `@hydra.main`、`_target_` + `instantiate`、`defaults` 组合、`${...}` 插值 |
| 09 *(可选)* | `automatic_optimization=False` + 多 optimizer | 进阶用法，需要时再补 |
