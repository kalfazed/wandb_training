# 练习 01 — 纯 PyTorch 训练循环

## 这一关在做什么

在我们正式接触 PyTorch Lightning **之前**，先用**最朴素的 PyTorch** 写一个能真正训练的 3D 物体检测器。  
任务：**单类 car 检测**，输入是 NuScenes 风格的 LIDAR 点云，输出是 BEV 上的中心点热力图 + 框尺寸 / 朝向回归（CenterPoint 风格）。

这一关刻意把所有"工程层样板代码"全部摆在明面上，**让你先体会到痛点**。后面每一关 Lightning 都会帮你删掉其中一部分，那时你才会感觉到 Lightning 真正的价值。

> 一句话总结：**01 关是为后面所有 Lightning 关做参照系的"对照组"**。

---

## 数据流（一图看懂）

```
              annotation/*.json           data/LIDAR_CONCAT/*.pcd.bin
                    │                              │
                    ▼                              ▼
        NuScenesLidarDetDataset.__getitem__
        ├─ read_lidar_pcd_bin()        →   points (N, 4)  [x,y,z,intensity]
        └─ sample_annotation + ego_pose
            └─ transform_global_to_ego →   List[Box3D] in ego frame
                    │
                    ▼
        points_to_bev()                 →   BEV (5, 250, 250)
                                            通道: log(count), mean_z, max_z,
                                                 mean_intensity, std_z
                    │
                    ▼
        BEVDetector (stem + 4×ResBlock + head)
        ├─ heatmap logits  (1, 250, 250)
        └─ regression      (8, 250, 250)
                                            通道 0..1  : 子像素偏移 dx, dy
                                            通道 2..3  : log(w), log(l)
                                            通道 4    : z (米)
                                            通道 5..6 : sin(yaw), cos(yaw)
                                            通道 7    : 类别 id（单类时恒为 0）
                    │                              │
                    └────────── ↓ ────────────────┘
                          detection_loss
                          ├─ focal_loss (heatmap, CornerNet 改进版)
                          └─ masked L1  (reg, 仅在 GT 中心格点上算)
```

GT 在 `sample_annotation.json` 里以 **全局坐标** 存储，需用对应 sweep 的 `ego_pose` 变到 ego/LIDAR 系（`LIDAR_CONCAT` 外参恰好是 identity，所以 LIDAR 系 = ego 系）。

---

## 文件清单（这一关相关的所有代码）

| 路径 | 作用 | 你应该重点读什么 |
|------|------|-----------------|
| `nusc_det/io.py` | 读 `pcd.bin`、加载 `annotation/*.json` 并按 token 建索引 | `read_lidar_pcd_bin` 怎么自动识别 4 / 5 列布局 |
| `nusc_det/geometry.py` | 四元数→旋转矩阵、yaw、global→ego 变换 | `transform_global_to_ego` 的两步：去平移→去旋转 |
| `nusc_det/dataset.py` | `NuScenesLidarDetDataset`：一帧 = 点云 + List[Box3D] | `_boxes_for_sample`：怎么把 instance→category→ego 框串起来 |
| `nusc_det/voxelize.py` | 点云栅格化为 5 通道 BEV | `scatter_add_` / `scatter_reduce_` 拼出 per-cell 统计 |
| `nusc_det/targets.py` | CenterPoint 风格目标：Gaussian heatmap + reg map + mask | `gaussian_radius`、`draw_gaussian`、reg 通道排布 |
| `nusc_det/model.py` | 轻量 BEV CNN + heatmap/regression head | head 的 bias=−2.19 是 focal loss 的一个标准 trick |
| `nusc_det/losses.py` | `focal_loss` + `reg_loss` + `detection_loss` | 这些是后面 Lightning 关里**完全不会变**的"研究代码" |
| `exercises/01_pure_pytorch/train.py` | **纯 PyTorch 训练入口** | 训练循环、device 搬运、scheduler、checkpoint —— 这些是后面会被 Lightning 替换的"工程代码" |

---

## 怎么跑

```bash
cd <repo_root>

# 快速 smoke test（前 8 帧，~10 秒，loss 应从 ~7 降到 ~0.6）
python exercises/01_pure_pytorch/train.py --max-frames 8 --epochs 30

# 全量 191 帧（建议 GPU）
python exercises/01_pure_pytorch/train.py --epochs 50
```

主要 CLI 参数：

| flag | 默认值 | 含义 |
|------|--------|------|
| `--data-root` | 你的 J6Gen2 mini 数据路径 | 数据集根目录 |
| `--epochs` | 200 | 训练轮数 |
| `--lr` | 1e-3 | AdamW 初始学习率（cosine 退火到 0） |
| `--max-points` | 60000 | 每帧随机下采样上限（原始 ~24 万） |
| `--max-frames` | -1（全量） | 只用前 N 帧，用来快速 debug |
| `--log-every` | 10 | 多少个 epoch 打一次日志 |
| `--device` | `cuda` | 没 CUDA 自动 fallback CPU |
| `--output-dir` | `runs/01_pure_pytorch` | checkpoint 输出目录 |

期望输出（smoke test）：

```
[data] 8 lidar frames available
  frame 0: points=237643  cars= 20  file=00000.pcd.bin
  ...
[bev] grid: 250 x 250 (voxel_size=0.4m)
[model] BEVDetector  params=0.10M  device=cuda
[epoch    0] loss=7.3015  hm=6.7666  reg=5.3485  lr=9.97e-04  elapsed=1.2s
[epoch   29] loss=0.5853  hm=0.4672  reg=1.1803  lr=0.00e+00  elapsed=9.8s
[done] saved checkpoint -> runs/01_pure_pytorch/final.pt
```

---

## 这一关刻意暴露的"工程层样板"

阅读 `train.py` 时，留意以下 **8 处 boilerplate**——这些就是后面每一关 Lightning 帮你删除/简化的目标：

```python
# ① 手动选择 device，并对 CUDA 不可用做 fallback
if cfg.device == "cuda" and not torch.cuda.is_available():
    cfg = Config(**{**cfg.__dict__, "device": "cpu"})
device = torch.device(cfg.device)

# ② 模型手动 .to(device)
model = BEVDetector(...).to(device)

# ③ 手写训练循环
for epoch in range(cfg.epochs):
    for sample in loader:
        # ④ 每个张量手动 .to(device)
        bev = points_to_bev(points).unsqueeze(0).to(device)
        targets = {k: v.to(device) ... for k, v in targets.items()}

        # ⑤ forward → loss
        outputs = model(bev)
        losses = detection_loss(outputs, targets)

        # ⑥ 手动 zero_grad → backward → grad clip → step
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    # ⑦ 手动调度 scheduler
    scheduler.step()

    # ⑧ 手动 logging（print） 与手动保存 checkpoint
```

---

## 推荐阅读顺序（学习路线）

1. **先跑通**：smoke test 看 loss 下降。
2. **顺着数据流读**：`dataset.py` → `geometry.py` → `voxelize.py` → `targets.py` → `model.py` → `losses.py` → `train.py`。
3. **回看 boilerplate 8 处**：在脑子里给每行打标签——「这是研究代码（模型/loss）」 vs 「这是工程代码（循环/IO/device）」。
4. **想想"如果是你写框架"**：哪些代码每个项目都长得一样？这就是 Lightning 要抽取的部分。

---

## 后续练习路线

| # | 主题 | 这一关相对前一关的"diff" |
|---|------|--------------------------|
| 01 | **纯 PyTorch 训练循环** ← 你在这里 | 全部工程代码暴露 |
| 02 | 重构为 `LightningModule` | 删掉 ③④⑥⑧，把 step 拆成 `training_step` / `configure_optimizers` |
| 03 | 抽出 `LightningDataModule` | 数据逻辑独立成一个类，方便 train/val/test 切换 |
| 04 | `WandbLogger` + `ModelCheckpoint` + `LearningRateMonitor` | 删掉 ⑧ 和手动 print；wandb 上看 loss + BEV heatmap |
| 05 | Trainer flag 一行切 `precision="bf16-mixed"`、`accumulate_grad_batches`、多卡 DDP | 体会"工程优化在框架层免费完成" |
| 06 | 自定义 Callback：每 N epoch dump 预测 BEV heatmap | 理解 Lightning 的 hook 系统 |
| 07 | `automatic_optimization=False` + 多 optimizer（GAN / 双优化器风格） | Lightning 的"逃生通道"：完全手动控制 |

每一关都**只新增**一个 `exercises/0N_xxx/` 目录，**沿用** `nusc_det/` 里同一份模型/loss/数据 —— 这样你能直观看到「研究代码不变、工程代码越来越短」。

---

## 常见问题排查

| 现象 | 可能原因 |
|------|---------|
| `RuntimeError: tensors on cpu and cuda` | 在 `voxelize` 之前就把 points 放到 GPU 了。本练习的 `points_to_bev` 在 CPU 上做 scatter，再把 5×H×W 的小张量送 GPU 即可（这正是 train.py 当前做法）。 |
| heatmap 始终是平的 | 检查 `build_center_targets` 是否把 box 中心落到了 BEV 范围内（`x ∈ [-50, 50]`）。打印 `len(boxes)` 与有多少落在范围内。 |
| reg loss 不下降 | reg 只在 GT 中心格点上算，`reg_mask.sum()` 必须 > 0；如果一个样本里没框，reg loss 会被 mask 成 0（这是预期）。 |
| OOM | 调小 `--max-points`（比如 30000），或缩小 BEV 范围 / 提高 `voxel_size`。 |
