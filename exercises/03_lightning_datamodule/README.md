# 练习 03 — 抽出 LightningDataModule + wrapper Dataset

## 这一关在做什么

把 ex02 里**数据相关**的逻辑全部从 `LightningModule` / `train.py` 里搬走，
放进两个**职责单一**的对象：

1. **`NuScenesBEVDataset`**（wrapper Dataset）—— `__getitem__` 里完成体素化 + target 渲染
2. **`LitBEVDataModule`** —— `setup()` 里建 Dataset 并做 train/val split；`*_dataloader()` 返回 loader

Module 本身（`LitBEVDetector`）变成**真正的研究代码容器**：只剩 `forward + loss + log`，
**删掉**了 `on_before_batch_transfer`，**新增**了 `validation_step`。

研究代码（`nusc_det/`）依然一行没动。

> 一句话：**ex02 让你认识 Lightning 的 hook；ex03 让你把数据 hook 放回它本来该在的位置。**

---

## ex02 → ex03 的 diff（按职责重排）

### 数据管线下沉

| ex02 在哪做 | ex03 在哪做 | 为什么这样更好 |
|---|---|---|
| `LitBEVDetector.on_before_batch_transfer` 里 subsample + `points_to_bev` + `build_center_targets` | `NuScenesBEVDataset.__getitem__` | 跟模型权重无关的固定 pipeline，本来就属于"数据"那一侧 |
| `train.py` 直接 `NuScenesLidarDetDataset(...)` + `DataLoader(...)` | `LitBEVDataModule.setup()` + `LitBEVDataModule.train_dataloader()` | 路径 / split / num_workers / collate **一个文件答完所有"数据从哪来"的问题** |
| 没有 val | `setup()` 切出 `val_frames`，`val_dataloader()` 返回；Module 加 `validation_step` | 训练曲线和泛化曲线分离，后续 `ModelCheckpoint(monitor="val/loss")` 才有意义 |
| `batch_size=1` 时用 `collate_single` 直接返回单条 sample dict | `collate_bev_batch` 真正按 batch 维 stack（哪怕 bs=1 也走标准路径） | 以后 bs>1 / 多卡 / `meta` 字段处理都不必再改 collate |
| Module hold 着 `max_points` 这个 hyperparameter | DataModule 持有 `max_points`（它是数据 pipeline 的参数） | 哪些 hyperparameter 属于谁，从此清晰 |

### Module 瘦身

`LitBEVDetector` 的变化：

```
 ex02                                    ex03
 ----                                    ----
- on_before_batch_transfer(...)          (删除：搬进 Dataset)
- self.bev_cfg / max_points             (删除：DataModule 管)
  training_step(...)                    + _shared_step(...) (抽取共用)
                                        + validation_step(...)  (新增)
```

Module 现在读起来非常直白：

```python
def training_step(self, batch, batch_idx):
    losses = self._shared_step(batch)
    self.log("train/loss", losses["loss"], ...)
    return losses["loss"]

def validation_step(self, batch, batch_idx):
    losses = self._shared_step(batch)
    self.log("val/loss", losses["loss"], ...)
```

这才是 Lightning 项目里 Module **应该**有的样子。

---

## LightningDataModule 的 5 个标准 hook

这张表你以后看任何 DataModule 都用得上：

| hook | 调用时机 | 你通常该在这里做什么 |
|---|---|---|
| `__init__` | 构造时 | 记下路径 / 超参；**不**做 I/O，**不**建 Dataset |
| `prepare_data()` | 在 rank 0 上调用**一次** | 下载 / 解压 / 全局生成索引；**不要**给 `self` 赋值（DDP 下其它 rank 看不到） |
| `setup(stage)` | 每个 rank 都会被调一次；`stage` ∈ `{"fit","validate","test","predict"}` | 建 Dataset、做 split、`self._train_ds = ...` |
| `train_dataloader() / val_dataloader() / test_dataloader() / predict_dataloader()` | `Trainer.fit / .validate / .test / .predict` 时 | 返回对应的 `DataLoader` |
| `teardown(stage)` | 阶段结束 | 关文件、释放资源；通常是 no-op |

在 ex03 里我们只用了 `__init__` / `setup` / `train_dataloader` / `val_dataloader`，
`prepare_data` 留了个空实现**故意**让你看见它存在。

---

## "一个 batch 的 hook 顺序" 在 ex03 里变成了什么

对比 ex02 的那张图，**Module 那侧的两个 transfer hook 不再被我们重写**——
voxelize/targets 现在已经在 Dataset 里跑完了：

```
LitBEVDataModule.train_dataloader()
        │
        ▼
DataLoader 迭代 NuScenesBEVDataset.__getitem__   ← voxelize + build_center_targets 在这里
        │  (每个 worker 进程并行跑)
        ▼
collate_bev_batch                                ← stack 出 (B, 5, H, W) 等
        │
        ▼
(default) transfer_batch_to_device               ← Lightning 自动把所有 tensor .to(device)
        │
        ▼
LitBEVDetector.training_step                     ← forward + loss + log
        │
        ▼
 (auto) zero_grad / backward / clip / step
```

val 那条路径只是把最后一格换成 `validation_step`，且**没有** backward/step——
这两件事 Lightning 帮你处理，你不用写 `with torch.no_grad():` 或 `model.eval()`。

---

## 一个值得记住的取舍

把数据 pipeline 放在 `Dataset.__getitem__` vs 放在 `on_before_batch_transfer`：

|  | Dataset.\_\_getitem\_\_ | Module hook |
|---|---|---|
| 并行（多 worker） | ✅ 每个 worker 都跑 | ❌ 只在主进程跑 |
| 可以依赖训练状态（如 `self.current_epoch`） | ❌ Dataset 不知道训练进度 | ✅ `self.trainer.current_epoch` 可用 |
| 多卡 / DDP 复杂度 | 简单（loader 各自跑） | 简单但要小心 hook 触发时机 |
| Module 行数 | 短 | 长 |
| 复用到 val/test/predict | ✅ 直接复用 | 需要再写一次或写条件 |

**经验法则：固定 pipeline 放 Dataset；和当前训练状态有关的 transform 放 Module hook。**

这正是 ex02→ex03 的核心：体素化和 target 渲染都是**固定的**，所以下沉到 Dataset 最自然。

---

## 怎么跑

```bash
# smoke test：8 帧（6 train + 2 val），30 epoch
python exercises/03_lightning_datamodule/train.py --max-frames 8 --epochs 30

# 全量
python exercises/03_lightning_datamodule/train.py --epochs 50

# 想关掉 val
python exercises/03_lightning_datamodule/train.py --max-frames 8 --val-frames 0 --epochs 30
```

主要 CLI 参数（**新**字段加了 ✨）：

| flag | 默认 | 含义 |
|---|---|---|
| `--data-root` | J6Gen2 mini 路径 | 数据根目录 |
| `--epochs` | 200 | 训练轮数 |
| `--lr` | 1e-3 | AdamW 初始学习率 |
| `--max-points` | 60000 | 每帧 subsample 上限（**DataModule** 控制） |
| `--max-frames` | -1（全量） | 只用前 N 帧 |
| `--val-frames` ✨ | 2 | 末尾 N 帧做 val |
| `--num-workers` ✨ | 0 | DataLoader 多 worker（pipeline 已下沉到 Dataset，这里调>0 真正有用） |
| `--output-dir` | `runs/03_lightning_datamodule` | 输出目录 |

期望输出（smoke test）：

```
[data] train_frames=6 val_frames=2
[model] LitBEVDetector  params=0.10M
[bev] grid: 250 x 250 (voxel_size=0.4m)
...
Epoch 29: ... train/loss=0.59 val/loss=...
[done] checkpoints + metrics under: runs/03_lightning_datamodule
```

`runs/03_lightning_datamodule/csv/version_0/metrics.csv` 里会同时有 `train/loss*` 和 `val/loss*` 两列家族。

---

## 推荐对照阅读顺序

1. **三窗对照**：`01/train.py`（baseline）、`02/train.py`（hook 在 Module）、`03/train.py`（拆开了）
2. 关注 ex03 里这 4 段：
   - `NuScenesBEVDataset.__getitem__` —— ex02 的 hook 内容**整段搬下来**
   - `LitBEVDataModule.setup` —— 注意 split 怎么做的、为什么用 `__new__` 复制基础 dataset
   - `LitBEVDetector.training_step` / `validation_step` —— 共用 `_shared_step`，避免 train/val 行为漂移
   - `trainer.fit(model, datamodule=datamodule)` —— 入口签名变了
3. 跑一次 smoke test，看 `metrics.csv` 里 train/val 两条曲线是否合理（数据量太小，val 抖动是预期的）

---

## 给"读乱 Lightning 代码"的人：DataModule 视角的 3 个常见坑

提前知道这几条，进新组看代码会少踩坑：

1. **`prepare_data` 里给 `self.xxx` 赋值** —— DDP 下其它 rank 看不到。建 Dataset 一定要在 `setup` 里。
2. **Dataset 里使用 `self.trainer.current_epoch`** —— `Dataset` 不持有 trainer。需要 epoch 相关随机性时，要么算法层面用 `worker_init_fn`，要么搬回 `on_before_batch_transfer`。
3. **多个 `*_dataloader` 返回结构不一致** —— 例如 train 返回单个 loader，val 返回 list[DataLoader]，会让 `validation_step` 多一个 `dataloader_idx` 参数。看到该参数没用上时，**注意是不是有人正在准备多个 val loader**。

---

## 这一关刻意没做的

| 没做 | 留到哪一关 |
|---|---|
| `ModelCheckpoint(monitor="val/loss")` 按 val 存 best | ex04 |
| `WandbLogger` + LR monitor | ex04 |
| `precision="bf16-mixed"` / 多卡 / `accumulate_grad_batches` | ex05 |
| 在 val 上 dump 一张 BEV heatmap 图 | ex06（用自定义 callback 实现） |
| `automatic_optimization=False` | ex07 |

这一关只关心**职责划分**——Module 干什么、DataModule 干什么、Dataset 干什么。
其它都是后面 4 关的事。
