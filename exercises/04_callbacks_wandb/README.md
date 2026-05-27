# 练习 04 — Callbacks + WandbLogger

## 这一关在做什么

ex03 已经把 **Module / DataModule / Dataset** 分工理顺了。  
ex04 在 **Trainer 外围**加一层「观测与持久化」——你不需要改 loss、不需要改数据管线：

| 新增 | 类型 | 作用 |
|---|---|---|
| `WandbLogger` | Logger | `self.log(...)` 自动同步到 wandb UI |
| `ModelCheckpoint` | 内置 Callback | 按 `val/loss` 存 `best-*.ckpt` + `last.ckpt` |
| `LearningRateMonitor` | 内置 Callback | 自动记录 LR 曲线（可删 Module 里手 log 的 `lr`） |
| `LogBEVHeatmapCallback` | **自定义** Callback | 每 N epoch 向 wandb 上传 pred/GT heatmap 拼图 |

研究代码（`nusc_det/`）依然不动。

> **读乱代码时的搜索关键词**：`callbacks=[`、`WandbLogger`、`class .*Callback`。

---

## ex03 → ex04 diff（只有 Trainer 外围变了）

```python
# ex03
trainer = pl.Trainer(
    logger=CSVLogger(...),
    enable_checkpointing=True,   # 默认 last.ckpt，不按 val 选 best
)

# ex04
trainer = pl.Trainer(
    logger=[CSVLogger(...), WandbLogger(...)],   # 可多个 logger
    callbacks=[
        ModelCheckpoint(monitor="val/loss", ...),
        LearningRateMonitor(logging_interval="epoch"),
        LogBEVHeatmapCallback(...),
    ],
)
```

Module 的 `training_step` / `validation_step` **几乎没变**——这就是 Lightning 的设计：  
**指标在 Module 里 `self.log`，存盘/上传在 Callback/Logger 里配置。**

---

## Callback 是什么？和 Module hook 有什么区别？

| | **LightningModule** hook | **Callback** |
|---|---|---|
| 例子 | `training_step`, `validation_step` | `ModelCheckpoint`, `EarlyStopping`, 自定义 `LogBEVHeatmapCallback` |
| 谁实现 | 你的模型类 | Lightning 内置或你写的 `pl.Callback` 子类 |
| 典型用途 | forward、loss、优化相关 | checkpoint、早停、可视化、打印、改学习率策略 |
| 在乱代码里 | 很长时先找 `*_step` | 很长时搜 `callbacks=` 和 `on_train_*` / `on_validation_*` |

一个 val epoch 的简化顺序（**多个 callback 都挂在这张图上**）：

```
on_validation_epoch_start          ← Callback 可以 hook
  for batch:
    on_validation_batch_start
    validation_step()              ← Module（算 loss + self.log）
    on_validation_batch_end
on_validation_epoch_end            ← LogBEVHeatmapCallback 在这里上传图片
```

`ModelCheckpoint` 在 epoch 末根据已 log 的 `val/loss` 决定是否覆盖 `best-*.ckpt`。

---

## 三个内置组件怎么用

### 1. `WandbLogger`

Module 里已有的：

```python
self.log("train/loss", losses["loss"], ...)
self.log("val/loss", losses["loss"], ...)
```

只要 Trainer 挂了 `WandbLogger`，这些会自动出现在 wandb 的 Charts 里，**不必**手写 `wandb.log({"train/loss": ...})`（自定义图除外）。

首次使用：

```bash
pip install wandb
wandb login          # 一次性
```

离线/无账号时：

```bash
export WANDB_MODE=offline
python exercises/04_callbacks_wandb/train.py ...
# 或不用 wandb：
python exercises/04_callbacks_wandb/train.py --no-wandb
```

### 2. `ModelCheckpoint`

```python
ModelCheckpoint(
    monitor="val/loss",    # 必须和 self.log 的名字完全一致
    mode="min",
    save_top_k=1,
    save_last=True,
    dirpath="runs/04_callbacks_wandb/checkpoints",
)
```

**ex03 的 `val/loss` 就是为这个准备的。** 没有 val 时脚本会 fallback 到 `train/loss` 并打印 warn。

训练结束后：

```
runs/04_callbacks_wandb/checkpoints/
├── best-029-val_loss=0.1234.ckpt
└── last.ckpt
```

恢复训练（了解即可，本练习未演示）：

```python
trainer.fit(model, datamodule=dm, ckpt_path=".../best-xxx.ckpt")
```

### 3. `LearningRateMonitor`

自动把 optimizer 的 LR 记到 logger。ex04 里我们从 `training_step` **删掉了**手动的 `self.log("lr", ...)`，避免和 monitor 重复。

---

## 自定义 Callback：`LogBEVHeatmapCallback`

这是 ex06「大改版可视化」的**预告版**：

- 在 `on_validation_epoch_end` 取 val 的第一个 batch
- `sigmoid(pred heatmap)` 与 GT 横向拼接
- `wandb.Image(...)` 上传

**为什么放在 Callback 而不是 `validation_step`？**

- `validation_step` 应保持「算指标」单一职责
- 可视化是**可选副作用**，换工具（存盘 / tensorboard）只改 Callback
- 团队乱代码里，**超大 `validation_step` 后面往往跟着该抽出来的 Callback**

ex06 会改成：按 epoch 存 PNG 到磁盘、可能 dump 多张、不依赖 wandb。

---

## 怎么跑

```bash
pip install -r requirements.txt   # 含 wandb

# smoke test + wandb（推荐）
python exercises/04_callbacks_wandb/train.py --max-frames 8 --epochs 30 --heatmap-every 5

# 不用 wandb（只留 CSV + checkpoint）
python exercises/04_callbacks_wandb/train.py --max-frames 8 --epochs 30 --no-wandb

# 自定义 run 名（方便在 UI 里找）
python exercises/04_callbacks_wandb/train.py --wandb-run-name ex04-smoke-8f
```

wandb UI 里应看到：

- **Charts**：`train/loss`, `val/loss`, `train/loss_hm`, …, `lr-AdamW`（来自 LR monitor）
- **Media**（若未 `--no-wandb`）：`val/bev_heatmap`，左 pred 右 GT

本地仍有：

- `runs/04_callbacks_wandb/csv/version_*/metrics.csv`
- `runs/04_callbacks_wandb/checkpoints/best-*.ckpt`

---

## 读别人项目时的检查清单

1. `Trainer(..., callbacks=[...])` —— 有哪些内置/自定义 callback？
2. `monitor="???"` 是否和 Module 里 `self.log("???", ...)` **字符串完全一致**？（`val/loss` vs `val_loss` 写错 = checkpoint 永远不更新）
3. Logger 是 `WandbLogger` 还是 `TensorBoardLogger` 还是多个？
4. 自定义 `on_validation_epoch_end` 里有没有 **额外 forward**（和 `validation_step` 重复算两遍）？
5. `self.log(..., on_step=?, on_epoch=?)` —— 曲线是 per-step 还是 per-epoch 聚合？

---

## 这一关刻意没做的（留给后面）

| 主题 | 练习 |
|---|---|
| `precision="bf16-mixed"` / DDP / `accumulate_grad_batches` | ex05 |
| 大批量 dump heatmap 到磁盘、复杂 callback 类 | ex06 |
| `automatic_optimization=False` | ex07 |

---

## 和 ex03 的小改进（顺带）

- **val Dataset** 使用 `random_subsample=False`，val 更可复现（train 仍随机下采样）
- 无 wandb 时 **CSV + checkpoint 仍完整工作**，不强制联网
