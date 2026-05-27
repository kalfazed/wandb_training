# 练习 02 — 把 ex01 重构成 LightningModule

## 这一关在做什么

**研究代码完全不动**（`nusc_det/` 里的 model / loss / dataset / voxelize / targets 一行都没改），
只是把 ex01 里的"工程层"——训练循环、device 搬运、梯度裁剪、scheduler、checkpoint、logging——
全部换成 **Lightning 的 hook + Trainer**。

这一关的产出是：

- `train.py` 里**没有任何 `for epoch in ...` / `loss.backward()` / `optimizer.step()`**
- 但训出来的 loss 曲线和 ex01 几乎一致
- 你**亲眼看到**：每一段 ex01 boilerplate 对应到 Lightning 的哪个 hook / Trainer flag

> 一句话：**ex02 的价值不是"代码变短了"，而是建立 ex01 ↔ Lightning 的双语翻译表。**
> 这张表是你将来读、调、改任何 Lightning 项目的底层认知。

---

## ex01 → ex02 翻译表

把这张表打印出来贴在屏幕边，比读官方文档高效得多。

| ex01 里写的 | ex02 里去了哪 | 备注 |
|---|---|---|
| ① `if cfg.device == "cuda" and not torch.cuda.is_available(): ...` | `Trainer(accelerator="auto", devices=1)` | Trainer 帮你挑设备 |
| ② `model = BEVDetector(...).to(device)` | `self.detector = BEVDetector(...)`（在 `__init__` 里） | Lightning 在 `Trainer.fit()` 启动时把整个 `LightningModule` 搬到设备 |
| ③ `for epoch in range(...): for sample in loader:` | `trainer.fit(model, train_dataloaders=loader)` | 双层循环整个消失 |
| ④ `bev = ...to(device)` / `{k: v.to(device) ...}` | Lightning 自动调用 `transfer_batch_to_device`；我们用 `on_before_batch_transfer` 做 CPU 端的体素化/target 渲染 | 见下方"hook 执行顺序" |
| ⑤ `outputs = model(bev); losses = detection_loss(...)` | **`training_step` 的函数体**（**唯一保留原样的部分**） | 这就是"研究代码" |
| ⑥ `optimizer.zero_grad(); loss.backward(); clip_grad_norm_(...); optimizer.step()` | Lightning 在 `training_step` 返回 loss 后自动做这四件事；clip 用 `Trainer(gradient_clip_val=10.0)` 一行 | 你**看不到**这些调用，但它们确实发生了 |
| ⑦ `scheduler = CosineAnnealingLR(...); scheduler.step()` | `configure_optimizers` 返回 `{"optimizer": ..., "lr_scheduler": {...}}` | Lightning 按 `interval="epoch"` 自动 step |
| ⑧ `print(...)` + `torch.save({"model": ..., "config": ...}, ckpt_path)` | `self.log(...)` + Trainer 默认存 `last.ckpt`（带 hyperparameters） | 后续 ex04 会把 logger 换成 wandb |

唯一**新增**的（不是从 ex01 翻译来的）：

- `save_hyperparameters(ignore=[...])` —— 把 `__init__` 入参刻进 ckpt，重要、几乎每个 Lightning 项目都用。
- `on_before_batch_transfer` —— 这是 ex02 故意引入的"读乱代码常用 hook"，见下一节。

---

## 一个 batch 内的 hook 执行顺序

读懂这张图，你以后看任何 LightningModule 都知道"业务逻辑会藏在哪几个位置"。

```
DataLoader.__iter__ ──▶ collate_single                    （CPU 上的 sample dict）
                          │
                          ▼
                  on_before_batch_transfer                ← 我们重写了，做体素化 + target
                          │   返回 {"bev": (1,5,H,W), "targets": {...}, "meta": {...}}
                          ▼
                  transfer_batch_to_device                ← Lightning 默认实现，递归把 tensor 搬上 device
                          │
                          ▼
                  on_after_batch_transfer                  ← 默认 no-op，常被乱代码用来做 GPU 端 aug
                          │
                          ▼
                     training_step                        ← 我们重写了，forward + loss + self.log
                          │   返回 loss (标量)
                          ▼
       (auto) optimizer.zero_grad
       (auto) loss.backward
       (auto) gradient clipping            （来自 Trainer(gradient_clip_val=...)）
       (auto) optimizer.step
                          │
                          ▼ 一个 batch 结束

       每个 epoch 末尾，最后一个 batch 之后：
       (auto) lr_scheduler.step()
```

记住一句话：**Lightning 没有黑魔法**，它只是按上面这个顺序调用 hook。乱的代码也只是在某个 hook 里塞了很多东西而已。

---

## 这一关有意引入的 3 个 Lightning idiom

| Idiom | 它解决什么 | 你以后会在乱代码里频繁看到的样子 |
|---|---|---|
| `self.save_hyperparameters(ignore=[...])` | 把 `__init__` 的入参刻到 `self.hparams` + checkpoint | `self.hparams.lr`, `self.hparams.num_classes` 这种用法满天飞 |
| `self.log("...", value, on_step=..., on_epoch=..., prog_bar=...)` | 取代 print / 手写 logger.log | 同一个名字在不同 hook 里 log 多次时要小心聚合行为 |
| `on_before_batch_transfer` / `on_after_batch_transfer` | 在自动 device 搬运 前/后 改 batch | 数据增强、CPU 端 prep、目标渲染常藏在这里 |

如果以后调试时找不到"输入怎么变成这个样子的"，第一件事**先搜这两个 hook**，再搜 `transfer_batch_to_device`。

---

## 怎么跑

```bash
pip install -r requirements.txt  # 新增了 lightning>=2.0

# 快速 smoke test（与 ex01 同一组参数，方便对比 loss）
python exercises/02_lightning_module/train.py --max-frames 8 --epochs 30

# 全量
python exercises/02_lightning_module/train.py --epochs 50
```

期望输出（smoke test，量级应与 ex01 几乎一致）：

```
[data] 8 lidar frames available
  frame 0: points=237643  cars= 20  file=00000.pcd.bin
  ...
[bev] grid: 250 x 250 (voxel_size=0.4m)
[model] LitBEVDetector  params=0.10M
GPU available: True (cuda), used: True
...
Epoch 29: 100%|███████| 8/8 [00:00<00:00, ... train/loss=0.59]
[done] checkpoints + metrics under: runs/02_lightning_module
```

跑完后看一眼 `runs/02_lightning_module/`：

```
runs/02_lightning_module/
├── csv/version_0/metrics.csv       ← self.log(...) 的累积结果
└── lightning_logs/version_0/checkpoints/last.ckpt   ← 自动保存
```

`metrics.csv` 用 `pandas.read_csv` / `tail` 都能直接看 loss 曲线，不需要任何额外工具。

---

## 推荐对照阅读顺序

1. **先把 `exercises/01_pure_pytorch/train.py` 和本目录 `train.py` 并排打开**
2. 在 ex01 里找 README 列出的 8 处 boilerplate，**用上面的翻译表**逐一确认它们去了哪
3. 重点盯三个函数：
   - `LitBEVDetector.training_step` —— 这就是研究代码（=ex01 的 forward+loss 段）
   - `LitBEVDetector.configure_optimizers` —— 优化器 + scheduler 的标准声明方式
   - `LitBEVDetector.on_before_batch_transfer` —— 业务逻辑藏在 hook 里的典型样本
4. 然后看 `main()` 末尾的 `pl.Trainer(...)` 构造：**每一个 flag 都对应 ex01 的一段代码**

---

## 给"读乱 Lightning 代码"的人 3 条小建议（提前剧透）

将来你接手别人 5000 行的 LightningModule 时，按这个顺序看最快：

1. **先看 `configure_optimizers`**：能告诉你 optimizer 个数、scheduler 类型——一眼判断是不是 `automatic_optimization=False`（ex07 的主题）
2. **再看所有以 `on_` 开头的方法**：业务逻辑大多藏在这里，**不**在 `training_step`
3. **最后看 Trainer 的构造**：`precision`, `accumulate_grad_batches`, `strategy`, `callbacks=[...]`——这些 flag 会改变上面 hook 图里的"auto"部分

ex04（callbacks + wandb）和 ex05（Trainer flags）会把第 2、3 步展开讲。

---

## 这一关刻意**没做**的事（避免一关塞太多）

| 没做 | 留到哪一关 |
|---|---|
| `validation_step` / val DataLoader | ex03 (`LightningDataModule` 一起做) |
| 把数据集逻辑抽出 `train.py` | ex03 |
| wandb / ModelCheckpoint(monitor=...) / LR monitor | ex04 |
| 多卡 / AMP / 梯度累积 | ex05 |
| 自定义 callback dump heatmap | ex06 |
| `automatic_optimization=False` | ex07 |

这一关的唯一焦点：**让你把 ex01 的训练循环 1:1 翻译进 LightningModule**。其它都不是这一关该担心的。
