# 练习 05 — Trainer flags（混合精度 / 梯度累积 / 多卡）

## 这一关在做什么

ex04 的 diff 在 **callbacks / loggers**。  
ex05 的 diff 只在 **`pl.Trainer(...)` 的关键字参数**——Module、DataModule、loss **一行不改**。

你要建立的对照：

| 你想做的事 | ex01 纯 PyTorch | ex05 Lightning |
|-----------|----------------|----------------|
| 混合精度 | 手写 `autocast` + `GradScaler` | `precision="bf16-mixed"` |
| 模拟大 batch | 手动 N 次 `backward` 再一次 `step` | `accumulate_grad_batches=N` |
| 多卡训练 | `DistributedDataParallel` + `DistributedSampler` | `devices=K`, `strategy="ddp"` |

> 读乱代码时：先搜 `Trainer(`，把 `precision` / `accumulate_grad_batches` / `devices` / `strategy` 抄到纸上，再往下看 Module。

---

## ex04 → ex05：Trainer 多了什么

```python
trainer = pl.Trainer(
    ...
    precision=resolve_precision(cfg.precision),       # NEW
    accumulate_grad_batches=cfg.accumulate_grad_batches,  # NEW
    devices=cfg.devices,                              # was hard-coded 1
    strategy="ddp" if cfg.devices > 1 else "auto",    # NEW
)
```

启动时会打印 `[trainer flags]` 横幅，包含 **effective batch** 估算：

```
effective batch ≈ batch_size × num_GPUs × accumulate_grad_batches
```

---

## 三个 flag 分别干什么

### 1. `precision`

- `"32-true"`：全 FP32（CPU 或调试时常用）
- `"16-mixed"`：AMP，多数算子 FP16，部分 FP32（省显存、常更快）
- `"bf16-mixed"`：AMP with bfloat16（A100/H100/部分 RTX 等）
- `"auto"`（本练习默认）：有 CUDA 且支持 bf16 → `bf16-mixed`，否则 `16-mixed`，无 CUDA → `32-true`

**Module 里不需要** `with torch.cuda.amp.autocast():` —— Trainer 在 `training_step` 外包好了。

不支持时会 **打印 warn 并 fallback**（见 `resolve_precision()`）。

### 2. `accumulate_grad_batches`

例如 `accumulate_grad_batches=4`：

```
training_step (micro 1) -> backward，梯度累积
training_step (micro 2) -> backward，累积
training_step (micro 3) -> backward，累积
training_step (micro 4) -> backward，累积 -> optimizer.step() + zero_grad
```

等价于「**物理 batch 很小，但优化器眼里 batch 大了 4 倍**」。

注意：

- `self.log(..., on_step=True)` 仍是 **每个 micro-batch** 记一次（进度条会跳得快）
- `on_epoch=True` 的指标仍是 **按 epoch 聚合**
- 学习率 **不会** 因你开了 accum 自动放大；大 effective batch 时常用 **linear LR scaling**（本练习只提示，不自动改）

### 3. `devices` + `strategy`

- `devices=1`：`strategy="auto"`（单卡）
- `devices=2`（或更多）：`strategy="ddp"`（Distributed Data Parallel）

Module / DataModule **不用写 DDP 包装**；`validation_step` 里已有的 `sync_dist=True` 在多卡 val 时会把各卡的 `val/loss` 聚合成一个数。

本数据集很小，多卡 **不一定更快**（通信开销），目的是让你认得 flag，不是刷速度。

---

## 推荐实验命令

```bash
# 基线（与 ex04 类似，默认 auto precision）
python exercises/05_trainer_flags/train.py --max-frames 8 --epochs 30 --no-wandb

# 混合精度（显式指定）
python exercises/05_trainer_flags/train.py --max-frames 8 --epochs 30 --precision bf16-mixed

# 梯度累积：有效 batch = 1 × 1 GPU × 4 = 4
python exercises/05_trainer_flags/train.py --max-frames 8 --epochs 30 --accumulate-grad-batches 4

# 多卡（需要至少 2 张 GPU）
python exercises/05_trainer_flags/train.py --max-frames 8 --epochs 30 --devices 2
```

对比实验时固定 `--wandb-run-name` 或看本地 `runs/05_trainer_flags/csv/` 里的 `metrics.csv`。

---

## 读别人项目时的 Trainer 检查清单

1. **`precision`**：是否 mixed？Module 里是否还有重复的 `autocast`（双重包一层会难 debug）？
2. **`accumulate_grad_batches`**：optimizer step 频率 = 1/accum；日志 `on_step` 含义是否被误解？
3. **`devices` / `strategy`**：多卡是 `ddp` 还是 `fsdp` / `deepspeed`？
4. **`gradient_clip_val`**：在 Trainer 上还是在 `on_before_optimizer_step` 里又 clip 了一次？
5. **`limit_train_batches` / `max_steps`**：是否只跑了子集（ smoke / debug ）？

---

## 这一关没做的（留给 ex06 / ex07）

| 主题 | 练习 |
|------|------|
| 自定义 Callback 批量存 heatmap 到磁盘 | ex06 |
| `automatic_optimization=False`、多 optimizer | ex07 |
| DeepSpeed / FSDP | 超出本教程范围 |

---

## 和 ex01 的「体感」对照

跑完 ex05，回头看 ex01 的 `train_one_epoch`：

- 若团队脚本里 **没有** 这些 Trainer flag，却 **在 Module 里手写** AMP / DDP → 等价逻辑，只是分散了
- 若 **Trainer 已经设了 `precision="bf16-mixed"`**，Module 里 **还有** `autocast` → 可能是历史遗留，读代码时要怀疑「是否重复」

这就是 ex05 想让你带走的：**工程优化常常不在 `training_step` 里，而在 `Trainer(` 那一行。**
