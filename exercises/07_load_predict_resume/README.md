# 练习 07 — Checkpoint：load / predict / resume

## 这一关在做什么

把前几关里 `ModelCheckpoint` 攒下来的 `.ckpt` 文件**真的用起来**：

1. **Load**：从 `.ckpt` 重建 `LitBEVDetector`（不需要记住构造参数）
2. **Predict**：用 `trainer.predict()` 跑推理，把结果存盘
3. **Resume**：训练到一半发现 epoch 不够，从 `last.ckpt` **续训**

这是日常工作里**频率最高**的几个 Lightning API——比 manual_optimization 实用得多。

研究代码（`nusc_det/`）和 Module/DataModule 的**结构**都没变，本关新增的只有：

- Module 里多了 `predict_step`
- 多了一个 `predict.py` 入口
- `train.py` 多了 `--resume-from` 参数

---

## .ckpt 文件里到底装了什么？

`ModelCheckpoint` 写出的 `last.ckpt` / `best-*.ckpt` 是一个 `torch.save` 的 dict，里面大致有：

| key | 内容 |
|-----|------|
| `state_dict` | **模型权重**（`LitBEVDetector` 所有参数和 buffer） |
| `optimizer_states` | AdamW 的动量、二阶矩等 |
| `lr_schedulers` | CosineAnnealing 的内部 step 计数 |
| `epoch`, `global_step` | 当前训练到第几个 epoch / 总共多少 step |
| `hyper_parameters` | 你 `save_hyperparameters()` 传的所有 `__init__` 参数 |
| `pytorch-lightning_version` | 框架版本（兼容性提示） |
| 各 callback 的 `state_dict` | 比如 `ModelCheckpoint` 记得「目前 best 是 0.12」 |

**两种 API 对应两种"用 ckpt 的方式"**，区别就在用了里面**哪些** key：

| API | 用到的 key | 适合场景 |
|-----|-----------|----------|
| `LitBEVDetector.load_from_checkpoint(p)` | `state_dict` + `hyper_parameters` | 推理 / 微调（fresh optimizer） |
| `trainer.fit(model, ckpt_path=p)` | **全部** | **续训**（不要重新走 epoch 0） |

---

## 1. Load + Predict

`Module.__init__` 里调用 `self.save_hyperparameters()` 是这一关的回报兑现点——它把所有构造参数都塞进了 `.ckpt`，所以这样就能直接重建：

```python
model = LitBEVDetector.load_from_checkpoint("runs/.../best-029-val_loss=0.12.ckpt")
# 自动从 ckpt 里读 base_channels / num_classes / lr ...，不必再传一遍
```

之后跑推理两种写法：

### 写法 A（推荐，Lightning 风格）

需要 Module 里有 `predict_step`（本关 `_lit.py` 已加）：

```python
def predict_step(self, batch, batch_idx, dataloader_idx=0):
    outputs = self(batch["bev"])
    return {"pred_heatmap": outputs["heatmap"].sigmoid().cpu(), ...}
```

然后：

```python
trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
outputs = trainer.predict(model, datamodule=datamodule)
# outputs 是 list[dict]，每个 dict 是 predict_step 的返回值
```

`trainer.predict` 帮你做了 4 件事：`model.eval()` / `torch.no_grad()` / 迭代 loader / 收集 per-batch 输出。

### 写法 B（纯 PyTorch，等价）

```python
model.eval()
with torch.no_grad():
    for batch in loader:
        out = model(batch["bev"])
```

新组的乱代码里两种都常见，知道**它们等价**就够了。

---

## 2. Resume Training（你最想要的场景）

> 训练设了 `--epochs 50`，跑完发现还能再降，想接着训到 100。

ex07 的 `train.py` 加了一个参数 `--resume-from`，关键就是一行：

```python
trainer.fit(model, datamodule=datamodule, ckpt_path=str(resume_from))
```

Lightning 收到 `ckpt_path` 时会**全量恢复**：

- 模型权重
- AdamW optimizer 状态（**动量恢复**，所以续训不会"破坏"已有的学习路径）
- CosineAnnealing scheduler 的内部步数（学习率从对应位置继续衰减，不会 reset）
- `current_epoch` / `global_step`
- `ModelCheckpoint` 记得历史 best，不会把新 epoch 误判成"突然变好了"

### 常见踩坑

| 现象 | 原因 |
|------|------|
| 续训立刻退出 | 忘了把 `--epochs` 调大到比 `last.ckpt.epoch` 还大（例如还是 50） |
| LR 突然从头开始衰减 | 用了 `load_from_checkpoint` 后又 `trainer.fit`（这会用 fresh optimizer/scheduler，**不是续训**） |
| best ckpt 名字 epoch 倒退 | `--output-dir` 改了；新的 `ModelCheckpoint` 不知道历史 best |
| wandb 起了新 run | 没指定 `--wandb-run-name`（这是 wandb 的"特性"，不是 ckpt 的） |

---

## 怎么跑（推荐流程）

```bash
# Step 1: 先训 50 epoch
python exercises/07_load_predict_resume/train.py --max-frames 8 --epochs 50

# Step 2: 看 last.ckpt 在哪
ls runs/07_load_predict_resume/checkpoints/

# Step 3: 跑一次 predict（用 best 也行）
python exercises/07_load_predict_resume/predict.py \
    --ckpt-path runs/07_load_predict_resume/checkpoints/last.ckpt \
    --max-frames 8

# Step 4: 续训到 100 epoch
python exercises/07_load_predict_resume/train.py \
    --max-frames 8 --epochs 100 \
    --resume-from runs/07_load_predict_resume/checkpoints/last.ckpt
```

第 4 步的终端应该看到：

```
[mode] resuming from runs/07_load_predict_resume/checkpoints/last.ckpt
       checkpoint epoch=49  global_step=...
       tip: bump --epochs above the saved epoch, otherwise fit() exits immediately.
```

且 wandb / CSV 里曲线**从 epoch 50 开始**，不是从 0。

---

## 3. Fine-tune（顺带提一下）

如果不是续训，而是**「拿训好的权重，重新开一段训练（新数据 / 新 LR）」**——这时用 `load_from_checkpoint`，然后 **`trainer.fit` 不传 `ckpt_path`**：

```python
model = LitBEVDetector.load_from_checkpoint("best.ckpt", lr=1e-4)  # 重写 lr
trainer = pl.Trainer(max_epochs=20, ...)
trainer.fit(model, datamodule=new_dm)        # fresh optimizer，新 schedule
```

注意 `load_from_checkpoint(path, **overrides)` 可以覆盖 `__init__` 参数——常用于**只换 LR 微调**。

---

## 检查清单（读乱代码时）

看到 ckpt 相关代码，先问自己 3 个问题：

1. **是 `load_from_checkpoint` 还是 `ckpt_path=`？**  
   决定了 optimizer / scheduler / epoch 是否被恢复。
2. **`Module.__init__` 里有没有 `save_hyperparameters()`？**  
   没有的话 `load_from_checkpoint` 必须显式传所有构造参数。
3. **`max_epochs` 改没改？**  
   续训最常见的"什么都没跑就退出"就是因为这个。

---

## 文件清单（这一关）

```
exercises/07_load_predict_resume/
├── _lit.py        # Module / DataModule / Dataset（共享，predict_step 在这里）
├── train.py       # trainer.fit + --resume-from
├── predict.py     # LitBEVDetector.load_from_checkpoint + trainer.predict
└── README.md
```

`train.py` 和 `predict.py` 都 `from _lit import ...`——一旦项目有多个入口，把 Module/DataModule 抽到共享模块是真实项目里**几乎一定**会做的重构。

---

## 关于原计划的 ex07 (manual_optimization)

`automatic_optimization=False` 在你"读代码"的优先级里比 load/predict/resume 低很多——
真用到的项目占少数（GAN / 对抗 / 双优化器）。

如果以后真的撞到，可以单独再做一关（暂记为 ex08）。本关之后，**最值得花时间的是去新组真实代码里跑一遍诊断**，参考之前我们聊到的「第 1/2 周计划」。
