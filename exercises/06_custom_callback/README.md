# 练习 06 — 自定义 Callback：把 BEV heatmap 存盘

## 这一关在做什么

ex04 用 **十几行** 的 `LogBEVHeatmapCallback` 往 wandb 传 **一张** 图。  
ex06 换成 **`DumpBEVHeatmapCallback`** —— 更接近真实项目里的可视化 callback：

- 在 **`on_validation_epoch_end`** 跑（整个 val epoch 的 metric 已经算完）
- 默认遍历 **整个 val DataLoader**（`--val-frames 2` 时会存 2 张）
- 每个样本写 3 个 PNG：`pred` / `gt` / `panel`（左 pred 右 gt）
- **DDP 只在 rank 0 写盘**（`trainer.is_global_zero`）

Module 的 `validation_step` **仍然只算 loss**，不负责画图。

---

## ex04 vs ex06

| | ex04 `LogBEVHeatmapCallback` | ex06 `DumpBEVHeatmapCallback` |
|---|---|---|
| 输出 | wandb Media | 本地 `visualizations/epoch_XXX/*.png` |
| 覆盖范围 | val 的第 1 个 batch | 默认 **全部** val batch |
| 文件数 | 1 张 panel / 次 | 每帧 3 张 PNG |
| 依赖 | 必须 wandb | 只需 matplotlib（`--no-wandb` 也行） |
| 结构 | 全在 `on_validation_epoch_end` 里 | 拆成 `_should_run` / `_dump_batch` / `_save_heatmap_png` |

---

## 目录结构（跑完后）

```
runs/06_custom_callback/
├── checkpoints/
│   ├── best-029-val_loss=0.12.ckpt
│   └── last.ckpt
├── csv/...
└── visualizations/
    ├── epoch_004/
    │   ├── 000_00006_pred.png
    │   ├── 000_00006_gt.png
    │   ├── 000_00006_panel.png
    │   ├── 001_00007_pred.png
    │   └── ...
    └── epoch_029/
        └── ...
```

文件名：`{val_batch_idx}_{pcd_stem}_{pred|gt|panel}.png`

---

## 为什么用 Callback 而不是写在 `validation_step` 里？

1. **单一职责**：`validation_step` = 指标；Callback = 副作用（I/O、画图）。
2. **可插拔**：从 `callbacks=[...]` 列表里删掉这一类，训练逻辑不变。
3. **读乱代码**：团队常把 500 行可视化塞进 `callbacks/foo_vis.py`，搜 `on_validation_epoch_end` 就能定位。

---

## Callback hook 顺序（和 ex04 相同位置，更重的工作）

```
on_validation_epoch_start
  for each val batch:
    on_validation_batch_start
    validation_step()          # Module：只 log val/loss
    on_validation_batch_end
on_validation_epoch_end        # DumpBEVHeatmapCallback：
                               #   再跑一遍 val_loader 做 forward + 存 PNG
```

注意：这里会在 epoch 末 **额外做一遍 val forward**（只为存图）。  
生产里有时改成在 `validation_step` 里缓存最后一 batch 的 tensor，避免双遍 val —— 那是性能优化，本练习优先 **清晰**。

---

## 怎么跑

```bash
pip install -r requirements.txt   # 含 matplotlib

# smoke：8 帧 (6 train + 2 val)，每 5 epoch 存一次 val 图
python exercises/06_custom_callback/train.py --max-frames 8 --epochs 30 --dump-every 5

# 只要 ex04 那种「只存 val 第一个 batch」
python exercises/06_custom_callback/train.py --max-frames 8 --epochs 30 --dump-first-val-batch-only

# 不连 wandb，只留 PNG + CSV + checkpoint
python exercises/06_custom_callback/train.py --max-frames 8 --epochs 30 --no-wandb
```

终端应出现：

```
[DumpBEVHeatmapCallback] epoch=4 saved 2 frame(s) -> runs/06_custom_callback/visualizations/epoch_004
```

用系统看图工具打开 `*_panel.png`：左 pred、右 gt。

---

## 读 `DumpBEVHeatmapCallback` 的建议顺序

1. `_should_run` —— 何时跳过（sanity check、DDP 非 0 号进程、epoch 间隔）
2. `on_validation_epoch_end` —— 总控：建目录、遍历 loader
3. `_dump_batch` —— 单 batch：eval forward、拆样本、调存盘
4. `_save_heatmap_png` —— 纯函数，matplotlib Agg，无 GUI

---

## DDP 提醒（呼应 ex05）

`trainer.is_global_zero` 为 False 的 rank **直接 return**。  
若乱代码里每个 rank 都在写同一路径，会导致文件损坏或互相覆盖 —— 这是 review 时很值钱的 bug。

---

## 下一关 ex07

`automatic_optimization=False` + 多 optimizer（GAN / 双优化器风格）—— Lightning 的「手动训练」逃生通道。
