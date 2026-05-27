# 练习 08 — Hydra + OmegaConf

## 这一关在做什么

把 ex07 的 `train.py` / `predict.py` 改写成 **Hydra 驱动**：
YAML 描述 Trainer / Module / DataModule / callbacks / logger，
**Python 主程序只负责"实例化"和"开跑"**。

研究代码（`nusc_det/`）和 `_lit.py` 一行不改——这一关的 diff **全部在 `conf/` 目录和入口脚本**。

> 一句话：**Hydra 教你把"训练超参"和"训练逻辑"彻底解耦。**

读懂这一关之后，你会发现新组那些"很乱很长"的代码很多其实只是 yaml 多，**Python 主体往往只有 30 行**。

---

## 目录布局

```
exercises/08_hydra_omegaconf/
├── _lit.py              # Module / DataModule / Dataset（=ex07，不变）
├── train.py             # @hydra.main -> instantiate -> trainer.fit
├── predict.py           # @hydra.main -> load_from_checkpoint -> trainer.predict
├── conf/
│   ├── config.yaml      # 训练入口（defaults + seed + ckpt_path）
│   ├── predict.yaml     # 预测入口（ckpt_path: ???）
│   ├── bev/default.yaml
│   ├── model/{default,big}.yaml
│   ├── data/{smoke,full}.yaml
│   ├── trainer/{default,bf16}.yaml
│   ├── callbacks/default.yaml
│   └── logger/{csv_only,wandb}.yaml
└── README.md (本文件)
```

每个子目录就是一个 **config group**。

---

## 5 个必须认得的 Hydra 模式

新组代码里你会反复看到这 5 个。把它们记住，再乱的项目都能拆。

### 1. `@hydra.main(...)`

```python
@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ...
```

- `config_path`：YAML 目录（**相对当前文件**）
- `config_name`：入口 yaml 不带后缀
- `version_base="1.3"`：固定 Hydra 默认行为（推荐写上）

启动时 Hydra 会：解析 CLI → 合并 yaml → 构造 `cfg` → 调用 `main(cfg)`。

### 2. `_target_` + `hydra.utils.instantiate`

这是 Hydra **最常见**的写法：

```yaml
# conf/model/default.yaml
_target_: _lit.LitBEVDetector
base_channels: 32
lr: 1e-3
```

```python
model = instantiate(cfg.model)
# 等价于：
# from _lit import LitBEVDetector
# model = LitBEVDetector(base_channels=32, lr=1e-3)
```

凡是构造函数能接受 kwargs 的类——Module、DataModule、Trainer、callbacks、loggers、甚至 `BEVConfig` dataclass——都能这样写。

读乱代码时的口诀：**看到 `_target_:`，就把它当成"懒加载的构造函数调用"**。

### 3. `defaults` 列表（config groups）

```yaml
# config.yaml
defaults:
  - bev: default        # 加载 conf/bev/default.yaml 到 cfg.bev
  - model: default
  - data: smoke
  - trainer: default
  - callbacks: default
  - logger: csv_only
  - _self_              # 把本文件内容覆盖在最后
```

每行的意思是「从对应**子目录**里挑一个 yaml 加载到对应 key 下」。

`_self_` 控制本文件相对 defaults 的优先级——放在最后表示**本文件的值最后写入**（最高优先）。

### 4. 命令行覆盖

这是日常用得最多的：

```bash
# 改单个值
python train.py model.lr=1e-4 trainer.max_epochs=100

# 换整个 group（model/default.yaml -> model/big.yaml）
python train.py model=big

# 同时换多个 group
python train.py data=full trainer=bf16

# 加 (`+`) / 删 (`~`) 一个 key
python train.py +trainer.deterministic=true
python train.py ~callbacks.lr_monitor

# 多 run sweep
python train.py -m model.lr=1e-3,5e-4,1e-4
```

### 5. OmegaConf 变量插值 `${...}`

我们在 `conf/model/default.yaml` 里写：

```yaml
epochs: ${trainer.max_epochs}
```

含义是「这个字段的值 = `cfg.trainer.max_epochs`，**延迟解析**」。
所以你 `python train.py trainer.max_epochs=200` 时，`cfg.model.epochs` 自动也变 200。

其他常见插值：

- `${hydra:runtime.output_dir}` —— 本次 run 的输出目录
- `${oc.env:HOME}` —— 环境变量
- `${now:%Y-%m-%d}` —— 当前日期（Hydra 提供）

---

## 一张图：cfg 是怎么"长出来"的

```
命令行:  python train.py model=big model.lr=1e-4 data=full

           ┌── conf/config.yaml ──┐
           │  defaults:           │
           │   - bev: default     │
           │   - model: default ◄─┼── 被 CLI 改成 "big"
           │   - data: smoke    ◄─┼── 被 CLI 改成 "full"
           │   - trainer: default │
           │   - ...              │
           └──────────────────────┘
                     │
        合并 yaml 内容 + _self_
                     │
                     ▼
           ┌──── cfg (DictConfig) ────┐
           │  cfg.bev      = {...}    │  ← 来自 bev/default.yaml
           │  cfg.model    = {...}    │  ← 来自 model/big.yaml
           │   .lr         = 1e-4     │  ← 被 CLI 覆盖
           │  cfg.data     = {...}    │  ← 来自 data/full.yaml
           │  cfg.trainer  = {...}    │
           │  ...                     │
           └──────────────────────────┘
                     │
                     ▼
              main(cfg) 被调用
```

每次启动开头我们都打印 `OmegaConf.to_yaml(cfg, resolve=True)`——这是排查"配置到底是什么"最快的办法。

---

## 怎么跑

### 训练（默认 smoke）

```bash
pip install hydra-core   # 装一次即可（自带 OmegaConf）

cd <repo-root>
python exercises/08_hydra_omegaconf/train.py
```

输出目录：`runs/08_hydra_omegaconf/<日期>/<时间>/`，里面会有：

```
.hydra/
  config.yaml         # Hydra 自动保存的"完全展开"配置（极重要）
  hydra.yaml          # Hydra 自身设置
  overrides.yaml      # 你这次跑给的 CLI overrides
checkpoints/
  best-029.ckpt
  last.ckpt
csv/
  version_0/metrics.csv
train.log             # Hydra 默认的 logging
```

`.hydra/config.yaml` 是你**复现一次实验最重要的文件**——后面贴给同事就够了。

### 几个常用变形

```bash
# 大模型 + bf16 + wandb
python exercises/08_hydra_omegaconf/train.py \
    model=big trainer=bf16 logger=wandb

# 全量数据
python exercises/08_hydra_omegaconf/train.py data=full

# 续训
python exercises/08_hydra_omegaconf/train.py \
    ckpt_path=runs/08_hydra_omegaconf/2026-05-27/22-30-00/checkpoints/last.ckpt \
    trainer.max_epochs=100

# 多 run sweep（3 个 lr 自动跑 3 次）
python exercises/08_hydra_omegaconf/train.py -m model.lr=1e-3,5e-4,1e-4
```

### 预测

```bash
python exercises/08_hydra_omegaconf/predict.py \
    ckpt_path=runs/08_hydra_omegaconf/2026-05-27/22-30-00/checkpoints/last.ckpt
```

忘了传 `ckpt_path=...` 会立刻报错——因为 `predict.yaml` 里写了 `ckpt_path: ???`。

---

## 常见坑（提前避雷）

1. **cwd 被改了**
   - 1.x 版本 Hydra 默认会把 `os.getcwd()` 换成 run 目录。我们在 `config.yaml` 里设了 `hydra.job.chdir: false` 关掉它。读老项目时若看到 `os.getcwd()` 突然变奇怪，先查这个。

2. **`${...}` 的解析时机**
   - 插值是**懒解析**的：你在 yaml 里写 `epochs: ${trainer.max_epochs}`，只有读到 `cfg.model.epochs` 的那一刻才会求值。CLI 覆盖发生在求值之前，所以 `python train.py trainer.max_epochs=200` 是有效的。

3. **`defaults` 里 `_self_` 的位置**
   - `_self_` 放在 defaults 末尾 → 本文件 override config groups（常用）
   - `_self_` 放在开头 → config groups override 本文件
   - 读乱代码时把位置看清楚，能省半小时调试。

4. **`_target_` 解析路径**
   - `_lit.LitBEVDetector` 要求 `_lit` 在 `sys.path` 里。我们在 `train.py` 开头手动插入，所以 OK。
   - 老项目用 `src.models.foo.Bar` 这种结构，是因为 `src` 在 PYTHONPATH 里。

5. **`???` 占位符**
   - 在 yaml 里 `ckpt_path: ???` 表示"必须由 CLI 提供"。访问时若仍是 `???` 会抛 `MissingMandatoryValue`。

6. **list vs DictConfig 在 yaml 里的表达**
   - `conf/logger/csv_only.yaml` 顶层就是个 `-` 起头的 list → `cfg.logger` 是 ListConfig
   - `conf/callbacks/default.yaml` 用 named keys → `cfg.callbacks` 是 DictConfig
   - 两者在 Python 里访问方式不同，乱代码里常常混用，看清楚再写。

---

## 读乱 Hydra 代码的检查清单

入新组打开一个 Hydra 项目，按这个顺序看最快：

1. **找入口** — `@hydra.main(...)` 一行，看 `config_name` 是哪份 yaml
2. **打开入口 yaml** — `defaults:` 列表告诉你 group 都有哪些
3. **顺着 `defaults` 进每个子 yaml** — 看 `_target_` 是哪个类
4. **在 main() 里找 `instantiate(...)`** — 这就是「类→实例」的全部魔法
5. **跑一次** — 让 Hydra 自动 dump 到 `.hydra/config.yaml`，那是当前**真正生效**的配置
6. **看 `${...}` 插值** — 哪些字段被串起来了

做完这 6 步，你比项目里 80% 的人都更懂这套配置。

---

## 这一关 vs LightningCLI（给你一个对照）

| | Hydra (你新组用的) | LightningCLI |
|---|---|---|
| 学习曲线 | 中等（5 个模式） | 低（auto CLI） |
| 配置组合 | defaults list（一等公民） | 单 yaml 或 `--config a --config b` |
| Sweep | `-m` 内置 | 需外部工具 |
| 改 cwd | 老版会，新版可关 | 不改 |
| 与 Lightning 耦合 | 无 | 紧密 |
| 你需要它做的 | **认得 `_target_`、`defaults`、`instantiate`** | 暂不学 |

ex08 之后你就齐了——已经能 **读懂任何用 Hydra + OmegaConf + Lightning 的项目结构**。

---

## 文件依赖

`requirements.txt` 里需要加：

```
hydra-core>=1.3
```

OmegaConf 是 Hydra 的依赖，会被自动安装。
