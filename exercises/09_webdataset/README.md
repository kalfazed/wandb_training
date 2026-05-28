# 练习 09 — WebDataset

## 这次练什么

把 NuScenes 这种「成千上万小文件」的数据集，**离线打包成少量 `.tar` shard**，训练时用 `webdataset` 流式读。模型 / loss / Trainer 都和 ex07-08 一样，只换 DataModule。

> 真正的收益不是「网络变快了」，而是把读法换成了 Lustre 这类并行文件系统**喜欢的姿势**：
>
> - 元数据操作（`open` / `stat`）从「每个 sample 多次」降到「每个 shard 一次」
> - 随机小文件读 → 顺序大块读
> - 多 worker / 多卡时，shard 是天然的分片单位

---

## 目录

```
exercises/09_webdataset/
├── pack_webdataset.py   # 一次性：NuScenes -> nuscenes-{split}-*.tar
├── _lit.py              # LitBEVWebDataModule（IterableDataset）+ 复用 LitBEVDetector
├── train.py             # argparse 入口；其余照搬 ex07
└── README.md            # 你正在看
```

打包产物默认落在仓库根目录的 `runs/09_webdataset/shards/`：

```
runs/09_webdataset/shards/
├── nuscenes-train-000000.tar
├── nuscenes-train-000001.tar
└── nuscenes-val-000000.tar
```

---

## 1. WebDataset 的核心约定

WebDataset **没有正式 schema**。它只有两条约定：

1. **数据 = 一堆普通的 `.tar` 文件**（每个叫一个 *shard*）。
2. **tar 里同名前缀的多个文件 = 一个 sample**。比如：

   ```
   000042.points.npy
   000042.boxes.json
   000042.meta.json
   ```

   这三个是同一个 sample 的三个字段（按扩展名区分），`__key__` 是 `000042`。

读的时候概念上是：

```python
for sample in wds.WebDataset(urls):
    # sample == {"__key__": "000042",
    #            "points.npy": b"...",     # raw bytes
    #            "boxes.json": b"...",
    #            "meta.json":  b"..."}
    ...
```

后续的 `.decode()` / `.map(...)` 负责把 raw bytes 变成 tensor。

---

## 2. 为什么团队多半「先 pickle」？

你看到的「转 pickle」一般是下面这条捷径：**一个 sample 只塞一个 `.pkl`**，里面是个 Python dict：

```
000042.pkl   <- pickle.dumps({"points": ndarray, "boxes_arr": ndarray, ...})
```

好处：

- 不用关心 schema、不用为每个字段写 decoder。
- 嵌套结构、numpy、dataclass 都 pickle 一行搞定。
- 解码只跑一次 `pickle.loads`。
- IO 友好：一个 sample 一个 member，少一次 tar 寻址。

代价：

- **跨语言不行**（pickle 只 Python 认）。
- **跨版本脆**：如果 pickle 里直接存了 `nusc_det.dataset.Box3D` 这种类实例，将来类被改名/搬家就读不出来。

> 本练习按团队风格，**用 pickle 做 sample 载荷**，但**只 pickle 普通类型**（numpy + dict + list[str]），不 pickle `Box3D` 这类项目特有类——训练时再用 `_arrays_to_boxes` 包回来。这是工程界最常见的折中。

---

## 3. 打包：`pack_webdataset.py`

```bash
# smoke test：只打 8 帧
python exercises/09_webdataset/pack_webdataset.py \
    --max-frames 8 --val-frames 2 --maxcount 4
```

关键参数：

| 参数 | 作用 |
|------|------|
| `--data-root` | 原始 NuScenes 风格目录（同 ex01-08） |
| `--out-dir` | 输出根（默认 `runs/09_webdataset/shards/`） |
| `--max-frames` | 总共打多少帧（smoke test 用） |
| `--val-frames` | 末尾 N 帧进 val |
| `--maxcount` | 一个 shard 装多少 sample（决定 shard 数量） |
| `--maxsize-mb` | 一个 shard 的大小上限（任一上限达到就切新 shard） |

代码里只有两件事：

1. 遍历 `NuScenesLidarDetDataset`，把 `points`/`boxes`/`meta` 收成 `dict`，`pickle.dumps` 成 bytes。
2. 用 `wds.ShardWriter` 写：

   ```python
   sink.write({"__key__": f"{i:06d}", "pkl": payload_bytes})
   ```

   字典里 `__key__` 是 sample 的「身份证」，其他键的名字（这里是 `"pkl"`）就是 tar 里那个 member 的**文件扩展名**。

写完看一眼：

```bash
tar -tf runs/09_webdataset/shards/nuscenes-train-000000.tar
# 000000.pkl
# 000001.pkl
# 000002.pkl
# 000003.pkl
```

---

## 4. 训练时怎么读：`_lit.py` 里的 pipeline

最核心的就那几行：

```python
ds = wds.WebDataset(
    urls,
    shardshuffle=100,                # 打乱 SHARD 顺序的缓冲区
    nodesplitter=wds.split_by_node,  # 多机：每个 rank 只读自己那份 shard
    workersplitter=wds.split_by_worker,  # 多 worker：每个 worker 只读自己那份 shard
    handler=wds.warn_and_continue,   # 坏 record 跳过，不炸训练
)
ds = (
    ds.shuffle(256)        # 在 buffer 里打乱 SAMPLE（近似全局 shuffle）
      .map(_decode_sample) # bytes -> dict (pickle.loads)
      .map(_SampleToBEV(...))  # dict -> {"bev", "targets", "meta"}
      .with_epoch(64)      # 关键：定义"一个 epoch = 多少 sample"
)
```

这几条都是 **Python 标准的 iterator 链**——`WebDataset` 是个 `IterableDataset`，没有 `__getitem__`，所以：

- DataLoader **不能** 传 `shuffle=True`，shuffle 通过 `.shuffle(buf)` 在 pipeline 内完成。
- 没有 `len(ds)`。**`with_epoch(N)` 告诉 Lightning「一个 epoch = N 个 sample」**，否则 progress bar / max_epochs 都不知道什么时候算一轮。
- DDP 多卡训练时 `split_by_node` 让 rank0 / rank1 各读不同的 shard，零协调开销。

---

## 5. 训练：`train.py`

```bash
# 先打包（见 §3），再训：
python exercises/09_webdataset/train.py \
    --train-shards 'runs/09_webdataset/shards/nuscenes-train-{000000..000001}.tar' \
    --val-shards   'runs/09_webdataset/shards/nuscenes-val-000000.tar' \
    --epochs 5 \
    --train-samples 32 --val-samples 8
```

注意 shard URL **支持 brace 展开**（`{000000..000001}` 是 webdataset 自带的语法），也支持普通 `*.tar` glob（DataModule 里会展开）。

输出和前面练习一样到 `runs/09_webdataset/`：

```
runs/09_webdataset/
├── checkpoints/best-XXX.ckpt + last.ckpt
└── csv/version_0/metrics.csv
```

---

## 6. 和 ex08 的逐行对比

| ex08（map-style） | ex09（webdataset） |
|---|---|
| `NuScenesBEVDataset(Dataset)` + `__getitem__` | `wds.WebDataset(urls).map(...)` |
| `DataLoader(..., shuffle=True)` | `ds.shuffle(buf)` + `DataLoader(shuffle=False)` |
| `Subset(full, train_indices)` | shard 划分（不同 tar = 不同 split） |
| `len(ds)` 已知 | 用 `ds.with_epoch(N)` 人为定义 |
| `points_to_bev` 在 `__getitem__` 里 | 在 `_SampleToBEV.__call__` 里（同样的代码） |
| 多机靠 `DistributedSampler` | 多机靠 `wds.split_by_node` |

**模型代码（`LitBEVDetector` / `BEVDetector` / `detection_loss`）一行都没动。**

---

## 7. 你可以单独动手玩玩的事

1. 改 `--maxcount` 看 shard 数量怎么变，再看 `tar -tf ...tar` 里的 sample 数。
2. 把 `_lit.py` 的 `ds.shuffle(256)` 改成 `ds.shuffle(2)` 训一轮，观察 train loss 曲线抖动。
3. 把 `pack_webdataset.py` 改成 **不 pickle 整个 dict**，而是写成 `000000.points.npy` + `000000.boxes.json` 两个 member；训练侧用 `.decode()` 自动解码再 `.to_tuple("points.npy", "boxes.json")`。看看两种风格的差异。
4. 把 `--num-workers 4` 打开，再加几个 shard，观察 GPU 利用率（`nvidia-smi -l 1`）和 dataloader 吞吐。
5. 加 wandb logger（参考 ex04），看 train/loss 曲线在 webdataset 下是否更平滑（一般会，因为 shard 顺序 + buffer shuffle 减少了 IO 抖动）。

---

## 8. 一句话总结

**WebDataset 没 schema，只有约定**：tar shard + 同名前缀分组。  
**「pickle 在 tar 里」** 是团队最常见的简化：一个 sample 一个 `.pkl`，离线打一次，训练时全程顺序读。  
**真正变快**是因为 Lustre 喜欢顺序大文件，而不是 webdataset 本身有什么魔法。
