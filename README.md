# 狗检测数据集 + YOLOv8s 训练工程

单类目标检测：输入一张图，输出「有没有狗」以及每只狗的边界框（`dog: 0`）。

- 练习集（train）：**1,710 张**（1,555 张有狗 / 155 张背景）
- 验证集（val）：**189 张**（172 有狗 / 17 背景）
- 测试集（test）：**476 张**（433 有狗 / 43 背景），**完全没参与训练**
- 标注框：train 1,857 个 / val 222 个 / test 530 个

## 1. 数据从哪来

| 来源 | 内容 | 在本工程中的作用 |
|---|---|---|
| **Pascal VOC 2012** `trainval` | 11,540 张，含 dog 的 1,297 张 | 训练主力 |
| **Pascal VOC 2007** `trainval` | 5,011 张，含 dog 的 430 张 | 补充训练 |
| **Pascal VOC 2007** `test` | 4,952 张，含 dog 的 433 张 | **测试集（官方划分，干净）** |

三个包都从 **ModelScope 镜像** `yolo_master/VOC0712` 下载（国内 CDN 实测 15–20 MB/s）；
官方源 `host.robots.ox.ac.uk` 实测只有 ~18 KB/s，2 GB 要下一天，不可用。

> 为什么不用 COCO？COCO `train2017` 是 19 GB，按本机国际带宽要 50 分钟以上；
> VOC 的 dog 类框质量同样可靠、体量小 10 倍，作为「练习集 + 测试集」完全够用。
> 想要更贴近真实场景，见文末「数据集升级路线」。

## 2. 目录结构

```
dog_yolo/
├─ dataset/                 # ← 直接喂给 YOLO 的数据集
│  ├─ images/{train,val,test}/*.jpg
│  ├─ labels/{train,val,test}/*.txt   # 0 cx cy w h（归一化），背景图是空 txt
│  ├─ data.yaml
│  └─ dataset_stats.json
├─ raw/                     # VOC 原始压缩包与解压结果（可删，省 5 GB）
├─ weights/yolov8s.pt       # 预训练权重（从本机已有权重复制，无需联网）
├─ tools/
│  ├─ build_dog_dataset.py  # VOC -> YOLO 转换 + 划分 + 负样本
│  ├─ check_dataset.py      # 数据集体检 + 标注抽样可视化
│  ├─ ms_download.py        # ModelScope 断点续传下载器
│  └─ nomp.py               # 沙箱兼容（禁止命名管道时替代 ThreadPool）
├─ train.py                 # 训练 + 自动在 val/test 上评估
├─ predict_demo.py          # 抽样预测并拼图
└─ runs/                    # 训练输出
```

`data.yaml`：

```yaml
path: D:/ds工作文档2/dog_yolo/dataset
train: images/train
val: images/val
test: images/test
nc: 1
names: {0: dog}
```

## 3. 怎么划分的

1. `test` 直接用 VOC2007 官方 test 里的狗图，**从训练池里彻底隔离**；
2. 训练池 = VOC2012 trainval + VOC2007 trainval 的狗图，按图片 9:1 随机切出 `val`；
3. 额外混入 **10% 无狗背景图**（同 split 内随机取，多为猫/人/室内场景），
   用来教模型「没有狗时不要乱框」——这是「判断有无」的关键，纯正样本训出来的模型
   在空图上也会硬框一个狗出来。

重新生成（换随机种子、换负样本比例都可以）：

```powershell
python tools\build_dog_dataset.py --voc_dir .\raw --out .\dataset --val_ratio 0.1 --neg_ratio 0.1 --seed 42
python tools\check_dataset.py --data .\dataset\data.yaml --grid .\runs\check_grid.jpg
```

## 4. 训练

在 `dog_yolo` 目录下（`py` 指本工程自带的便携 Python）：

```powershell
$py = "..\py312\python.exe"
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"   # 让 ultralytics 不去联网下字体
& $py train.py --epochs 80 --batch 16        # 8G 显存够用；OOM 就 --batch 8
```

其他用法：

```powershell
& $py train.py --smoke                        # 2 epoch 跑通流程
& $py train.py --model weights\yolov8n.pt --epochs 50   # 换更小的模型
& $py predict_demo.py                         # 抽样预测并拼图
```

训练结束后 `train.py` 会自动用 `best.pt` 在 `val` 和 `test` 上各评估一次并打印 mAP。

单独评估 / 预测 / 导出（`yolo.exe` 在 `..\py312\Scripts\`）：

```powershell
$yolo = "..\py312\Scripts\yolo.exe"
& $yolo detect val     model=runs\dog_yolov8s\weights\best.pt data=dataset\data.yaml split=test
& $yolo detect predict model=runs\dog_yolov8s\weights\best.pt source=dataset\images\test save=True conf=0.25
& $yolo export         model=runs\dog_yolov8s\weights\best.pt format=onnx imgsz=640
```

## 5. 训练结果（yolov8s，80 epoch，640）

| 集合 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val（189 张） | 0.906 | 0.784 | **0.882** | 0.658 |
| test（476 张，未见过） | 0.861 | 0.747 | **0.859** | 0.621 |

- 配置：`imgsz=640, batch=16, AdamW(auto), cos_lr, close_mosaic=10`，单卡 RTX 5060 约 **41 分钟**，显存峰值 3.6 GB。
- 推理：约 **5 ms/张**（≈200 FPS，RTX 5060，FP16）。
- test 比 val 低约 2 个点，符合预期：test 是 VOC2007 官方测试集，年份/场景分布与训练用的 2007 trainval + 2012 trainval 不完全一致。
- 产物：`runs/dog_yolov8s/weights/best.pt`、`results.png`（曲线）、`confusion_matrix.png`、`val_batch*_pred.jpg`；
  `runs/test_predict.jpg` 是测试集抽样预测拼图，`runs/check_grid.jpg` 是标注核验图。

> mAP50 ≈ 0.86 的含义：IoU>0.5 时，PR 曲线下面积 86%；单类狗检测里这已经是「能直接交作业」的水平。
> 再往上通常要靠更多数据（COCO）或更大模型，而不是继续调参。

## 6. 可视化图表（可直接贴进报告）

```powershell
& $py tools\make_report.py            # 用缓存，秒出
& $py tools\make_report.py --refresh  # 重新推理 val 并重算曲线
& $py tools\make_curves.py            # 2×3 六宫格训练曲线（报告常用版式）
```

输出到 `runs/report/`：

| 文件 | 内容 |
|---|---|
| `01_dashboard.png` | 3×3 总览：训练/验证损失、mAP 曲线、P/R 曲线、学习率、每轮耗时、数据集规模、目标尺寸分布、训练摘要卡 |
| `02_pr_errors.png` | PR 曲线（AP=0.882）+ TP/FP/FN 误差分解 |
| `03_qualitative.png` | 验证集「标注 vs 预测」上下对比 |
| `04_curves.png` | 2×3 六宫格：box/cls/dfl 损失（训练 vs 验证）、P/R、mAP、学习率，标题带最优轮次 |

读图要点（汇报时可以直接讲）：

- **第 70 轮损失同时下跳、mAP 上抬**是 `close_mosaic` 关掉 mosaic 增强，属正常现象，不是训练异常。
- 训练与验证损失（对数轴）同步下降、没有张口发散 → **没有过拟合**。
- PR 曲线在 recall < 0.8 的区间几乎贴着 precision ≈ 0.97，说明高置信度下误检很少。
- 误差构成 **174 正确 / 18 误检 / 48 漏检**：漏检是主要瓶颈（远景小目标、遮挡、只露局部），
  下一步该补数据，而不是调阈值。
- 目标尺寸中位数 223 px、<128 px 占 21%，解释了为什么把 `imgsz` 从 512 提到 640 是有收益的。

> 图表字体用的是系统自带 Microsoft YaHei；换机器如果中文变方块，改 `tools/make_report.py` 顶部的 `CJK` 路径即可。

## 7. 本机环境

| 项 | 值 |
|---|---|
| Python | 3.12.10（`.\py312\python.exe`，便携版，无需装 conda） |
| torch | 2.11.0+cu128 |
| ultralytics | 8.4.203 |
| GPU | RTX 5060 Laptop 8 GB |
| 权重 | `weights/yolov8s.pt`（22 MB，已就位） |

## 8. 踩过的坑

1. **`PermissionError [WinError 5]`**：本机沙箱禁止创建命名管道，`multiprocessing.Pool`
   一初始化就崩。`tools/nomp.py` 把 `ThreadPool` 换成串行实现即可，`--workers 0`。
2. **ultralytics 联网下 `Arial.ttf`**：把 `arial.ttf` 放进 `YOLO_CONFIG_DIR` 指向的
   `Ultralytics/` 子目录，或直接 `plots=False`。
3. **VOC 图偏小**（多为 500×375），`imgsz=640` 会放大图像；想更快可 `--imgsz 512`。
4. 训练日志、曲线、混淆矩阵都在 `runs/dog_yolov8s/`。

## 9. 数据集升级路线（想做更好时）

| 方案 | 体量 | 说明 |
|---|---|---|
| COCO2017 `val2017` | 815 MB | 5,000 张高质量图，含狗约 260 张，适合当**跨数据集测试集**验泛化 |
| COCO2017 `train2017` | 19 GB | 含狗约 1.2 万张，效果上限最高，但下载+解压约 1 小时 |
| Open Images V7（dog） | 按需 | 类别更杂、场景更真实，需要先按类别清单筛图 |

三者都能从 ModelScope 镜像拉：`PAI/COCO2017`、`anyforge/coco-2017`（脚本 `tools/ms_download.py`
改一下 `ns/name` 和文件名即可）。

## 10. 顺手给同组同学

- **猫**：把 `tools/build_dog_dataset.py` 里的 `DOG = "dog"` 改成 `"cat"` 重跑一遍，
  VOC 里猫的图比狗还多一点，流程完全一样。
- **老鼠**：VOC 和 COCO 都**没有** mouse 类。用 Open Images（有 `Mouse` 类，约 1,200 框）
  或 Roboflow 上的 `mouse detection` 数据集，转成同样的 `images/ + labels/` 结构即可，
  `data.yaml` 不用改。

## 11. 传到 GitHub

群里要求「把数据推到 GitHub，然后发网址」。仓库里放什么：

| 进仓库 | 大小 | 不进仓库（`.gitignore` 已排除） |
|---|---|---|
| `dataset/`（2,375 张图 + 标签 + data.yaml） | 241 MB | `raw/`（5.5 GB VOC 原始包） |
| `tools/`、`train.py`、`predict_demo.py` | < 100 KB | `runs/dog_yolov8s/` 的中间图、`last.pt` |
| `runs/report/`（4 张可视化图） | 2.3 MB | `.ultralytics/`、`*.cache` |
| `runs/dog_yolov8s/weights/best.pt` | 21 MB | `weights/yolov8s.pt`（官方可再下） |

合计约 **265 MB**，没有单文件超过 GitHub 的 100 MB 上限。

### 三条路，任选一条

**路线 A：图形界面（VS Code 便携版，已装好，不需要管理员）**

双击 `打开VSCode.cmd` → 左侧点「源代码管理」图标（`Ctrl+Shift+G`）→ 填提交信息 → `Ctrl+Enter` 提交 →
点「发布分支 / Publish Branch」→ 浏览器登录 GitHub → 自动建仓库并推送。

> 便携版在 `../tools/VSCode`，设置里已经把 `git.path` 指到 `../tools/PortableGit`，不用再配。

**路线 B：命令行（最快）**

```powershell
cd D:\ds工作文档2\dog_yolo
.\tools\push_github.ps1 -Repo https://github.com/你的用户名/dog-detection-yolov8s.git
```

脚本会做：`git init` → 配身份（第一次问一次）→ `add` + `commit` → 关联 origin → `push`。

**路线 C：GitHub Desktop（需要管理员密码）**

安装包在 `../_dl/GitHubDesktopSetup-x64.exe`。普通用户双击会报
`Unable to write to C:\ProgramData\...`——它要往系统目录写东西，而这台机器的当前账号不是管理员。
右键 →「以管理员身份运行」才能装上。

**密码处要填 Personal Access Token，不是 GitHub 登录密码**（2021 年起就不能用密码了）：

github.com → Settings → Developer settings → Personal access tokens → **Tokens (classic)** →
Generate new token (classic) → 勾 `repo` → 生成后复制 `ghp_xxxx`，粘贴到密码提示处。
Windows 会弹「Git Credential Manager」窗口，选 token 方式粘贴即可，之后会记住。

### 传不动 / 太慢怎么办

- **只想先交代码**：加 `-NoDataset`，仓库降到 ~25 MB，秒传；数据集另外用网盘或 Release 挂链接。
- **开了代理还是慢**：给 git 单独设代理（端口照你代理软件的实际端口改）
  ```powershell
  git config --global http.proxy http://127.0.0.1:12450
  git config --global https.proxy http://127.0.0.1:12450
  ```
- **中途断了**：直接重跑同一个脚本，git 会复用已经打包好的对象，不用从头开始。
- **push 被拒（rejected）**：说明远端不是空仓库（多半建库时勾了 README），执行
  `git pull --rebase origin main` 再重跑脚本。
- **真的传不上去**：GitHub 网页支持拖拽上传，但单次 100 个文件、单文件 25 MB，
  2,375 张图要拖 24 次，不推荐。



