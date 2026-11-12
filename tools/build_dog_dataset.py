"""把 Pascal VOC 的 dog 类转成 YOLOv8 单类检测数据集。

输出结构::

    dataset/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt      # 0 cx cy w h (归一化)
      data.yaml
      dataset_stats.json

划分策略
--------
* test  : VOC2007 test 中的狗图 + 同比例背景图（官方测试集，完全不参与训练）
* train/val: VOC2007 trainval + VOC2012 trainval 的狗图，按图片 9:1 分层随机划分
* 背景图: 从同 split 的无狗图片中抽取，占狗图的 10%，用于教模型「有/无」

用法::

    python build_dog_dataset.py --voc_dir ../raw --out ../dataset
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:  # Windows 控制台默认 GBK，避免中文/符号打印报错
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

DOG = "dog"

# (年份, split 名, 属于哪个输出集)
SPLITS = [
    ("VOC2007", "trainval", "trainpool"),
    ("VOC2012", "trainval", "trainpool"),
    ("VOC2007", "test", "test"),
]


def find_roots(voc_dir: Path) -> dict[str, Path]:
    """定位 VOC2007 / VOC2012 根目录（兼容 tar 解压后的层级）。"""
    roots: dict[str, Path] = {}
    for year in ("VOC2007", "VOC2012"):
        hits = [p for p in voc_dir.rglob(year) if (p / "Annotations").is_dir()]
        if hits:
            roots[year] = sorted(hits, key=lambda p: len(str(p)))[0]
    return roots


def read_split_ids(root: Path, split: str) -> list[str]:
    """读取 ImageSets/Main/{split}.txt；trainval = train + val。"""
    base = root / "ImageSets" / "Main"
    ids: list[str] = []
    files = ["train.txt", "val.txt"] if split == "trainval" else [f"{split}.txt"]
    for fn in files:
        f = base / fn
        if not f.exists():
            raise FileNotFoundError(f)
        ids += [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return list(dict.fromkeys(ids))


def parse_dog_boxes(xml_path: Path) -> tuple[list[tuple[float, float, float, float]], int]:
    """返回 (dog 框列表(xmin,ymin,xmax,ymax), difficult 数量)。"""
    root = ET.parse(xml_path).getroot()
    boxes, n_diff = [], 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name != DOG:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            continue
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        if (obj.findtext("difficult") or "0").strip() == "1":
            n_diff += 1
        boxes.append((x1, y1, x2, y2))
    return boxes, n_diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_dir", default=r"..\raw", help="VOCdevkit 所在目录")
    ap.add_argument("--out", default=r"..\dataset", help="输出数据集目录")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--neg_ratio", type=float, default=0.10, help="背景图占正样本比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", default=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    voc_dir, out = Path(args.voc_dir).resolve(), Path(args.out).resolve()
    roots = find_roots(voc_dir)
    print("VOC 根目录:", {k: str(v) for k, v in roots.items()})
    if not roots:
        raise SystemExit(f"在 {voc_dir} 下找不到 VOC2007/VOC2012")

    pool: list[dict] = []   # 训练池（含正负）
    test: list[dict] = []   # 测试集
    for year, split, bucket in SPLITS:
        root = roots.get(year)
        if root is None:
            print(f"[warn] 缺少 {year}，跳过")
            continue
        ann_dir, img_dir = root / "Annotations", root / "JPEGImages"
        ids = read_split_ids(root, split)
        pos = neg = n_diff = 0
        for iid in ids:
            xml = ann_dir / f"{iid}.xml"
            if not xml.exists():
                continue
            boxes, nd = parse_dog_boxes(xml)
            if not boxes:
                neg += 1
                continue
            pos += 1
            n_diff += nd
            rec = {"id": iid, "img": img_dir / f"{iid}.jpg", "boxes": boxes, "src": f"{year}/{split}"}
            (test if bucket == "test" else pool).append(rec)
        print(f"  {year}/{split}: {len(ids)} 张图, 含狗 {pos}, 无狗 {neg}, difficult {n_diff}")

    # 训练池里再切 val
    rng.shuffle(pool)
    n_val = max(1, int(len(pool) * args.val_ratio))
    val, train = pool[:n_val], pool[n_val:]
    print(f"\n划分: train={len(train)} val={len(val)} test={len(test)}")

    # 背景图（无狗）候选：同 year/split 里没有 dog 的图
    def neg_candidates(year: str, split: str) -> list[Path]:
        root = roots[year]
        ann_dir, img_dir = root / "Annotations", root / "JPEGImages"
        out_paths = []
        for iid in read_split_ids(root, split):
            xml = ann_dir / f"{iid}.xml"
            if not xml.exists():
                continue
            boxes, _ = parse_dog_boxes(xml)
            if not boxes:
                out_paths.append(img_dir / f"{iid}.jpg")
        return out_paths

    negs_train: list[dict] = []
    for year, split in (("VOC2012", "trainval"), ("VOC2007", "trainval")):
        if year in roots:
            for p in neg_candidates(year, split):
                negs_train.append({"id": p.stem, "img": p, "boxes": [], "src": f"{year}/{split}"})
    negs_test = []
    if "VOC2007" in roots:
        for p in neg_candidates("VOC2007", "test"):
            negs_test.append({"id": p.stem, "img": p, "boxes": [], "src": "VOC2007/test"})

    rng.shuffle(negs_train)
    rng.shuffle(negs_test)
    n_neg_train = int(len(train) * args.neg_ratio)
    n_neg_val = int(len(val) * args.neg_ratio)
    n_neg_test = int(len(test) * args.neg_ratio)
    gn = negs_train[: n_neg_train + n_neg_val]
    train_neg, val_neg = gn[:n_neg_train], gn[n_neg_train:]
    test_neg = negs_test[:n_neg_test]

    splits = {
        "train": train + train_neg,
        "val": val + val_neg,
        "test": test + test_neg,
    }

    stats = {"class": DOG, "neg_ratio": args.neg_ratio, "val_ratio": args.val_ratio, "splits": {}}
    total_boxes = 0
    for name, recs in splits.items():
        n_box = sum(len(r["boxes"]) for r in recs)
        total_boxes += n_box
        stats["splits"][name] = {
            "images": len(recs),
            "with_dog": sum(1 for r in recs if r["boxes"]),
            "background": sum(1 for r in recs if not r["boxes"]),
            "dog_boxes": n_box,
            "sources": dict(Counter(r["src"] for r in recs)),
        }
        print(f"  {name:5s}: {len(recs):5d} 张, 有狗 {stats['splits'][name]['with_dog']}, "
              f"背景 {stats['splits'][name]['background']}, 框 {n_box}")

    if args.dry_run:
        print("\n[dry-run] 不写文件")
        return

    if out.exists():
        shutil.rmtree(out)
    for name, recs in splits.items():
        img_out = out / "images" / name
        lbl_out = out / "labels" / name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for r in recs:
            if not Path(r["img"]).exists():
                continue
            dst = img_out / f"{r['id']}.jpg"
            if not dst.exists():
                shutil.copy2(r["img"], dst)
            lines = []
            for (x1, y1, x2, y2) in r["boxes"]:
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
                # VOC 尺寸以 XML 为准
                w, h = _size_of(r["img"])
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_out / f"{r['id']}.txt").write_text("\n".join(lines), encoding="utf-8")

    yaml = (
        f"# dog 单类检测数据集 (Pascal VOC 2007+2012 转换)\n"
        f"path: {out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: 1\n"
        f"names:\n  0: dog\n"
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    (out / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] 数据集已生成: {out}  (共 {total_boxes} 个狗框)")
    print(f"   data.yaml -> {out / 'data.yaml'}")


_SIZE_CACHE: dict[str, tuple[int, int]] = {}


def _size_of(path: Path) -> tuple[int, int]:
    from PIL import Image

    key = str(path)
    if key not in _SIZE_CACHE:
        with Image.open(path) as im:
            _SIZE_CACHE[key] = im.size
    return _SIZE_CACHE[key]


if __name__ == "__main__":
    main()
