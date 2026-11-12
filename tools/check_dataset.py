"""数据集体检：统计 + 标注合法性校验 + 抽样可视化。

用法: python check_dataset.py --data ../dataset/data.yaml --grid ../runs/check_grid.jpg
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"..\dataset\data.yaml")
    ap.add_argument("--grid", default=r"..\runs\check_grid.jpg")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.data))
    root = Path(cfg["path"])
    print(f"数据集根目录: {root}\n类别: {cfg['names']}")
    total_bad = 0
    for split in ("train", "val", "test"):
        imgs = sorted((root / cfg[split]).glob("*.jpg"))
        n_box = n_neg = bad = 0
        for img in imgs:
            lbl = root / "labels" / split / f"{img.stem}.txt"
            if not lbl.exists():
                bad += 1
                print(f"  [缺标签] {img.name}")
                continue
            rows = [ln.split() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not rows:
                n_neg += 1
            for r in rows:
                cls, *xywh = r
                v = [float(x) for x in xywh]
                if int(cls) != 0 or any((x < 0 or x > 1) for x in v) or v[2] <= 0 or v[3] <= 0:
                    bad += 1
                    print(f"  [非法标注] {lbl.name}: {r}")
                n_box += 1
        total_bad += bad
        print(f"{split:5s}: {len(imgs):5d} 图 | {n_box:5d} 框 | 背景图 {n_neg:4d} | 异常 {bad}")
    print(f"\n合计异常: {total_bad}")

    # 抽样可视化（只抽有狗的图）
    random.seed(0)
    pool = []
    for split in ("train", "val", "test"):
        for img in sorted((root / cfg[split]).glob("*.jpg")):
            lbl = root / "labels" / split / f"{img.stem}.txt"
            if lbl.exists() and lbl.read_text(encoding="utf-8").strip():
                pool.append((split, img, lbl))
    picks = random.sample(pool, min(args.n, len(pool)))
    cell = 320
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (28, 28, 30))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for i, (split, img, lbl) in enumerate(picks):
        im = Image.open(img).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        for ln in lbl.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            _, cx, cy, bw, bh = [float(x) for x in ln.split()]
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 80), width=max(2, w // 200))
        im.thumbnail((cell, cell - 18))
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.paste(im, (x + (cell - im.width) // 2, y + 18))
        draw.text((x + 4, y + 2), f"{split}/{img.stem}", fill=(200, 220, 255), font=font)
    out = Path(args.grid)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print(f"抽样图: {out}")


if __name__ == "__main__":
    main()
