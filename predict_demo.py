"""用训练好的 best.pt 在测试集上抽样预测，拼成一张对比图。

    python predict_demo.py                       # 默认 runs/dog_yolov8s/weights/best.pt
    python predict_demo.py --n 12 --conf 0.25
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))
import nomp  # noqa: E402,F401  沙箱兼容，必须在 ultralytics 之前


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "runs" / "dog_yolov8s" / "weights" / "best.pt"))
    ap.add_argument("--images", default=str(HERE / "dataset" / "images" / "test"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=str(HERE / "runs" / "test_predict.jpg"))
    args = ap.parse_args()

    from ultralytics import YOLO

    files = sorted(Path(args.images).glob("*.jpg"))
    random.seed(1)
    picks = random.sample(files, min(args.n, len(files)))
    model = YOLO(args.weights)
    results = model.predict([str(p) for p in picks], conf=args.conf, imgsz=args.imgsz, verbose=False)

    cell, cols = 320, 4
    rows = (len(results) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (28, 28, 30))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for i, r in enumerate(results):
        im = Image.fromarray(r.plot()[:, :, ::-1])  # BGR -> RGB
        im.thumbnail((cell, cell - 18))
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.paste(im, (x + (cell - im.width) // 2, y + 18))
        draw.text((x + 4, y + 2), Path(r.path).stem, fill=(200, 220, 255), font=font)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print(f"[OK] {out}  ({len(results)} 张)")

    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        confs = [f"{float(b.conf):.2f}" for b in (r.boxes or [])]
        print(f"  {Path(r.path).stem}: {n} 只 [{', '.join(confs)}]")


if __name__ == "__main__":
    main()
