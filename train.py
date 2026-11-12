"""YOLOv8s 单类（dog）检测训练脚本。

    python train.py                      # 默认 100 epoch, 640, batch 16
    python train.py --epochs 50 --batch 8
    python train.py --smoke              # 2 epoch 快速跑通流程
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import nomp  # noqa: E402,F401  沙箱兼容：ThreadPool -> 串行（必须在 ultralytics 之前）


def parse() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(here / "weights" / "yolov8s.pt"), help="预训练权重")
    ap.add_argument("--data", default=str(here / "dataset" / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="8G 显存建议 16；OOM 就调成 8")
    ap.add_argument("--device", default="0", help="0 或 cpu")
    ap.add_argument("--workers", type=int, default=0, help="Windows 下 0 最稳")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--project", default=str(here / "runs"))
    ap.add_argument("--name", default="dog_yolov8s")
    ap.add_argument("--smoke", action="store_true", help="2 epoch 冒烟测试")
    return ap.parse_args()


def main() -> None:
    args = parse()
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=2 if args.smoke else args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name + ("_smoke" if args.smoke else ""),
        exist_ok=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        plots=True,
        seed=0,
        deterministic=False,
        val=True,
        pretrained=True,
    )

    # 训练结束自动在验证集 + 测试集上评估，并打印指标
    best = Path(args.project) / (args.name + ("_smoke" if args.smoke else "")) / "weights" / "best.pt"
    if best.exists():
        m = YOLO(str(best))
        for split in ("val", "test"):
            metrics = m.val(
                data=args.data,
                split=split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=0,
            )
            print(
                f"[{split}] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f} "
                f"P={metrics.box.mp:.4f} R={metrics.box.mr:.4f}"
            )


if __name__ == "__main__":
    mp.freeze_support()
    main()
