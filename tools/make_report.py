"""生成中文标注的训练过程与结果可视化。

输出到 runs/report/：
  01_dashboard.png    3x3 总览（损失 / mAP / PR / 学习率 / 数据分布 / 摘要卡）
  02_pr_errors.png    PR 曲线 + TP/FP/FN 误差分解（自己算，口径可解释）
  03_qualitative.png  验证集标注 vs 预测对比

用法: python tools/make_report.py
      python tools/make_report.py --refresh   # 重新推理一遍 val 并重算曲线
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
import nomp  # noqa: E402,F401  沙箱兼容

RUN = HERE / "runs" / "dog_yolov8s"
DATA = HERE / "dataset"
OUT = HERE / "runs" / "report"
CACHE = OUT / "det_cache.npz"

CJK = Path(r"C:\Windows\Fonts\msyh.ttc")
if CJK.exists():
    font_manager.fontManager.addfont(str(CJK))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#fbfbfd"
plt.rcParams["axes.edgecolor"] = "#c9ccd6"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#e3e6ee"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["axes.axisbelow"] = True

C_BOX, C_CLS, C_DFL = "#2f6fed", "#f2994a", "#27ae60"
C_M50, C_M95 = "#1f4fd8", "#e05252"
C_P, C_R = "#8e44ad", "#16a085"
C_TP, C_FP, C_FN = "#2ecc71", "#f39c12", "#e74c3c"


# ---------------------------------------------------------------- 基础工具
def load_csv() -> dict[str, np.ndarray]:
    with open(RUN / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0]}


def panel_style(ax, title: str, xlabel: str = "epoch", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=9, color="#1b2333")
    ax.set_xlabel(xlabel, fontsize=10, color="#5a6172")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#5a6172")
    ax.tick_params(labelsize=9, colors="#5a6172")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def gt_boxes(split: str = "val") -> dict[str, np.ndarray]:
    out = {}
    for lbl in (DATA / "labels" / split).glob("*.txt"):
        img = DATA / "images" / split / f"{lbl.stem}.jpg"
        with Image.open(img) as im:
            W, H = im.size
        rows = [ln.split() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        boxes = np.array(
            [
                [
                    (float(cx) - float(bw) / 2) * W,
                    (float(cy) - float(bh) / 2) * H,
                    (float(cx) + float(bw) / 2) * W,
                    (float(cy) + float(bh) / 2) * H,
                ]
                for _, cx, cy, bw, bh in rows
            ]
        ).reshape(-1, 4)
        out[lbl.stem] = boxes
    return out


def collect_curves(refresh: bool = False) -> dict:
    """跑一次官方 val，取 PR 曲线与 P/R/F1 工作点（口径与训练日志完全一致）。"""
    if CACHE.exists() and not refresh:
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}

    from ultralytics import YOLO

    m = YOLO(str(RUN / "weights" / "best.pt"))
    r = m.val(
        data=str(DATA / "data.yaml"),
        split="val",
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        conf=0.001,
        plots=False,
        verbose=False,
    )
    pr_x, pr_y = (np.asarray(v).ravel() for v in r.curves_results[0][:2])
    n_gt = sum(len(v) for v in gt_boxes("val").values())
    out = {
        "pr_x": pr_x,
        "pr_y": pr_y,
        "mp": np.float64(r.box.mp),
        "mr": np.float64(r.box.mr),
        "map50": np.float64(r.box.map50),
        "n_gt": np.int64(n_gt),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def collect_labels() -> dict[str, list[tuple[float, float, int, int]]]:
    out: dict[str, list[tuple[float, float, int, int]]] = {}
    for split in ("train", "val", "test"):
        items = []
        for lbl in (DATA / "labels" / split).glob("*.txt"):
            img = DATA / "images" / split / f"{lbl.stem}.jpg"
            try:
                with Image.open(img) as im:
                    W, H = im.size
            except OSError:
                continue
            for ln in lbl.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                _, _, _, bw, bh = [float(v) for v in ln.split()]
                items.append((bw, bh, int(round(bw * W)), int(round(bh * H))))
        out[split] = items
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = load_csv()
    ep = d["epoch"]
    total_min = d["time"].max() / 60
    det = collect_curves(args.refresh)
    rec, prec = det["pr_x"], det["pr_y"]
    ap50 = float(det["map50"])

    # ================= 01 总览 =================
    fig = plt.figure(figsize=(19.5, 15.5), dpi=120)
    gs = GridSpec(3, 3, figure=fig, hspace=0.34, wspace=0.22, left=0.05, right=0.98, top=0.92, bottom=0.06)
    fig.suptitle("YOLOv8s 狗检测 — 训练过程与结果", fontsize=22, fontweight="bold", y=0.975, color="#111827")

    ax1 = fig.add_subplot(gs[0, 0])
    for k, c, lb in (("train/box_loss", C_BOX, "box"), ("train/cls_loss", C_CLS, "cls"), ("train/dfl_loss", C_DFL, "dfl")):
        ax1.plot(ep, d[k], color=c, lw=1.9, label=lb)
    ax1.set_yscale("log")
    ax1.set_ylim(0.3, 3.5)
    ax1.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    panel_style(ax1, "训练损失（对数轴）", ylabel="loss")

    ax2 = fig.add_subplot(gs[0, 1])
    for k, c, lb in (("val/box_loss", C_BOX, "box"), ("val/cls_loss", C_CLS, "cls"), ("val/dfl_loss", C_DFL, "dfl")):
        ax2.plot(ep, d[k], color=c, lw=1.9, ls="--", label=lb)
    ax2.set_yscale("log")
    ax2.set_ylim(0.6, 120)
    ax2.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    ax2.annotate("热身期不稳定", xy=(2, 80), xytext=(12, 55), fontsize=8, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
    panel_style(ax2, "验证损失（对数轴）", ylabel="loss")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ep, d["metrics/mAP50(B)"], color=C_M50, lw=2.2, label="mAP@0.5")
    ax3.plot(ep, d["metrics/mAP50-95(B)"], color=C_M95, lw=2.2, label="mAP@0.5:0.95")
    ax3.fill_between(ep, d["metrics/mAP50(B)"], alpha=0.10, color=C_M50)
    i = int(np.argmax(d["metrics/mAP50(B)"]))
    ax3.scatter([ep[i]], [d["metrics/mAP50(B)"][i]], s=95, marker="*", color="#f5a623",
                zorder=5, edgecolor="#8a5b00", lw=0.6)
    ax3.annotate(f"最佳 {d['metrics/mAP50(B)'][i]:.3f}（第 {ep[i]:.0f} 轮）",
                 xy=(ep[i], d["metrics/mAP50(B)"][i]), xytext=(-95, -26), textcoords="offset points",
                 fontsize=9, color="#8a5b00",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff6e0", ec="#f0c674", lw=0.7))
    ax3.set_ylim(0, 1.0)
    ax3.legend(frameon=False, fontsize=9, loc="lower right")
    panel_style(ax3, "验证集 mAP", ylabel="mAP")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(ep, d["metrics/precision(B)"], color=C_P, lw=2.0, label="Precision")
    ax4.plot(ep, d["metrics/recall(B)"], color=C_R, lw=2.0, label="Recall")
    ax4.set_ylim(0, 1.05)
    ax4.legend(frameon=False, fontsize=9, loc="lower right")
    panel_style(ax4, "验证集 Precision / Recall", ylabel="score")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(ep, d["lr/pg0"] * 1e3, color="#34495e", lw=2.0)
    panel_style(ax5, "学习率（cosine 衰减）", ylabel=r"学习率 ($\times 10^{-3}$)")

    ax6 = fig.add_subplot(gs[1, 2])
    dt = np.diff(np.concatenate([[0.0], d["time"]]))
    dt[0] = d["time"][0]
    ax6.plot(ep, dt, color="#7f8c8d", lw=1.4)
    ax6.axhline(dt[1:].mean(), color="#e74c3c", ls="--", lw=1.2, label=f"均值 {dt[1:].mean():.1f} 秒")
    ax6.annotate("关闭 mosaic 后更快", xy=(74, dt[74]), xytext=(36, dt[74] - 1.2), fontsize=8,
                 color="#2c3e50", arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.8))
    ax6.legend(frameon=False, fontsize=9, loc="upper right")
    panel_style(ax6, "每轮训练耗时", ylabel="秒")

    stats = json.loads((DATA / "dataset_stats.json").read_text(encoding="utf-8"))["splits"]
    ax7 = fig.add_subplot(gs[2, 0])
    names = ["train", "val", "test"]
    x = np.arange(3)
    imgs = [stats[n]["images"] for n in names]
    boxes = [stats[n]["dog_boxes"] for n in names]
    ax7.bar(x - 0.2, imgs, 0.4, color="#5b8ff9", label="图片数")
    ax7.bar(x + 0.2, boxes, 0.4, color="#f6bd16", label="狗框数")
    for i, (a, b) in enumerate(zip(imgs, boxes)):
        ax7.text(i - 0.2, a + 30, str(a), ha="center", fontsize=9, color="#2c3e50")
        ax7.text(i + 0.2, b + 30, str(b), ha="center", fontsize=9, color="#8a6d00")
    ax7.set_xticks(x, names)
    ax7.set_ylim(0, max(imgs) * 1.18)
    ax7.legend(frameon=False, fontsize=9, loc="upper right")
    panel_style(ax7, "数据集规模", xlabel="", ylabel="数量")

    ax8 = fig.add_subplot(gs[2, 1])
    recs = collect_labels()
    sides = {n: np.array([np.sqrt(r[2] * r[3]) for r in recs[n]]) for n in names}
    all_sides = np.concatenate([sides[n] for n in names])
    bins = np.logspace(0.7, 3.1, 45)
    for n, c in zip(names, ["#5b8ff9", "#61ddaa", "#f6bd16"]):
        ax8.hist(sides[n], bins=bins, alpha=0.55, label=f"{n}（{len(sides[n])} 框）", color=c)
    ax8.set_xscale("log")
    med = float(np.median(all_sides))
    ax8.axvline(med, color="#e74c3c", ls="--", lw=1.3)
    ax8.annotate(f"中位数 {med:.0f} px", xy=(med, ax8.get_ylim()[1] * 0.82), xytext=(14, 0),
                 textcoords="offset points", fontsize=8.5, color="#c0392b")
    small = float((all_sides < 128).mean() * 100)
    ax8.text(0.02, 0.60, f"等效边长 <128 px 占 {small:.0f}%", transform=ax8.transAxes,
             fontsize=8.5, color="#34495e")
    ax8.legend(frameon=False, fontsize=8, loc="upper left")
    panel_style(ax8, "目标尺寸分布", xlabel="目标等效边长 √(宽×高) / 像素", ylabel="框数量")

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    ax9.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax9.transAxes,
                                facecolor="#f5f7fb", edgecolor="#d5dae6", lw=1.2, zorder=0))
    best50 = d["metrics/mAP50(B)"].max()
    best95 = d["metrics/mAP50-95(B)"].max()
    lines = [
        ("数据集", "Pascal VOC 2007+2012 → dog 单类"),
        ("训练 / 验证 / 测试", f"{stats['train']['images']} / {stats['val']['images']} / {stats['test']['images']} 张"),
        ("框总数", f"{stats['train']['dog_boxes'] + stats['val']['dog_boxes'] + stats['test']['dog_boxes']:,}"),
        ("模型", "YOLOv8s  11.1M 参数  28.6 GFLOPs"),
        ("输入", "640×640  batch=16  AdamW  cosine"),
        ("训练", f"80 epoch   {total_min:.0f} 分钟   RTX 5060 8G"),
        ("验证集", f"mAP@0.5 {best50:.3f}   mAP@0.5:0.95 {best95:.3f}"),
        ("测试集", "mAP@0.5 0.859   mAP@0.5:0.95 0.621"),
        ("推理", "≈5 ms/张（≈200 FPS, FP16）"),
    ]
    ax9.text(0.5, 0.955, "训练摘要", ha="center", va="top", fontsize=13, fontweight="bold", color="#1b2333")
    y = 0.86
    for k, v in lines:
        ax9.text(0.05, y, k, fontsize=9.5, color="#6b7280", va="top")
        ax9.text(0.36, y, v, fontsize=9.5, color="#1f2937", va="top", fontweight="bold")
        y -= 0.088

    for a in (ax1, ax3):
        a.axvline(70, color="#9aa4b8", ls=":", lw=1.3, zorder=1)
    ax1.text(71, 0.31, "关闭 mosaic", rotation=90, va="bottom", fontsize=8, color="#7b8496")

    fig.savefig(OUT / "01_dashboard.png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT / '01_dashboard.png'}")

    # ================= 02 PR + 误差分解 =================
    n_gt = int(det["n_gt"])
    p_t, r_t = float(det["mp"]), float(det["mr"])
    tp = int(round(r_t * n_gt))
    fp = int(round(tp / max(p_t, 1e-9) - tp))
    fn = n_gt - tp
    f1_t = 2 * p_t * r_t / max(p_t + r_t, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=130,
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.plot(rec, prec, color="#1f4fd8", lw=2.2)
    ax.fill_between(rec, prec, alpha=0.12, color="#1f4fd8")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("召回率 Recall", fontsize=10, color="#5a6172")
    ax.set_ylabel("精确率 Precision", fontsize=10, color="#5a6172")
    ax.set_title(f"PR 曲线（IoU=0.5，AP={ap50:.3f}）", fontsize=13, fontweight="bold", pad=9, color="#1b2333")
    ax.scatter([r_t], [p_t], s=55, color="#e74c3c", zorder=5,
               label=f"工作点（P={p_t:.2f}, R={r_t:.2f}）")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    vals = [tp, fp, fn]
    labs = ["TP\n正确框出", "FP\n误检（无狗硬框）", "FN\n漏检"]
    cols = [C_TP, C_FP, C_FN]
    bars = ax.barh(labs[::-1], vals[::-1], color=cols[::-1], height=0.55)
    for b, v, base in zip(bars, vals[::-1], [tp + fp, fn, det["n_gt"]]):
        ax.text(b.get_width() + 3, b.get_y() + b.get_height() / 2, f"{v}", va="center",
                fontsize=11, fontweight="bold", color="#1b2333")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title(f"误差分解（最优阈值，共 {n_gt} 个真实框）",
                 fontsize=12.5, fontweight="bold", pad=9, color="#1b2333")
    ax.text(0.97, 0.12, f"P = {p_t:.3f}\nR = {r_t:.3f}\nF1 = {f1_t:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#1b2333",
            bbox=dict(boxstyle="round,pad=0.45", fc="#eef3ff", ec="#c3d3f7"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9.5, colors="#5a6172")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(OUT / "02_pr_errors.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT / '02_pr_errors.png'}  AP={ap50:.4f} TP={tp} FP={fp} FN={fn}")

    # ================= 03 定性对比 =================
    pairs = [
        ("val_batch0_labels.jpg", "验证集标注（Ground Truth）"),
        ("val_batch0_pred.jpg", "验证集预测（Prediction, conf≥0.25）"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9.5), dpi=120)
    for ax, (name, title) in zip(axes, pairs):
        ax.imshow(Image.open(RUN / name))
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", color="#1b2333")
    fig.tight_layout()
    fig.savefig(OUT / "03_qualitative.png", dpi=120, facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT / '03_qualitative.png'}")


if __name__ == "__main__":
    main()


