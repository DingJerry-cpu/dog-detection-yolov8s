"""按报告样式生成 2x3 训练曲线图（六个子图，中文标题，训练/验证同图对比）。

输出: runs/report/04_curves.png

用法: python tools/make_curves.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent.parent
RUN = HERE / "runs" / "dog_yolov8s"
OUT = HERE / "runs" / "report" / "04_curves.png"

CJK = Path(r"C:\Windows\Fonts\msyh.ttc")
if CJK.exists():
    font_manager.fontManager.addfont(str(CJK))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#cfd4dd"
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["grid.color"] = "#e6e9f0"
plt.rcParams["grid.linewidth"] = 0.9

C_TRAIN, C_VAL = "#2f6fed", "#f2994a"
C_P, C_R = "#2f6fed", "#f2994a"
C_M50, C_M95 = "#2f6fed", "#7fa8f0"
C_LR = "#27ae60"


def load_csv() -> dict[str, np.ndarray]:
    with open(RUN / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0]}


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8, color="#1b2333")
    ax.set_xlabel("训练轮次 (epoch)", fontsize=9, color="#5a6172")
    ax.set_ylabel(ylabel, fontsize=9, color="#5a6172")
    ax.tick_params(labelsize=8.5, colors="#5a6172")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")


def main() -> None:
    d = load_csv()
    ep = d["epoch"]
    best50 = d["metrics/mAP50(B)"].max()
    best_ep = int(ep[int(np.argmax(d["metrics/mAP50(B)"]))])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), dpi=150)
    fig.suptitle(
        f"YOLOv8s 狗检测 · 训练过程（{int(ep[-1])} 轮，最优 mAP50={best50:.3f} @第 {best_ep} 轮）",
        fontsize=15, fontweight="bold", y=0.975, color="#111827",
    )

    # 1 边界框损失
    ax = axes[0, 0]
    ax.plot(ep, d["train/box_loss"], color=C_TRAIN, lw=1.5, label="训练")
    ax.plot(ep, d["val/box_loss"], color=C_VAL, lw=1.5, label="验证")
    ax.axvline(70, color="#b0b7c6", ls=":", lw=1.0)
    style(ax, "边界框损失 (box loss)", "loss")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # 2 分类损失
    ax = axes[0, 1]
    ax.plot(ep, d["train/cls_loss"], color=C_TRAIN, lw=1.5, label="训练")
    ax.plot(ep, d["val/cls_loss"], color=C_VAL, lw=1.5, label="验证")
    style(ax, "分类损失 (cls loss)", "loss")
    ax.set_ylim(0, 6)
    ax.text(0.02, 0.94, "第 2 轮验证损失 80.8（热身期不稳定，已截断）", transform=ax.transAxes,
            fontsize=7.5, color="#a33a2a", va="top")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # 3 分布焦点损失
    ax = axes[0, 2]
    ax.plot(ep, d["train/dfl_loss"], color=C_TRAIN, lw=1.5, label="训练")
    ax.plot(ep, d["val/dfl_loss"], color=C_VAL, lw=1.5, label="验证")
    style(ax, "分布焦点损失 (dfl loss)", "loss")

    # 4 精确率与召回率
    ax = axes[1, 0]
    ax.plot(ep, d["metrics/precision(B)"], color=C_P, lw=1.5, label="精确率 P")
    ax.plot(ep, d["metrics/recall(B)"], color=C_R, lw=1.5, label="召回率 R")
    ax.set_ylim(0, 1.02)
    style(ax, "验证集精确率与召回率", "score")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # 5 检测精度
    ax = axes[1, 1]
    ax.plot(ep, d["metrics/mAP50(B)"], color=C_M50, lw=1.7, label="mAP@0.5")
    ax.plot(ep, d["metrics/mAP50-95(B)"], color=C_M95, lw=1.7, label="mAP@0.5:0.95")
    ax.axvline(best_ep, color="#e74c3c", ls="--", lw=1.0)
    ax.scatter([best_ep], [best50], s=70, marker="*", color="#f5a623", edgecolor="#8a5b00", lw=0.5, zorder=5)
    ax.annotate(f"最优 mAP50={best50:.3f}\n（第 {best_ep} 轮）", xy=(best_ep, best50),
                xytext=(0.04, 0.62), textcoords="axes fraction", fontsize=8, color="#a33a2a",
                arrowprops=dict(arrowstyle="-", color="#f0a99c", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.32", fc="#fff3f0", ec="#f0a99c", lw=0.7))
    ax.set_ylim(0, 1.02)
    style(ax, "检测精度 mAP", "mAP")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # 6 学习率
    ax = axes[1, 2]
    ax.plot(ep, d["lr/pg0"], color=C_LR, lw=1.7)
    style(ax, "学习率调度 (cos lr)", "lr")
    ax.get_legend().remove() if ax.get_legend() else None

    fig.tight_layout(rect=(0, 0.03, 1, 0.945), h_pad=2.0, w_pad=1.6)
    fig.text(
        0.5, 0.012,
        "注：第 70 轮关闭 mosaic 增强（close_mosaic=10），三个损失同时下跳、mAP 随之上抬，属正常现象。",
        ha="center", fontsize=9, color="#6b7280",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()

