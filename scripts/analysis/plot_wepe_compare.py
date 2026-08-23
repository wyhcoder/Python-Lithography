"""对比多条 wEPE 收敛曲线 (例如 bisector vs xy)。

读取若干 errors.xlsx, 把 wEPE 随迭代的曲线画到同一张图中并区分。

用法:
    # 1) 默认: 工字型 bisector vs xy
    python scripts/analysis/plot_wepe_compare.py

    # 2) 自定义: 传 "标签=xlsx路径" (可多个), 可选 --out 指定输出前缀
    python scripts/analysis/plot_wepe_compare.py "bisector=路径A/errors.xlsx" "xy=路径B/errors.xlsx" --out outputs/diagnostics/wepe_pipeline

输出:
    <out>.png / .pdf / .svg  (默认 out = outputs/diagnostics/wepe_compare)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import DEFAULT

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

REPO = Path(__file__).resolve().parents[2]

# 默认序列 (标签, 路径)
# DEFAULT_SERIES = [
#     ("bisector_MEEF", REPO / "outputs/opc/MEEF_pipeline/BS/matrix_0525_k=5_sym=none_srafArc=5.0_BS_pipeline_test_bisector/errors.xlsx"),
#     ("xy_MEEF",       REPO / "outputs/opc/MEEF_pipeline/BS/matrix_0525_k=5_sym=none_srafArc=5.0_BS_pipeline_test_xy/errors.xlsx"),
#     ("bisector_gradient", REPO / "outputs/opc/MEEF_pipeline/BS/matrix_0525_k=5_sym=none_srafArc=5.0_BS_pipeline_test_bisector_gradient/errors.xlsx"),
#     ("xy_gradient", REPO / "outputs/opc/MEEF_pipeline/BS/matrix_0525_k=5_sym=none_srafArc=5.0_BS_pipeline_test_xy_gradient/errors.xlsx")
# ]
DEFAULT_SERIES = [
    ("bisector_MEEF",       REPO / "outputs/opc/MEEF_pipeline/BS/工字型_k=5_sym=none_srafArc=5.0_BS_pipeline_test_bisector/errors.xlsx"),
    ("xy_gradient", REPO / "outputs/opc/MEEF_pipeline/BS/工字型_k=5_sym=none_srafArc=5.0_BS_pipeline_test_xy_gradient/errors.xlsx"),
    ("xy_MEEF", REPO / "outputs/opc/MEEF_pipeline/BS/工字型_k=5_sym=none_srafArc=5.0_BS_pipeline_test_xy/errors.xlsx"),
]

# 颜色循环 (色相+饱和度高对比), 用于未在 LABEL_COLORS 中登记的标签
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

# 固定标签 -> 颜色映射: 4 种方法在色相环上互相远离, 便于分辨
# 红 / 蓝 / 绿 / 橙 —— 是 Matplotlib default (tab10) 的前 4 个, 对比强
LABEL_COLORS = {
    "bisector_MEEF":     "#d62728",   # 红
    "bisector_gradient": "#1f77b4",   # 蓝
    "xy_MEEF":           "#2ca02c",   # 绿
    "xy_gradient":       "#ff7f0e",   # 橙
}

# 固定标签 -> 线型/标记: 即便黑白打印也能区分
# MEEF (TSVD) 系列用实线, gradient 系列用虚线; bisector 用圆, xy 用方块
LABEL_STYLES = {
    "bisector_MEEF":     {"ls": "-",  "marker": "o", "markevery": 3},
    "bisector_gradient": {"ls": "--", "marker": "o", "markevery": 3},
    "xy_MEEF":           {"ls": "-",  "marker": "s", "markevery": 3},
    "xy_gradient":       {"ls": "--", "marker": "s", "markevery": 3},
}
DEFAULT_STYLE = {"ls": "-", "marker": None, "markevery": 3}


def _parse_args():
    """解析命令行: 'label=path' 列表 + 可选 --out 前缀。"""
    args = sys.argv[1:]
    out = REPO / "outputs" / "diagnostics" / "wepe_compare"
    series = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--out":
            out = Path(args[i + 1])
            i += 2
            continue
        if "=" in a:
            label, path = a.split("=", 1)
            series.append((label, Path(path)))
        i += 1
    if not series:
        series = [(lbl, p) for lbl, p in DEFAULT_SERIES]
    return series, out


def main():
    series, out = _parse_args()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    # 不同序列的最优点标注用不同偏移, 避免文字重叠
    annot_offsets = [(12, 20), (12, -14), (-80, 10), (-80, -24)]

    for k, (label, path) in enumerate(series):
        path = Path(path)
        # 优先用固定标签颜色/风格; 未登记的标签退回颜色循环
        color = LABEL_COLORS.get(label, COLORS[k % len(COLORS)])
        style = LABEL_STYLES.get(label, DEFAULT_STYLE)
        if not path.is_file():
            print(f"[warn] 缺少文件: {path}")
            continue
        df = pd.read_excel(path)
        # 只画 Iteration 1-50 (去掉第 0 行初值)
        df = df[(df["Iteration"] >= 1) & (df["Iteration"] <= 50)]
        it = df["Iteration"].to_numpy()
        wepe = df["wEPE"].to_numpy()
        ax.plot(it, wepe, color=color, lw=1.6,
                linestyle=style["ls"],
                marker=style["marker"], markersize=4,
                markevery=style["markevery"],
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.5,
                label=label, zorder=3)

        # 标出各自最优点 (不同偏移避免重叠)
        best_i = int(np.argmin(wepe))
        dx, dy = annot_offsets[k % len(annot_offsets)]
        ax.scatter([it[best_i]], [wepe[best_i]], s=32,
                   facecolor=color, edgecolor="white", lw=0.8, zorder=4)
        ax.annotate(f"{label} best={wepe[best_i]:.4f}",
                    xy=(it[best_i], wepe[best_i]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.8, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5))

    ax.set_xlabel("Iteration")
    ax.set_ylabel("wEPE")
    ax.set_title("wEPE convergence comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.4)

    fig.tight_layout(pad=0.4)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[done] saved: {out}.png / .pdf / .svg")


if __name__ == "__main__":
    main()
