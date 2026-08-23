"""绘制 SRAF 优化 CMA-ES 损失曲线（Nature 风格）。

用法:
    python scripts/analysis/plot_cost_curve.py [cost_curve_iter_1.csv]

未传参数时默认使用:
    outputs/opc/.../骨架线参数化_宽度=2.5_分阶=是_PVBand=是_固定宽度=否_BS_对角通孔/cost_curve_iter_1.csv

输出 (与输入 CSV 同目录):
    cost_curve_iter_1_fig.pdf
    cost_curve_iter_1_fig.svg
    cost_curve_iter_1_fig.png   (600 dpi 预览)

设计要点:
  - 核心结论: CMA-ES 在约第 50 次评估后进入最优盆地, 后续仅在该盆地内做微小抖动 -> 已收敛.
  - 主图 (hero): 单面板, 显示每次评估的 cost 散点 + running-min 折线.
  - 内嵌残差子图: log 轴显示 (cost - cost_min) 体现末段抖动幅度.
  - 配色: 中性低饱和, 单一信号家族 + 一个 accent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# 全局风格 (Nature/sans-serif, 可编辑字体, 细轴线)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


# ---------------------------------------------------------------------------
# 调色板 (中性低饱和)
# ---------------------------------------------------------------------------
COLOR_SCATTER = "#7f8c9b"   # 灰蓝, 单次评估
COLOR_RUNMIN  = "#2f4858"   # 深墨蓝, 运行最小值 (hero)
COLOR_ACCENT  = "#c0524a"   # 砖红, 收敛标注
COLOR_BAND    = "#dfe5ec"   # 浅灰带, 末段抖动盆地


def _resolve_csv() -> Path:
    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    default = repo_root / (
        "outputs/opc/sraf_width_opt/中期检查/"
        "骨架线参数化_宽度=2.5_分阶=是_PVBand=是_固定宽度=否_BS_对角通孔/"
        "cost_curve_iter_1.csv"
    )
    return default


def _detect_convergence(cost: np.ndarray, frac_tol: float = 0.01) -> int:
    """以 running-min 距 final-best 的相对距离 < frac_tol 作为'进入最优盆地'."""
    run_min = np.minimum.accumulate(cost)
    best = float(run_min[-1])
    span = float(run_min[0] - best) + 1e-12
    rel = (run_min - best) / span     # 1 -> 0 的归一化下降
    idx = int(np.argmax(rel <= frac_tol))
    return max(idx, 1)


def main():
    csv_path = _resolve_csv()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    eval_id = df["eval_id"].to_numpy()
    cost = df["cost"].to_numpy()
    run_min = np.minimum.accumulate(cost)

    cost_init = float(cost[0])
    cost_best = float(run_min[-1])
    converge_idx = _detect_convergence(cost, frac_tol=0.02)
    converge_eval = int(eval_id[converge_idx])

    # ----- 单栏 hero 面板, 单图 + 内嵌残差子图 -----
    # 单栏宽度 ~89 mm = 3.5 inch; 高度按黄金比例
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    # 1. 末段盆地阴影带 (running-min ± 残差范围) - 视觉上提示"抖动区"
    tail = run_min[converge_idx:]
    band_low, band_high = float(tail.min()), float(cost[converge_idx:].max())
    ax.axhspan(band_low, band_high,
               xmin=converge_idx / len(cost), xmax=1.0,
               color=COLOR_BAND, alpha=0.55, zorder=0,
               linewidth=0)

    # 2. 单次评估散点 (受噪声驱动)
    ax.scatter(eval_id, cost,
               s=4, c=COLOR_SCATTER, alpha=0.45,
               linewidths=0,
               label="Per-evaluation cost",
               zorder=2)

    # 3. running-min (核心趋势线)
    ax.plot(eval_id, run_min,
            color=COLOR_RUNMIN, lw=1.4,
            label="Running minimum",
            zorder=3)

    # 4. 收敛点竖虚线 + 顶部文字标注（标注挪到坐标轴顶部, 不再遮挡曲线）
    ax.axvline(converge_eval, color=COLOR_ACCENT,
               lw=0.8, ls=(0, (3, 2)), zorder=2.5)

    # 计算坐标范围, 给标注预留顶部空间
    y_min_plot = max(cost_best - 0.4, 0)
    y_max_plot = max(cost_init, band_high) * 1.18  # 多留 18% 顶部空间给标注
    ax.set_xlim(0, eval_id[-1] + 5)
    ax.set_ylim(y_min_plot, y_max_plot)

    # 收敛点文字: 置于顶部, 避开数据区
    ax.text(converge_eval + 4, y_max_plot * 0.96,
            f"converged ≈ eval {converge_eval}",
            fontsize=6.5, color=COLOR_ACCENT, va="top", ha="left")

    # 初值 / 终值用紧凑的圆点 + 偏左/偏右文字, 不压住散点
    ax.scatter([eval_id[0]], [cost_init], s=18,
               facecolor="white", edgecolor=COLOR_RUNMIN, lw=0.9, zorder=4)
    ax.scatter([eval_id[-1]], [cost_best], s=18,
               facecolor=COLOR_ACCENT, edgecolor="white", lw=0.6, zorder=4)
    # start 文字放在散点上方
    ax.annotate(f"start  {cost_init:.3f}",
                xy=(eval_id[0], cost_init),
                xytext=(8, 8), textcoords="offset points",
                fontsize=6.2, color=COLOR_RUNMIN, va="bottom", ha="left")
    # best 文字放在散点下方右对齐
    ax.annotate(f"best  {cost_best:.4f}",
                xy=(eval_id[-1], cost_best),
                xytext=(-6, -6), textcoords="offset points",
                fontsize=6.2, color=COLOR_ACCENT, va="top", ha="right")

    ax.set_xlabel("Objective evaluation")
    ax.set_ylabel("Joint cost")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # legend 放右上
    ax.legend(loc="upper right", fontsize=6.5,
              handlelength=1.6, borderaxespad=0.4)

    fig.tight_layout(pad=0.4)

    out_stem = csv_path.with_suffix("").as_posix() + "_fig"
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] saved:")
    print(f"  {out_stem}.pdf")
    print(f"  {out_stem}.svg")
    print(f"  {out_stem}.png")
    print(f"[info] start={cost_init:.4f}, best={cost_best:.4f}, "
          f"converge_eval={converge_eval}, n_eval={len(cost)}")


if __name__ == "__main__":
    main()
