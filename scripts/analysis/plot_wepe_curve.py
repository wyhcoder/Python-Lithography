"""
将多个 errors.xlsx 的 wEPE 损失曲线叠在一张 MATLAB 风格图中.

要点:
- 每条曲线从 Iteration >= 1 开始 (跳过 LSM 起点)
- 图例: pattern1 / pattern2 ... (按 XLSX_LIST 顺序)
- y 轴标题写 "EPE" (实际数据是 wEPE)
- 标题: "EPE loss"
- 关闭网格, MATLAB 风格边框 + ticks
- 紧凑布局

用法:
    python -m scripts.analysis.plot_wepe_curve
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# 要叠加绘制的所有 xlsx, 每条 (路径, 图例标签)
# 0011 -> pattern1, 0525 -> pattern2; 区分 cpu / gpu
XLSX_LIST = [
    ("outputs/opc/DPS/BS/0.9_BS_matrix_0011_cpu/errors.xlsx", "pattern1 (CPU)"),
    ("outputs/opc/DPS/BS/0.9_BS_matrix_0011_GPU/errors.xlsx", "pattern1 (GPU)"),
    ("outputs/opc/DPS/BS/0.9_BS_matrix_0525_cpu/errors.xlsx", "pattern2 (CPU)"),
    ("outputs/opc/DPS/BS/0.9_BS_matrix_0525_GPU/errors.xlsx", "pattern2 (GPU)"),
]

# 输出目录 (None 表示放第一个 xlsx 的同目录)
OUT_DIR = "outputs/diagnostics"
OUT_NAME = "wepe_curve_4runs.png"

# MATLAB 默认 7 色循环 (前两条用蓝/橙)
MATLAB_COLORS = [
    (0.0000, 0.4470, 0.7410),   # 蓝
    (0.8500, 0.3250, 0.0980),   # 橙
    (0.9290, 0.6940, 0.1250),   # 黄
    (0.4940, 0.1840, 0.5560),   # 紫
    (0.4660, 0.6740, 0.1880),   # 绿
    (0.3010, 0.7450, 0.9330),   # 青
    (0.6350, 0.0780, 0.1840),   # 暗红
]


def load_curve(xlsx_path: str | Path, max_iter: int | None = None):
    """读取 xlsx 并截取 1 <= Iteration <= max_iter 的部分."""
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"找不到 xlsx 文件: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    if "Iteration" not in df.columns or "wEPE" not in df.columns:
        raise ValueError(
            f"xlsx 缺少必要列, 当前列: {list(df.columns)}, "
            f"应包含 'Iteration' 与 'wEPE'."
        )
    mask = df["Iteration"] >= 1
    if max_iter is not None:
        mask &= df["Iteration"] <= max_iter
    df_run = df[mask].reset_index(drop=True)
    return df_run["Iteration"].to_numpy(), df_run["wEPE"].to_numpy(), xlsx_path


def _style_for_label(label: str, pattern_color_map: dict) -> dict:
    """根据图例标签自动决定颜色 (pattern) 与线型/marker (CPU/GPU)。"""
    # 颜色按 pattern 分组 (相同 pattern 用相同颜色)
    pname = label.split()[0]   # 'pattern1' / 'pattern2'
    if pname not in pattern_color_map:
        pattern_color_map[pname] = MATLAB_COLORS[
            len(pattern_color_map) % len(MATLAB_COLORS)
        ]
    color = pattern_color_map[pname]

    # 线型/marker 按设备区分
    if "GPU" in label.upper():
        linestyle = "--"
        marker = "s"     # 方块
    else:
        linestyle = "-"
        marker = "o"     # 圆点
    return {"color": color, "linestyle": linestyle, "marker": marker}


def plot_multi_wepe(xlsx_list: list,
                    max_iter: int | None = None) -> None:
    # xlsx_list: list of (path, label) 或 list of path (兼容老格式)
    normalized = []
    for item in xlsx_list:
        if isinstance(item, (tuple, list)):
            normalized.append((item[0], item[1]))
        else:
            normalized.append((item, None))

    curves = []
    for path, label in normalized:
        iters, wepe, p = load_curve(path, max_iter=max_iter)
        curves.append((iters, wepe, p, label))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    iters_min = +10**9
    iters_max = -10**9
    wepe_min = +10**9
    wepe_max = -10**9

    pattern_color_map: dict = {}
    for idx, (iters, wepe, _, label) in enumerate(curves):
        if label is None:
            label = f"pattern{idx + 1}"
        style = _style_for_label(label, pattern_color_map)
        ax.plot(iters, wepe,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                marker=style["marker"],
                markersize=4,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=label)
        iters_min = min(iters_min, int(iters.min()))
        iters_max = max(iters_max, int(iters.max()))
        wepe_min = min(wepe_min, float(wepe.min()))
        wepe_max = max(wepe_max, float(wepe.max()))

    # 坐标轴
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("EPE", fontsize=11)
    ax.set_title("EPE loss", fontsize=12)

    # 关闭网格
    ax.grid(False)

    # MATLAB 风格边框 + 内向 ticks (上右也加)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", which="both",
                   direction="in", length=4, width=0.8,
                   top=True, right=True,
                   labelsize=10, colors="black")

    # x 轴整数刻度 (最多 ~12 个), 末尾刻度若与 iters_max 距离过近, 用 iters_max 替换
    span = iters_max - iters_min + 1
    step = max(1, span // 12)
    xticks = list(range(iters_min, iters_max + 1, step))
    if xticks[-1] != iters_max:
        if iters_max - xticks[-1] < step // 2 + 1:
            xticks[-1] = iters_max
        else:
            xticks.append(iters_max)
    ax.set_xticks(xticks)

    # 紧凑边距
    ax.set_xlim(iters_min - 0.5, iters_max + 0.5)
    y_pad = (wepe_max - max(0.0, wepe_min)) * 0.05
    ax.set_ylim(max(0.0, wepe_min - y_pad), wepe_max + y_pad)

    # MATLAB 风图例: 黑边白底, 不带阴影
    leg = ax.legend(loc="upper right", fontsize=10, frameon=True,
                    edgecolor="black", framealpha=1.0)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout(pad=0.4)

    # 输出到指定目录 (默认第一个 xlsx 同目录)
    if OUT_DIR is not None:
        out_dir = Path(OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(curves[0][2]).parent
    out_path = out_dir / OUT_NAME
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"✓ 已保存: {out_path}")
    for idx, (iters, wepe, src, label) in enumerate(curves):
        tag = label if label is not None else f"pattern{idx + 1}"
        print(f"  [{tag}] {src}")
        print(f"      iter {iters.min()}..{iters.max()}, "
              f"start={wepe[0]:.4f}, end={wepe[-1]:.4f}, "
              f"min={wepe.min():.4f} @ iter {iters[wepe.argmin()]}")


if __name__ == "__main__":
    # 所有曲线统一截到 iter <= MAX_ITER, 让对齐对比更公平
    MAX_ITER = 40
    plot_multi_wepe(XLSX_LIST, max_iter=MAX_ITER)
