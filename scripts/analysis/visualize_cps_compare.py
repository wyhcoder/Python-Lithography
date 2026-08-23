"""对比两组 "WEPE最优的控制点坐标.txt" 在同一张图上的分布 + B-spline 拟合曲线.

文件格式 (set_meef / set_sraf 等已有的写法):
    第 i 行 = 第 i 条 contour, 内含若干 "[y x]" 形式的点
    例:
        [159.75 97.16] [161.80 93.53] ... [...]
        [ 69.00 189.23] [ 69.01 187.00] ...

用法:
    python3 -m scripts.analysis.visualize_cps_compare
        默认对比 demo_meef 老结果 vs MEEF_pipeline 新结果
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---- 默认对比的两组 cps ----
PATH_OLD = ROOT / "outputs/opc" / "DPS" / "BS" / "0.9_BS_matrix_0011_GPU" / "WEPE最优的控制点坐标.txt"
PATH_NEW = ROOT / "outputs/opc" / "MEEF_pipeline" / "BS" / "matrix_0011_k=5_sym=none_srafArc=5.0_BS_pipeline_test" / "WEPE最优的控制点坐标.txt"

# 背景版图 (用 LSM mask 比 target 更直观, 能看到 SRAF)
BG_LSM = ROOT / "outputs/opc" / "CTM和levelset图像" / "Ls_mask" / "ls_imagematrix_0011.txt"


# ---------- 解析 ----------
_PT_RE = re.compile(r"\[\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\]")


def parse_cps_txt(path: Path) -> List[np.ndarray]:
    """解析 [y x] 形式 contour 文件, 返回 list[ndarray(N,2)] (按 [y, x] 排列)."""
    contours: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pts = [(float(m.group(1)), float(m.group(2))) for m in _PT_RE.finditer(line)]
            if pts:
                contours.append(np.asarray(pts, dtype=np.float64))
    return contours


# ---------- B-spline (闭合) ----------
def fit_closed_bspline(pts_yx: np.ndarray, smoothing: float = 0.3,
                       num_points: int = 400) -> np.ndarray:
    """周期 B-spline 拟合, 返回 (num_points, 2) 的 [y, x] 曲线."""
    if len(pts_yx) < 4:
        return pts_yx
    y, x = pts_yx[:, 0], pts_yx[:, 1]
    tck, _ = splprep([x, y], s=smoothing, per=True)
    u = np.linspace(0.0, 1.0, num_points)
    x_fit, y_fit = splev(u, tck)
    return np.stack([y_fit, x_fit], axis=1)


# ---------- 绘图 ----------
def main():
    cps_old = parse_cps_txt(PATH_OLD)
    cps_new = parse_cps_txt(PATH_NEW)
    print(f"[old] {PATH_OLD.name}: {len(cps_old)} contours, "
          f"total {sum(len(c) for c in cps_old)} pts")
    print(f"[new] {PATH_NEW.name}: {len(cps_new)} contours, "
          f"total {sum(len(c) for c in cps_new)} pts")

    # 背景: 优先用 LSM mask, 没有就给个空白
    if BG_LSM.exists():
        bg = np.loadtxt(BG_LSM)
    else:
        bg = np.zeros((256, 256))
        print(f"[warn] LSM mask not found at {BG_LSM}, using blank background")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)

    # 左: 控制点散点 + 拟合曲线 (双色叠加)
    ax = axes[0]
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
    color_old = "#e74c3c"   # 红: 老结果
    color_new = "#3498db"   # 蓝: 新结果

    for i, c in enumerate(cps_old):
        curve = fit_closed_bspline(c)
        ax.plot(curve[:, 1], curve[:, 0], "-", color=color_old, lw=2,
                label="old (DPS/0.9_BS) curve" if i == 0 else None)
        ax.scatter(c[:, 1], c[:, 0], s=18, c=color_old, edgecolors="white",
                   linewidths=0.5,
                   label="old CPs" if i == 0 else None, zorder=5)

    for i, c in enumerate(cps_new):
        curve = fit_closed_bspline(c)
        ax.plot(curve[:, 1], curve[:, 0], "-", color=color_new, lw=2,
                label="new (MEEF_pipeline) curve" if i == 0 else None)
        ax.scatter(c[:, 1], c[:, 0], s=18, c=color_new, edgecolors="white",
                   linewidths=0.5,
                   label="new CPs" if i == 0 else None, zorder=5)

    ax.set_title("Overlay: old vs new CPs + B-spline curves")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xlim(-0.5, bg.shape[1] - 0.5)
    ax.set_ylim(bg.shape[0] - 0.5, -0.5)
    ax.set_axis_off()

    # 右: 局部放大对比 (主图形 bbox + padding)
    ax = axes[1]
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
    all_pts = np.vstack([np.vstack(cps_old), np.vstack(cps_new)])
    pad = 6
    y0, x0 = all_pts.min(axis=0) - pad
    y1, x1 = all_pts.max(axis=0) + pad

    for c in cps_old:
        curve = fit_closed_bspline(c)
        ax.plot(curve[:, 1], curve[:, 0], "-", color=color_old, lw=2.2)
        ax.scatter(c[:, 1], c[:, 0], s=28, c=color_old, edgecolors="white",
                   linewidths=0.6, zorder=5)
    for c in cps_new:
        curve = fit_closed_bspline(c)
        ax.plot(curve[:, 1], curve[:, 0], "-", color=color_new, lw=2.2)
        ax.scatter(c[:, 1], c[:, 0], s=28, c=color_new, edgecolors="white",
                   linewidths=0.6, zorder=5)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # 图像 y 向下
    ax.set_title("Zoom-in (main pattern bbox)")
    ax.set_axis_off()

    plt.tight_layout()
    out_dir = ROOT / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compare_old_vs_new_cps.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
