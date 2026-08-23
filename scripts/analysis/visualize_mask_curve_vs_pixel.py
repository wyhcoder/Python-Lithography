"""
对比可视化: 主图形控制点的 (左) 理想曲线几何 vs (右) 项目 MSAA 渲染像素 mask.

设计:
    左图 = matplotlib 多边形填充, 显示控制点 B-spline 拟合的理想曲线几何,
           不经过任何离散化, 仅作"参考形状"用.
    右图 = 项目内 utils_model.demo_MSAA.AntiAliasRenderer.MSAA (经
           ParametricDemo.render_curve 调用), 是优化时 cost_compute
           实际看到的 257x257 像素化 mask, 灰度反映 16x 子像素覆盖率.

输入:
    EPE 或 WEPE 最优的控制点坐标 txt (路径见 main 默认值)
    格式: 每行一条闭合曲线, 一系列 "[x y]"

输出:
    outputs/opc/.../mask_curve_vs_pixel.png
"""
from pathlib import Path
import re
import sys
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import splev, splprep

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_model.demo_parametric import ParametricDemo

PathLike = Union[str, Path]


def _read_cps_txt_ragged(path: PathLike) -> List[np.ndarray]:
    """读取 cps txt -> list of (N_i, 2) ndarray.

    与项目内 read_cps_txt() 等价, 但不把外层强转 ndarray
    (新版 numpy 禁止 ragged ndarray, 会报 inhomogeneous shape).
    """
    curves: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            matches = re.findall(r"\[([^\]]+)\]", line)
            if not matches:
                continue
            curve = []
            for m in matches:
                x, y = map(float, m.split())
                curve.append([x, y])
            curves.append(np.array(curve, dtype=np.float64))
    return curves


def _fit_bspline_curves(cps_list, smoothing: float = 0.3, num_points: int = 400):
    """对每条 contour 拟合 B-spline, 返回采样后的曲线点列表 (用于左图绘制)."""
    fitted = []
    for contour in cps_list:
        contour = np.asarray(contour, dtype=np.float64)
        # 与 ParametricDemo.b_spile 对齐: 列 0 当作 y, 列 1 当作 x
        y, x = contour[:, 0], contour[:, 1]
        tck, _ = splprep([x, y], s=smoothing, per=True)
        u_fine = np.linspace(0, 1, num_points)
        x_fit, y_fit = splev(u_fine, tck)
        fitted.append((np.asarray(x_fit), np.asarray(y_fit)))
    return fitted


def visualize(
    cps_txt: PathLike,
    mask_shape: tuple = (257, 257),
    curve_type: str = "BS",
    out_path: Optional[PathLike] = None,
    fill_curve: bool = True,        # 左图: True 填充曲线内部, False 仅描边
):
    """生成左右对比图.

    Args:
        cps_txt:     主图形控制点 txt 路径.
        mask_shape:  右图 (像素 mask) 的 (H, W), 默认 257x257.
        curve_type:  曲线类型, 与 ParametricDemo 一致 (BS / BZ / OA).
        out_path:    输出 PNG; None -> cps_txt 同目录的 mask_curve_vs_pixel.png.
        fill_curve:  左图是否填充曲线内部 (False 只画轮廓线).
    """
    cps_txt = Path(cps_txt)
    if out_path is None:
        out_path = cps_txt.with_name("mask_curve_vs_pixel.png")
    out_path = Path(out_path)

    # --- 读取 cps ---
    cps_list = _read_cps_txt_ragged(cps_txt)
    n_contours = len(cps_list)
    n_pts_total = sum(len(c) for c in cps_list)
    print(f"[读取] {cps_txt}")
    print(f"       contours={n_contours}, total control points={n_pts_total}")

    # --- 左图数据: 与 ParametricDemo.b_spile 同款 B-spline 拟合得到连续曲线点 ---
    fitted_curves = _fit_bspline_curves(cps_list, smoothing=0.3, num_points=400)

    # --- 右图数据: 走项目 MSAA 渲染管线 ---
    # ParametricDemo.render_curve(cps) 内部:
    #   1) self.b_spile(cps) -> 200 个采样点 / contour 的 B-spline 曲线
    #   2) AntiAliasRenderer(msaa_level=16).MSAA(curves, mask_template, 'gray')
    #      -> 每像素 4x4=16 个子样本射线投射 -> 灰度覆盖率
    mask_template = np.zeros(mask_shape, dtype=np.float64)
    parametric = ParametricDemo(curve_type=curve_type, mask_template=mask_template)
    pixel_mask = parametric.render_curve(cps_list)
    pixel_mask = np.clip(pixel_mask, 0.0, 1.0)
    print(f"[渲染] MSAA pixel mask shape={pixel_mask.shape}, "
          f"foreground px (>0.5)={int((pixel_mask > 0.5).sum())}, "
          f"covered px (>0)={int((pixel_mask > 0).sum())}, "
          f"range=[{pixel_mask.min():.3f}, {pixel_mask.max():.3f}]")

    # --- 画图 ---
    H, W = mask_shape
    fig, axes = plt.subplots(
        1, 2,
        figsize=(2 * (W / 100) * 3.5, (H / 100) * 3.5),
        dpi=120,
    )

    # 左图: 理想曲线几何 (matplotlib 多边形填充, 不经离散化)
    ax_l = axes[0]
    ax_l.set_facecolor("#000000")
    for x_fit, y_fit in fitted_curves:
        if fill_curve:
            ax_l.fill(x_fit, y_fit, color="#ffffff", edgecolor="#ffffff",
                      linewidth=0.6, antialiased=True)
        else:
            ax_l.plot(np.append(x_fit, x_fit[0]),
                      np.append(y_fit, y_fit[0]),
                      color="#ffffff", linewidth=1.2, antialiased=True)
    ax_l.set_xlim(-0.5, W - 0.5)        # 与右图 imshow 默认坐标系完全对齐
    ax_l.set_ylim(H - 0.5, -0.5)         # 翻转 y 让坐标系与右图 imshow 一致
    ax_l.set_aspect("equal")
    ax_l.set_title(f"Curve mask ({curve_type} B-spline, matplotlib fill)\n"
                   f"{n_contours} contours, {n_pts_total} ctrl pts",
                   fontsize=10)
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    for spine in ax_l.spines.values():
        spine.set_visible(False)
    # 占位 axes: 给左图也分配一个与右图 colorbar 同宽的隐藏区, 让两图主体完全等大
    divider_l = make_axes_locatable(ax_l)
    cax_l = divider_l.append_axes("right", size="4%", pad=0.08)
    cax_l.axis("off")

    # 右图: 项目 MSAA 灰度覆盖率
    ax_r = axes[1]
    im = ax_r.imshow(pixel_mask, cmap="gray", vmin=0.0, vmax=1.0,
                     interpolation="nearest", aspect="equal")
    ax_r.set_title(f"Pixel mask (project AntiAliasRenderer.MSAA, 16x sub-pixel)\n"
                   f"{mask_shape[0]}x{mask_shape[1]}, "
                   f"covered={int((pixel_mask > 0).sum())} px "
                   f"(>0.5: {int((pixel_mask > 0.5).sum())})",
                   fontsize=10)
    ax_r.set_xticks([])
    ax_r.set_yticks([])
    for spine in ax_r.spines.values():
        spine.set_visible(False)
    # 关键: colorbar 用 axes_locatable 从外部分配空间, 不挤压右图主体宽度
    # (默认 fig.colorbar 会把 colorbar 塞进 ax_r 自己的 box, 让右图比左图窄)
    divider = make_axes_locatable(ax_r)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("MSAA coverage", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Main feature: ideal curve geometry vs MSAA pixel rendering",
                 fontsize=11, y=1.02)
    src_label = f"source: {cps_txt.parent.name}/{cps_txt.stem}.txt"
    fig.text(0.5, -0.02, src_label, ha="center", va="top", fontsize=8,
             color="#666666")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[保存] {out_path}")
    return out_path


if __name__ == "__main__":
    DEFAULT_CPS = (
        PROJECT_ROOT
        / "outputs/opc/DPS/BS/0.9_BS_matrix_0011_GPU/WEPE最优的控制点坐标.txt"
    )
    visualize(DEFAULT_CPS)
