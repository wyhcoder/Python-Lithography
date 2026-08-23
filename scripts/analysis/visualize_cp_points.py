"""
可视化控制点 (cp_points.txt) 矩阵 —— 无白边输出.

输入:
    cp_points.txt  二维整数矩阵 (空格分隔):
        0       = 背景
        1       = 主图形/SRAF 轮廓
        cp_value(默认=6) = 控制点
输出:
    cp_points_viz.png  与矩阵分辨率 1:1 的三色图 (背景/轮廓/控制点),
    无任何白边/坐标轴.
"""
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

PathLike = Union[str, Path]


def visualize_cp_points(
    txt_path: PathLike,
    out_path: Optional[PathLike] = None,
    cp_value: int = 6,                 # 控制点的标记值
    bg_color: str = "#000000",         # 黑色背景 (值=0)
    contour_color: str = "#404040",    # 暗灰色轮廓 (值=1)
    cp_color: str = "#ffffff",         # 白色控制点 (值=cp_value)
    upscale: int = 1,                  # 输出分辨率倍数, 默认 1:1
) -> Path:
    """读取 cp_points.txt 并保存无白边可视化图.

    Args:
        txt_path: 输入整数矩阵 txt 路径.
        out_path: 输出 PNG 路径; None 则保存到输入同目录 cp_points_viz.png.
        cp_value: 哪个值算控制点 (默认 6).
        bg_color: 背景 (值=0) 颜色.
        contour_color: 轮廓 (值=1) 颜色.
        cp_color: 控制点 (值=cp_value) 颜色.
        upscale:  输出像素 = 矩阵尺寸 * upscale (整数, ≥1).

    Returns:
        实际保存的输出路径.
    """
    txt_path = Path(txt_path)
    if out_path is None:
        out_path = txt_path.with_name("cp_points_viz.png")
    out_path = Path(out_path)

    # --- 读矩阵 ---
    mat = np.loadtxt(txt_path, dtype=np.int32)
    if mat.ndim != 2:
        raise ValueError(f"期望二维矩阵, 实际 shape={mat.shape}")
    h, w = mat.shape
    n_contour = int((mat == 1).sum())
    n_cp = int((mat == cp_value).sum())
    print(f"[读取] {txt_path}  shape=({h},{w})  "
          f"轮廓(==1)={n_contour}, 控制点(=={cp_value})={n_cp}")

    # --- 三类映射: 0=背景, 1=轮廓, 2=控制点 ---
    label = np.zeros_like(mat, dtype=np.uint8)
    label[mat == 1] = 1
    label[mat == cp_value] = 2          # 控制点优先级最高 (覆盖轮廓)
    cmap = ListedColormap([bg_color, contour_color, cp_color])

    # --- 关键: figsize/dpi 完全对齐像素, 关闭坐标轴 + 0 边距 ---
    upscale = max(1, int(upscale))
    out_h, out_w = h * upscale, w * upscale
    dpi = 100
    fig = plt.figure(
        figsize=(out_w / dpi, out_h / dpi),
        dpi=dpi,
        frameon=False,
    )
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))   # 占满整张 figure, 无 margin
    ax.set_axis_off()
    ax.imshow(
        label,
        cmap=cmap,
        vmin=0,
        vmax=2,                          # 锁死 colormap 索引, 防止只有两类时映射偏移
        interpolation="nearest",         # 保持像素硬边
        aspect="equal",
    )
    fig.savefig(
        out_path,
        dpi=dpi,
        pad_inches=0,
        bbox_inches=None,
    )
    plt.close(fig)

    print(f"[保存] {out_path}  ({out_w}x{out_h})")
    return out_path


if __name__ == "__main__":
    # 默认目标: 你提到的那张 cp_points
    DEFAULT_TXT = (
        Path(__file__).resolve().parents[2]
        / "outputs/opc/DPS/BS/0.9_BS_matrix_0525_cpu/cp_points.txt"
    )
    visualize_cp_points(DEFAULT_TXT)
