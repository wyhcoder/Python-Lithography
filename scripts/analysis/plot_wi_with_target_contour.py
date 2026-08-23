"""画 wafer image (WI) 并把目标版图轮廓用红线叠加.

输入:
    - target_txt: 目标版图 txt (二值 0/1, np.loadtxt 可读)
    - wi_txt:     优化后的版图生成的晶圆像 txt (灰度, 一般 [0,1])
输出:
    - PNG 图: WI 用 inferno cmap, 目标版图轮廓用红线 (level=0.5) 叠加,
              带 colorbar. 图像尺寸/dpi 可调.

用法 (默认指向 MEEF_pipeline 的产物):
    python3 -m scripts.analysis.plot_wi_with_target_contour
或在脚本底部修改 TARGET_TXT / WI_TXT 路径自定义.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---- 默认输入 (按当前已有的产物路径) ----
DEFAULT_RESULT_DIR = ROOT / "outputs/opc" / "DPS" / "BS" / \
    "0.9_BS_matrix_0525_GPU"
DEFAULT_TARGET = DEFAULT_RESULT_DIR / "target_mask.txt"   # 支持 .txt / .bmp / .png
DEFAULT_WI_TXT = DEFAULT_RESULT_DIR / "wI_txt" / "latest.txt"   # 优化器最后一帧 wI


def _load_2d(path: Path) -> np.ndarray:
    """读 txt / bmp / png 为 2D ndarray; 报清晰错误.

    - .txt: np.loadtxt
    - .bmp / .png / .jpg: 走项目内 load_mask_image (返回二值 0/1)
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        arr = np.loadtxt(path)
    elif suffix in (".bmp", ".png", ".jpg", ".jpeg"):
        from litho_model.load_mask_bmp import load_mask_image
        # load_mask_image 接收 image_name (不带扩展名), 这里直接用底层 cv2 读
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        # 二值化到 0/1 (与 load_mask_image 行为一致)
        arr = (img > 127).astype(np.float32)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")
    if arr.ndim != 2:
        raise ValueError(f"{path} 应为 2D 数组, 实际 shape={arr.shape}")
    return arr


def plot_wi_with_target_contour(
    target_mask: np.ndarray,
    wi: np.ndarray,
    out_png: Path,
    cmap: str = "inferno",
    contour_level: float = 0.5,
    contour_color: str = "red",
    contour_lw: float = 1.4,
    title: Optional[str] = None,
    figsize=(7, 6),
    dpi: int = 140,
    show: bool = False,
):
    """画 WI + target 轮廓.

    Args:
        target_mask:    目标版图 (二值 0/1 或灰度); 取 contour_level 等高线.
        wi:             晶圆像 (灰度).
        out_png:        输出 PNG 路径.
        cmap:           imshow 配色, 默认 inferno.
        contour_level:  轮廓等高线值, 二值版图取 0.5 即可.
        contour_color:  轮廓颜色, 默认红.
        contour_lw:     轮廓线宽 (像素).
        title:          图标题, None 时按文件名自动生成.
        figsize/dpi:    matplotlib figure 参数.
        show:           是否 plt.show() (服务器跑通常 False).
    """
    # 形状不一致时, 自动 center crop / pad 到二者较小尺寸
    if target_mask.shape != wi.shape:
        H = min(target_mask.shape[0], wi.shape[0])
        W = min(target_mask.shape[1], wi.shape[1])
        print(f"[warn] shape mismatch target={target_mask.shape} wi={wi.shape}, "
              f"crop both to ({H}, {W})")

        def _ccrop(a):
            sy = (a.shape[0] - H) // 2
            sx = (a.shape[1] - W) // 2
            return a[sy:sy + H, sx:sx + W]

        target_mask = _ccrop(target_mask)
        wi = _ccrop(wi)

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(wi, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Wafer intensity", fontsize=10)

    # 用 marching squares 提目标版图等高线 (亚像素精度)
    contours = measure.find_contours(target_mask.astype(float), contour_level)
    for c in contours:
        # find_contours 返回 (row, col) = (y, x); imshow 需 (x, y)
        ax.plot(c[:, 1], c[:, 0], "-", color=contour_color,
                linewidth=contour_lw, alpha=0.95)

    if title is None:
        title = "Wafer Image with Target Contour"
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, default=None,
                    help="目标版图 txt 路径")
    ap.add_argument("--wi", type=Path, default=None,
                    help="晶圆像 wI txt 路径")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出 PNG 路径; 缺省时与 wI 同目录")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--level", type=float, default=0.5,
                    help="目标版图轮廓等高线值, 二值 mask 用 0.5")
    ap.add_argument("--no-show", action="store_true", default=True)
    args = ap.parse_args()


    target_path = args.target or DEFAULT_TARGET
    wi_path = args.wi or DEFAULT_WI_TXT
    out_png = args.out or (wi_path.parent / "wi_with_target_contour.png")

    print(f"[input] target = {target_path}")
    print(f"[input] wi     = {wi_path}")

    target = _load_2d(target_path)
    wi = _load_2d(wi_path)

    plot_wi_with_target_contour(
        target_mask=target,
        wi=wi,
        out_png=out_png,
        cmap=args.cmap,
        contour_level=args.level,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
