"""对比同一张 mask/wi 在不同 colormap 下的视觉效果。

用法:
    python scripts/analysis/cmap_compare.py [可选: 输入文件路径]

支持的输入文件:
    - .txt   : 通过 np.loadtxt 加载 2D 数组
    - .bmp/.png/.jpg : 通过 matplotlib.image.imread 加载，自动转灰度
不传参数时, 会自动找一张 SRAF 优化输出目录下的 wi `latest.txt` 作为示例。

输出:
    ./cmap_compare.png  (与脚本同目录)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 要对比的 colormap 列表（顺序 = 子图顺序）
# 中性柔和系: 既不发白刺眼, 也不会像 inferno 那样浓重
CMAPS = [
    "viridis",
    "cividis",
    "magma",
    "bone",
    "ocean",
    "gist_earth",
    "copper",
    "pink",
    "afmhot",
    "gist_heat",
    "coolwarm",
    "Spectral_r",
]


def _load_image(path: Path) -> np.ndarray:
    """加载 2D 数组 (灰度图或 txt 矩阵)。"""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        arr = np.loadtxt(str(path))
    elif suffix in {".bmp", ".png", ".jpg", ".jpeg"}:
        arr = mpimg.imread(str(path))
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")
    return np.asarray(arr, dtype=np.float32)


def _autoload_default() -> Path:
    """自动找一张示例 wi/mask txt 作为默认输入。"""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        # repo_root / "outputs/opc/sraf_width_opt/中期检查/宽度=2.5_分阶=否_PVBand=是_固定宽度=否_BS_非对称图案/wi_txt/latest.txt",
        repo_root / "outputs/opc/sraf_width_opt/中期检查/宽度=2.5_分阶=否_PVBand=是_固定宽度=否_BS_非对称图案/current_mask_txt/latest.txt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # 退化: 在 outputs/opc 下随便找一张 wi_txt/latest.txt
    found = list(repo_root.glob("outputs/opc/**/wi_txt/latest.txt"))
    if found:
        return found[0]
    raise FileNotFoundError("找不到示例输入文件，请显式传入路径。")


def main():
    # 1) 解析输入
    if len(sys.argv) >= 2:
        inp = Path(sys.argv[1])
    else:
        inp = _autoload_default()
        print(f"[info] 未传入路径，默认使用: {inp}")

    if not inp.is_file():
        raise FileNotFoundError(f"输入文件不存在: {inp}")

    img = _load_image(inp)
    print(f"[info] 已加载 {inp}, shape={img.shape}, "
          f"min={img.min():.4f}, max={img.max():.4f}")

    # 归一化到 [0,1]，避免不同数据范围导致视觉差异
    vmin, vmax = float(img.min()), float(img.max())
    if vmax - vmin > 1e-12:
        img_norm = (img - vmin) / (vmax - vmin)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)

    # 2) 排版: 3 行 4 列
    ncols = 4
    nrows = int(np.ceil(len(CMAPS) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.4, nrows * 3.4)
    )
    axes = np.asarray(axes).reshape(-1)

    for idx, cmap_name in enumerate(CMAPS):
        ax = axes[idx]
        im = ax.imshow(img_norm, cmap=cmap_name)
        ax.set_title(cmap_name, fontsize=11)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 多余子图隐藏
    for j in range(len(CMAPS), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(f"Colormap comparison ({inp.name})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # 3) 保存
    out_path = repo_root / "outputs" / "diagnostics" / "cmap_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] 已保存对比图: {out_path}")


if __name__ == "__main__":
    main()
