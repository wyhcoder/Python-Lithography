"""
可视化工具 — 绘制 mask / aerial image / wafer image。

跨后端通用: 接受 numpy 数组或 torch tensor。
"""
from __future__ import annotations

import math
from typing import Optional, List, Union

import numpy as np

# 可选 torch 支持
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _to_numpy(arr) -> np.ndarray:
    """将 numpy 或 torch tensor 统一转为 numpy。"""
    if _HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def plot_imaging_result(
    mask,
    aerial,
    wafer,
    titles: Optional[List[str]] = None,
    figsize: tuple = (18, 5),
    cmap: str = "magma",
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 120,
) -> None:
    """
    并排绘制 mask | aerial image | wafer image 三张图。

    Parameters
    ----------
    mask    : [N, N] mask 图案
    aerial  : [N, N] aerial intensity (空间像)
    wafer   : [N, N] wafer image (光刻胶像)
    titles  : 三张图的标题, 默认 ["Mask", "Aerial Image", "Wafer Image"]
    figsize : 图形尺寸
    cmap    : 颜色映射 (aerial/wafer 用 magma, mask 固定用 gray)
    suptitle: 总标题
    save_path : 保存路径 (None 则 plt.show())
    dpi     : 分辨率
    """
    import matplotlib.pyplot as plt

    mask_np = _to_numpy(mask)
    aerial_np = _to_numpy(aerial)
    wafer_np = _to_numpy(wafer)

    if titles is None:
        titles = ["Mask", "Aerial Image", "Wafer Image"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    # Mask: 二值用 gray, 灰度用 magma
    im0 = axes[0].imshow(mask_np, cmap="gray", interpolation="bilinear")
    axes[0].set_title(titles[0], fontsize=13, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Aerial Image
    im1 = axes[1].imshow(aerial_np, cmap=cmap, interpolation="bilinear")
    axes[1].set_title(titles[1], fontsize=13, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Wafer Image
    im2 = axes[2].imshow(wafer_np, cmap=cmap, interpolation="bilinear")
    axes[2].set_title(titles[2], fontsize=13, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  [plot] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_batch_results(
    masks,
    aerials,
    wafers,
    max_show: int = 6,
    ncols: int = 3,
    cmap: str = "magma",
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 120,
) -> None:
    """
    批量结果可视化: 每行一张 mask, 并排显示 mask/aerial/wafer。

    Parameters
    ----------
    masks    : [B, N, N]
    aerials  : [B, N, N]
    wafers   : [B, N, N]
    max_show : 最多显示多少张
    ncols    : 每行几个子图 (每张 mask 占 3 列 → mask/aerial/wafer)
    suptitle : 总标题
    save_path: 保存路径
    dpi      : 分辨率
    """
    import matplotlib.pyplot as plt

    masks_np = _to_numpy(masks)
    aerials_np = _to_numpy(aerials)
    wafers_np = _to_numpy(wafers)

    B = masks_np.shape[0]
    show_n = min(B, max_show)

    # 每张 mask 占 3 列
    ncols_total = 3 * min(show_n, ncols)
    nrows = math.ceil(show_n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols_total,
        figsize=(5 * ncols_total, 4 * nrows),
        dpi=dpi,
    )
    # 保证 axes 是 2D
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols_total == 1:
        axes = axes.reshape(-1, 1)

    for b in range(show_n):
        row = b // ncols
        col_base = (b % ncols) * 3

        # Mask
        ax_m = axes[row, col_base]
        ax_m.imshow(masks_np[b], cmap="gray", interpolation="bilinear")
        ax_m.set_title(f"Mask #{b+1}", fontsize=10, fontweight="bold")
        ax_m.axis("off")

        # Aerial
        ax_a = axes[row, col_base + 1]
        ax_a.imshow(aerials_np[b], cmap=cmap, interpolation="bilinear")
        ax_a.set_title(f"Aerial #{b+1}", fontsize=10)
        ax_a.axis("off")

        # Wafer
        ax_w = axes[row, col_base + 2]
        ax_w.imshow(wafers_np[b], cmap=cmap, interpolation="bilinear")
        ax_w.set_title(f"Wafer #{b+1}", fontsize=10)
        ax_w.axis("off")

    # 隐藏多余子图
    for b in range(show_n, nrows * ncols):
        row = b // ncols
        col_base = (b % ncols) * 3
        for dc in range(3):
            if col_base + dc < ncols_total:
                axes[row, col_base + dc].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  [plot] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_socs_kernels(
    kernels,
    eigenvalues,
    max_show: int = 8,
    ncols: int = 4,
    cmap: str = "RdBu",
    save_path: Optional[str] = None,
    dpi: int = 120,
) -> None:
    """
    可视化 SOCS 相干核的实部/虚部/振幅。

    Parameters
    ----------
    kernels     : [K, N, N] 复数 SOCS 核
    eigenvalues : [K] 特征值
    max_show    : 最多显示多少个核
    ncols       : 每行多少个核
    cmap        : 颜色映射 (复数用 RdBu 显示正负)
    save_path   : 保存路径
    dpi         : 分辨率
    """
    import matplotlib.pyplot as plt

    kernels_np = _to_numpy(kernels)
    eigenvalues_np = _to_numpy(eigenvalues)

    K = kernels_np.shape[0]
    show_n = min(K, max_show)

    total_energy = float(np.sum(eigenvalues_np))

    # 每个核显示 3 行 (实部/虚部/振幅)
    nrows = show_n * 3
    ncols = min(ncols, show_n)
    nrows_actual = math.ceil(show_n / ncols) * 3

    fig, axes = plt.subplots(
        nrows_actual, ncols,
        figsize=(3.5 * ncols, 3 * nrows_actual),
        dpi=dpi,
    )
    if nrows_actual == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(show_n):
        col = i % ncols
        row_base = (i // ncols) * 3

        kr = np.real(kernels_np[i])
        ki = np.imag(kernels_np[i])
        kmag = np.abs(kernels_np[i])
        energy_pct = eigenvalues_np[i] / total_energy * 100

        label = f"Kernel {i+1}  λ={eigenvalues_np[i]:.2e} ({energy_pct:.1f}%)"

        # 实部
        ax_r = axes[row_base, col]
        im = ax_r.imshow(kr, cmap=cmap, interpolation="bilinear")
        ax_r.set_title(f"{label}\nReal", fontsize=8)
        ax_r.axis("off")
        plt.colorbar(im, ax=ax_r, fraction=0.046)

        # 虚部
        ax_i = axes[row_base + 1, col]
        im = ax_i.imshow(ki, cmap=cmap, interpolation="bilinear")
        ax_i.set_title("Imag", fontsize=8)
        ax_i.axis("off")
        plt.colorbar(im, ax=ax_i, fraction=0.046)

        # 振幅
        ax_m = axes[row_base + 2, col]
        im = ax_m.imshow(kmag, cmap="magma", interpolation="bilinear")
        ax_m.set_title("|Amplitude|", fontsize=8)
        ax_m.axis("off")
        plt.colorbar(im, ax=ax_m, fraction=0.046)

    # 隐藏多余
    for idx in range(show_n * 3, nrows_actual * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    fig.suptitle("SOCS Coherent Kernels", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  [plot] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_energy_spectrum(
    eigenvalues,
    K_show: Optional[int] = None,
    save_path: Optional[str] = None,
    dpi: int = 120,
) -> None:
    """
    绘制 SOCS 特征值能量分布 (Scree Plot)。

    Parameters
    ----------
    eigenvalues : [K] 特征值
    K_show      : 横轴显示范围 (None = 全部)
    save_path   : 保存路径
    dpi         : 分辨率
    """
    import matplotlib.pyplot as plt

    eig = _to_numpy(eigenvalues)
    total = float(np.sum(eig))
    cumsum = np.cumsum(eig) / total * 100
    K = len(eig)

    if K_show is None:
        K_show = min(K, 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    # 左图: 特征值柱状图 + 累积曲线
    x = np.arange(1, K_show + 1)
    ax1.bar(x, eig[:K_show] / total, color="steelblue", alpha=0.7, label="Individual")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, cumsum[:K_show] / 100, "r-o", ms=4, lw=1.5, label="Cumulative")
    ax1.set_xlabel("Kernel Index")
    ax1.set_ylabel("Normalized Energy")
    ax1_twin.set_ylabel("Cumulative Energy")
    ax1.set_title("Eigenvalue Spectrum (Energy Distribution)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # 右图: 累积能量曲线 (全范围)
    ax2.plot(np.arange(1, K + 1), cumsum, "b-", lw=2)
    ax2.axhline(y=95, color="gray", ls="--", alpha=0.5, label="95%")
    ax2.axhline(y=99, color="gray", ls=":", alpha=0.5, label="99%")
    ax2.set_xlabel("Number of Kernels")
    ax2.set_ylabel("Cumulative Energy (%)")
    ax2.set_title(f"Cumulative Energy (Total: {K} kernels)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 标注
    for pct in [90, 95, 99]:
        idx = np.searchsorted(cumsum, pct)
        if idx < K:
            ax2.annotate(
                f"{pct}% → {idx+1} kernels",
                xy=(idx + 1, cumsum[idx]),
                xytext=(idx + 5, cumsum[idx] - 5),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  [plot] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
