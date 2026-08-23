#!/usr/bin/env python3
"""
scalar_socs_imaging 完整演示脚本

展示:
  1) numpy 后端: 单张 + 批量 SOCS 成像
  2) pytorch 后端: GPU 加速 + autograd 梯度 + 批量仿真
  3) 生成测试 mask (contact hole + line/space)
  4) 性能对比: numpy vs torch, single vs batch
  5) SOCS 能量谱分析

用法:
  python demo.py                        # 默认: 两种后端都跑
  python demo.py --backend numpy        # 仅 numpy
  python demo.py --backend torch        # 仅 torch
  python demo.py --device cuda          # torch 使用 GPU
  python demo.py --batch 16             # 批量测试 16 张 mask
  python demo.py --grid 129             # 小网格 (快速)
"""
from __future__ import annotations

import argparse
import time
import sys
import os

import numpy as np

# 将父目录加入 path 以便直接运行
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent_dir)

from scalar_socs_imaging.config import OpticalConfig
from scalar_socs_imaging.numpy_backend import ScalarImagingNumpy
from scalar_socs_imaging.torch_backend import ScalarImagingTorch
from scalar_socs_imaging.visualization import (
    plot_imaging_result,
    plot_batch_results,
    plot_socs_kernels,
    plot_energy_spectrum,
)


# ============================================================
#  测试 Mask 生成
# ============================================================

def make_contact_hole(N: int, radius: int = 20) -> np.ndarray:
    """
    生成圆形 contact hole mask。

    Parameters
    ----------
    N      : 网格大小
    radius : 通孔半径 [像素]

    Returns
    -------
    mask : [N, N] 二值 mask (0=背景, 1=通孔)
    """
    y, x = np.ogrid[:N, :N]
    cx, cy = N // 2, N // 2
    mask = np.where((x - cx) ** 2 + (y - cy) ** 2 <= radius**2, 1.0, 0.0)
    return mask.astype(np.float64)


def make_line_space(N: int, pitch: int = 20) -> np.ndarray:
    """
    生成 line/space 周期图案。

    Parameters
    ----------
    N     : 网格大小
    pitch : 周期 [像素]

    Returns
    -------
    mask : [N, N] 二值 mask
    """
    x = np.arange(N)
    mask = np.where((x % pitch) < pitch // 2, 1.0, 0.0)
    return np.tile(mask, (N, 1)).astype(np.float64)


def make_random_perturbed(
    base_mask: np.ndarray,
    B: int,
    delta: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    生成 B 张带随机扰动的 mask (用于批量测试)。

    扰动方式: 边缘像素随机 +/- delta。

    Parameters
    ----------
    base_mask : [N, N] 基础 mask
    B         : 批量大小
    delta     : 扰动量 [像素值]
    seed      : 随机种子

    Returns
    -------
    masks : [B, N, N]
    """
    rng = np.random.default_rng(seed)
    N = base_mask.shape[0]

    masks = np.tile(base_mask[None, :, :], (B, 1, 1)).astype(np.float64)

    # 只扰动边缘像素 (mask 值与周围不同的像素)
    from scipy.ndimage import binary_dilation, binary_erosion
    binary = base_mask > 0.5
    edge = binary_dilation(binary) ^ binary_erosion(binary)

    for b in range(B):
        noise = np.zeros((N, N), dtype=np.float64)
        noise[edge] = rng.uniform(-delta, delta, size=int(edge.sum()))
        masks[b] = np.clip(masks[b] + noise, 0.0, 1.0)

    return masks


# ============================================================
#  Demo 函数
# ============================================================

def demo_numpy(config: OpticalConfig, N: int, B: int, plot: bool = True, svd_method: str = "randomized"):
    """numpy 后端演示: 单张 + 批量 + 能量谱。"""
    print("\n" + "#" * 60)
    print("#  DEMO: NumPy Backend")
    print("#" * 60 + "\n")

    sim = ScalarImagingNumpy(config, N=N)
    sim.prepare(svd_method=svd_method)

    # 能量谱
    sim.print_energy_report()

    if plot:
        # 绘制 SOCS 核
        print("\n[Plot] SOCS Kernels...")
        plot_socs_kernels(
            sim.socs.get_kernels(),
            sim._eigenvalues,
            max_show=6,
            ncols=3,
        )
        # 绘制能量谱
        print("[Plot] Energy Spectrum...")
        plot_energy_spectrum(sim._eigenvalues)

    # ---- 单张 contact hole ----
    print("\n>>> Single mask: Contact Hole")
    mask_ch = make_contact_hole(N, radius=N // 10)
    aerial_ch, wafer_ch = sim.forward(mask_ch, verbose=True)

    if plot:
        print("[Plot] Contact Hole imaging result...")
        plot_imaging_result(
            mask_ch, aerial_ch, wafer_ch,
            suptitle="Contact Hole — SOCS Imaging (NumPy)",
        )

    # ---- 单张 line/space ----
    print("\n>>> Single mask: Line/Space")
    mask_ls = make_line_space(N, pitch=N // 10)
    aerial_ls, wafer_ls = sim.forward(mask_ls, verbose=True)

    if plot:
        print("[Plot] Line/Space imaging result...")
        plot_imaging_result(
            mask_ls, aerial_ls, wafer_ls,
            suptitle="Line/Space — SOCS Imaging (NumPy)",
        )

    # ---- 批量成像 ----
    print(f"\n>>> Batch: {B} perturbed masks")
    masks_batch = make_random_perturbed(mask_ch, B=B, delta=0.3)
    t0 = time.time()
    aerials_b, wafers_b = sim.forward_batch(masks_batch, verbose=False)
    elapsed = time.time() - t0
    print(
        f"  Batch {B} masks: total={elapsed:.3f}s, "
        f"per_mask={elapsed/B*1000:.1f}ms, "
        f"aerial range=[{aerials_b.min():.4f}, {aerials_b.max():.4f}]"
    )

    if plot:
        print("[Plot] Batch imaging results...")
        plot_batch_results(
            masks_batch, aerials_b, wafers_b,
            max_show=4, ncols=2,
            suptitle=f"Batch SOCS Imaging — {B} Perturbed Masks (NumPy)",
        )

    # ---- 对比: 串行 vs 批量 ----
    print("\n>>> Benchmark: Serial vs Batch (numpy)")
    t0 = time.time()
    for b in range(B):
        sim.forward(masks_batch[b], verbose=False)
    serial_time = time.time() - t0

    t0 = time.time()
    sim.forward_batch(masks_batch, verbose=False)
    batch_time = time.time() - t0

    print(f"  Serial: {serial_time:.3f}s ({serial_time/B*1000:.1f} ms/mask)")
    print(f"  Batch:  {batch_time:.3f}s ({batch_time/B*1000:.1f} ms/mask)")
    print(f"  Speedup: {serial_time/batch_time:.1f}x")

    return sim


def demo_torch(config: OpticalConfig, N: int, B: int, device: str, plot: bool = True, svd_method: str = "randomized"):
    """PyTorch 后端演示: 单张 + 批量 + autograd + GPU。"""
    print("\n" + "#" * 60)
    print(f"#  DEMO: PyTorch Backend (device={device})")
    print("#" * 60 + "\n")

    import torch

    sim = ScalarImagingTorch(config, N=N)
    model = sim.prepare(device=device, svd_method=svd_method)

    # ---- 单张 contact hole ----
    print("\n>>> Single mask: Contact Hole (PyTorch)")
    mask_ch = make_contact_hole(N, radius=N // 10)
    mask_t = torch.from_numpy(mask_ch).float().to(device)

    t0 = time.time()
    with torch.no_grad():
        wafer_t = model(mask_t)
        aerial_t = model.forward_aerial(mask_t)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(
        f"  Single forward: {elapsed*1000:.1f}ms, "
        f"aerial max={aerial_t.max().item():.4f}, "
        f"wafer max={wafer_t.max().item():.4f}"
    )

    if plot:
        print("[Plot] Contact Hole (PyTorch)...")
        plot_imaging_result(
            mask_ch, aerial_t, wafer_t,
            suptitle=f"Contact Hole — SOCS Imaging (PyTorch, {device})",
        )

    # ---- 批量成像 ----
    print(f"\n>>> Batch: {B} perturbed masks (PyTorch)")
    masks_batch = make_random_perturbed(mask_ch, B=B, delta=0.3)
    masks_b = torch.from_numpy(masks_batch).float().to(device)

    t0 = time.time()
    with torch.no_grad():
        wafers_b = model.forward_batch(masks_b)
        aerials_b = model.forward_batch_aerial(masks_b)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(
        f"  Batch {B} masks: total={elapsed*1000:.1f}ms, "
        f"per_mask={elapsed/B*1000:.1f}ms, "
        f"aerial range=[{aerials_b.min().item():.4f}, {aerials_b.max().item():.4f}]"
    )

    if plot:
        print("[Plot] Batch results (PyTorch)...")
        plot_batch_results(
            masks_batch, aerials_b, wafers_b,
            max_show=4, ncols=2,
            suptitle=f"Batch SOCS Imaging — {B} Perturbed Masks (PyTorch, {device})",
        )

    # ---- Autograd 梯度测试 ----
    print("\n>>> Autograd: Gradient of wafer w.r.t mask")
    mask_grad = torch.from_numpy(mask_ch).float().to(device).requires_grad_(True)

    wafer = model(mask_grad)
    loss = ((wafer - 0.5) ** 2).mean()
    loss.backward()

    grad = mask_grad.grad
    print(
        f"  Loss: {loss.item():.6f}, "
        f"||grad||_max: {grad.abs().max().item():.4f}, "
        f"||grad||_mean: {grad.abs().mean().item():.6f}"
    )

    # ---- 批量 vs 串行 benchmark ----
    print(f"\n>>> Benchmark: Serial vs Batch (torch, {device})")
    with torch.no_grad():
        t0 = time.time()
        for b in range(B):
            model(masks_b[b])
        if device == "cuda":
            torch.cuda.synchronize()
        serial_time = time.time() - t0

        t0 = time.time()
        model.forward_batch(masks_b)
        if device == "cuda":
            torch.cuda.synchronize()
        batch_time = time.time() - t0

    print(f"  Serial: {serial_time*1000:.1f}ms ({serial_time/B*1000:.1f} ms/mask)")
    print(f"  Batch:  {batch_time*1000:.1f}ms ({batch_time/B*1000:.1f} ms/mask)")
    if batch_time > 0:
        print(f"  Speedup: {serial_time/batch_time:.1f}x")

    return model


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="scalar_socs_imaging Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", choices=["numpy", "torch", "both"],
        default="numpy", help="Which backend to run"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size for timing tests"
    )
    parser.add_argument(
        "--grid", type=int, default=129,
        help="Grid size N (small=faster, large=realistic)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable visualization plots"
    )
    parser.add_argument(
        "--svd-method", default="randomized",
        choices=["randomized", "full"],
        help="SVD method: randomized (fast) or full (slow, accurate)"
    )
    args = parser.parse_args()

    # 配置
    config = OpticalConfig.default()
    config.mask.pixel_size_nm = 4.0 if args.grid <= 257 else 4.0
    config.socs_k = 30 if args.grid <= 129 else 50

    print("=" * 60)
    print("  Scalar SOCS Imaging System — Demo")
    print("=" * 60)
    print(f"  Grid:      {args.grid}×{args.grid}")
    print(f"  Batch:     {args.batch}")
    print(f"  Backend:   {args.backend}")
    print(f"  Device:    {args.device}")
    print(f"  Plot:      {not args.no_plot}")
    print(f"  SVD:       {args.svd_method}")
    print("=" * 60)

    plot_enabled = not args.no_plot
    svd_method = args.svd_method

    if args.backend in ("numpy", "both"):
        demo_numpy(config, args.grid, args.batch, plot=plot_enabled, svd_method=svd_method)

    if args.backend in ("torch", "both"):
        demo_torch(config, args.grid, args.batch, args.device, plot=plot_enabled, svd_method=svd_method)

    print("\n" + "=" * 60)
    print("  Demo completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
