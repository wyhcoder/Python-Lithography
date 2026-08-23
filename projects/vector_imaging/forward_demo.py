"""
端到端矢量成像 demo + 自动求导验证。

支持两种成像方式，通过 --method 选择：
    python3 forward_demo.py                        # 默认 Abbe
    python3 forward_demo.py --method socs           # SOCS 加速
    python3 forward_demo.py --method socs --K 20    # SOCS，指定核函数数量

流程：
1. 构建 grid / pupil / illumination / vector_transfer / forward_imaging
2. 用一个可学习的 mask（requires_grad=True）做一次前向
3. 计算简单 loss = MSE(aerial, target)，做 backward()
4. 检查 mask.grad 是否存在且数值有限
5. 出图：mask / source / aerial(x-pol) / aerial(y-pol) / aerial(unpol) / mask.grad
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from litho_model.config import SimulationConfig
from litho_model.forwardImage import VectorForwardImaging, VectorSOCSImaging, ScalarForwardImaging,ScalarSOCSImaging
from litho_model.grid import Grid
from litho_model.pupil import Pupil
from litho_model.socs import VectorSOCS,ScalarSOCS
from litho_model.source import Illumination, SourceType
from litho_model.vector_transfer import VectorTransfer
from tool.paths import output_path, project_root


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="矢量光刻成像 demo")
    p.add_argument(
        "--method", choices=["abbe", "socs"], default="socs",
        help="成像方式：abbe（默认，逐源点求和）或 socs（截断 SVD 加速）",
    )
    p.add_argument(
        "--K", type=int, default=10,
        help="SOCS 保留的核函数数量（仅 --method socs 时生效）",
    )
    p.add_argument(
        "--device", default="cpu",
        help="torch device",
    )
    p.add_argument(
        '--type',choices=['vector','scalar'],default='scalar'
    )
    return p.parse_args()


# ---------- 构建光学系统 ----------
def build_optical_system(cfg, device, dtype):
    """构造 Grid / Pupil / Illumination / VectorTransfer，返回 (grid, pupil, illum, vt)。"""
    grid = Grid(
        N=cfg.system.mask_size,
        dx=cfg.system.pixel_size_nm,
        device=device,
        dtype=dtype,
    )
    pupil = Pupil(
        NA=cfg.pupil.na,
        wave_length=cfg.system.wavelength_nm,
        refractive_index=cfg.pupil.n_image,
        grid=grid,
        aberrations=cfg.pupil.aberration,
        defocus_nm=cfg.pupil.defocus_nm,
        device=device,
    )
    illum = Illumination(
        pupil=pupil,
        grid=grid,
        source_type=SourceType(cfg.source.type),
        sigma_in=cfg.source.sigma_in,
        sigma_out=cfg.source.sigma_out,
        smoothing=cfg.source.smoothing,
        upsample=cfg.source.upsample,
    )
    np.savetxt(output_path("illum.txt", sub="diagnostics"), illum.source_map.cpu().numpy())
    vt = VectorTransfer(pupil=pupil, grid=grid)
    return grid, pupil, illum, vt


# ---------- 生成 demo mask ----------
def make_demo_mask(N: int, dtype: torch.dtype) -> torch.Tensor:
    """生成两条对称竖直缝作为 demo mask。"""
    m = torch.zeros((N, N), dtype=dtype)
    cy, cx = N // 2, N // 2
    half_h, w_half, gap = 70, 11, 16
    m[cy - half_h: cy + half_h, cx - gap - 2 * w_half: cx - gap] = 1.0
    m[cy - half_h: cy + half_h, cx + gap: cx + gap + 2 * w_half] = 1.0
    return m


# ---------- 出图 ----------
def plot_results(
    mask_np, source_np, fx_n, fy_n,
    aerial_un_np, aerial_x_np, aerial_y_np, grad_np,
    method_label: str, out_name: str,
) -> None:
    """2×3 子图：mask / source / unpolarized / x-pol / y-pol / gradient。"""
    vmax = max(aerial_x_np.max(), aerial_y_np.max(), aerial_un_np.max())
    g_abs = float(np.max(np.abs(grad_np)) or 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    # mask
    ax = axes[0, 0]
    im = ax.imshow(mask_np, cmap="gray", origin="lower")
    ax.set_title("mask (input, requires_grad)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # source
    ax = axes[0, 1]
    im = ax.imshow(
        source_np,
        extent=[fx_n[0].item(), fx_n[-1].item(), fy_n[0].item(), fy_n[-1].item()],
        cmap="hot", origin="lower",
    )
    ax.set_title("annular source J(σ)")
    ax.set_xlabel("σx"); ax.set_ylabel("σy")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # unpolarized
    ax = axes[0, 2]
    im = ax.imshow(aerial_un_np, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
    ax.set_title(f"aerial: unpolarized [{method_label}]")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # x-pol
    ax = axes[1, 0]
    im = ax.imshow(aerial_x_np, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
    ax.set_title("aerial: x-pol")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # y-pol
    ax = axes[1, 1]
    im = ax.imshow(aerial_y_np, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
    ax.set_title("aerial: y-pol")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # gradient
    ax = axes[1, 2]
    im = ax.imshow(grad_np, cmap="seismic", origin="lower", vmin=-g_abs, vmax=g_abs)
    ax.set_title("∂loss/∂mask (autograd)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out = output_path(out_name)
    plt.savefig(out, dpi=140)
    print(f"  -> 保存: {out}")


# ---------- main ----------
def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    dtype = torch.float32
    device = torch.device(args.device)

    # ---- 1) 读配置 ----
    cfg_path = project_root() / "configs" / "litho_system.yaml"
    cfg = SimulationConfig.from_yaml(cfg_path)

    # ---- 2) 构造光学系统 ----
    print(f"[1] 构造光学系统 (method={args.method}) ...")
    grid, pupil, illum, vt = build_optical_system(cfg, device, dtype)

    if args.type == 'vector':
        if args.method == "abbe":
            # ---------- Abbe 模式 ----------
            print("    使用 Abbe（逐源点求和）成像 ...")
            forward = VectorForwardImaging(
                grid=grid, pupil=pupil, illumination=illum,
                vector_transfer=vt, weight_threshold=1e-6, chunk_size=64,
            )
            method_label = f"Abbe (Ns≈{forward.fs_phys.numel()})"

            mask_init = make_demo_mask(grid.N, dtype)
            mask = mask_init.clone().requires_grad_(True)

            print("[2] 前向 + backward ...")
            aerial_x = forward.forward(mask, [1.0, 0.0])
            aerial_y = forward.forward(mask, [0.0, 1.0])
            aerial_unpol = 0.5 * (aerial_x + aerial_y)

        else:
            # ---------- SOCS 模式 ----------
            print(f"    使用 SOCS 成像 (K={args.K}) ...")
            socs = VectorSOCS(
                pupil=pupil, illumination=illum, vector_transfer=vt,
                K=args.K, weight_threshold=1e-6, unpolarized=True,
            )
            socs_model = VectorSOCSImaging(socs=socs, grid=grid)
            method_label = f"SOCS K={socs.K}"

            mask_init = make_demo_mask(grid.N, dtype)
            mask = mask_init.clone().requires_grad_(True)

            print("[2] 前向 + backward ...")
            # SOCS 非偏振：一次前向直接出 (Ix+Iy)/2
            aerial_unpol = socs_model.forward(mask)

            # X/Y 偏振：单偏振 SOCS 分别计算（用于对比）
            socs_x = VectorSOCS(
                pupil=pupil, illumination=illum, vector_transfer=vt,
                K=args.K, weight_threshold=1e-6,
                unpolarized=False, jones=(1.0, 0.0),
            )
            socs_x_model = VectorSOCSImaging(socs=socs_x, grid=grid)
            aerial_x = socs_x_model.forward(mask)

            socs_y = VectorSOCS(
                pupil=pupil, illumination=illum, vector_transfer=vt,
                K=args.K, weight_threshold=1e-6,
                unpolarized=False, jones=(0.0, 1.0),
            )
            socs_y_model = VectorSOCSImaging(socs=socs_y, grid=grid)
            aerial_y = socs_y_model.forward(mask)
    if args.type == 'scalar':
        if args.method == "abbe":
            # ---------- Abbe 模式 ----------
            print("    使用 Abbe（逐源点求和）成像 ...")
            forward = ScalarForwardImaging(
                grid=grid, pupil=pupil, illumination=illum,
             weight_threshold=1e-6, chunk_size=64,
            )
            method_label = f"Abbe (Ns≈{forward.fs_phys.numel()})"

            mask_init = make_demo_mask(grid.N, dtype)
            mask = mask_init.clone().requires_grad_(True)

            print("[2] 前向 + backward ...")
            aerial_unpol = forward.forward(mask)
            aerial_x = aerial_unpol
            aerial_y = aerial_unpol
        else:
            # ---------- SOCS 模式 ----------
            print(f"    使用 SOCS 成像 (K={args.K}) ...")
            socs = ScalarSOCS(
                pupil=pupil, illumination=illum,
                K=args.K, weight_threshold=1e-6)
            socs_model = ScalarSOCSImaging(socs=socs, grid=grid)
            method_label = f"SOCS K={socs.K}"
            mask_init = make_demo_mask(grid.N, dtype)
            mask = mask_init.clone().requires_grad_(True)

            aerial_unpol = socs_model.forward(mask)
            aerial_x = aerial_unpol
            aerial_y = aerial_unpol

    # ---- 3) loss + backward ----
    target = mask_init / (mask_init.max() + 1e-12)
    loss = ((aerial_unpol - target) ** 2).mean()
    loss.backward()

    grad = mask.grad
    grad_finite = torch.isfinite(grad).all().item()
    print(
        f"  - loss = {loss.item():.6e}, "
        f"|grad| max = {grad.abs().max().item():.3e}, "
        f"all finite = {grad_finite}"
    )
    assert grad_finite, "mask.grad 出现 NaN/Inf"
    assert grad.abs().max().item() > 0, "mask.grad 全 0，链路不可导"

    # ---- 4) 出图 ----
    print("[3] 绘图 ...")
    plot_results(
        mask_np=mask.detach().cpu().numpy(),
        source_np=illum.get_source().detach().cpu().numpy(),
        fx_n=illum.get_axes()[0],
        fy_n=illum.get_axes()[1],
        aerial_un_np=aerial_unpol.detach().cpu().numpy(),
        aerial_x_np=aerial_x.detach().cpu().numpy(),
        aerial_y_np=aerial_y.detach().cpu().numpy(),
        grad_np=grad.detach().cpu().numpy(),
        method_label=method_label,
        out_name=f"forward_demo_{args.method}.png",
    )


if __name__ == "__main__":
    main()
