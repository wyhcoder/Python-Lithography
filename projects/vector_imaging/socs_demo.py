"""
SOCS 加速验证 demo：
1. 构建 Abbe 参考模型
2. 用不同 K 值构建 SOCS 模型，对比精度 (NMSE) 和速度
3. 出图：参考 / K=10 / K=20 / 差值图，以及 K-NMSE 曲线
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from litho_model.config import SimulationConfig
from litho_model.forwardImage import VectorForwardImaging, VectorSOCSImaging
from litho_model.grid import Grid
from litho_model.pupil import Pupil
from litho_model.socs import VectorSOCS
from litho_model.source import Illumination, SourceType
from litho_model.vector_transfer import VectorTransfer
from tool.paths import output_path, project_root


def build_system(cfg, device, dtype):
    grid = Grid(N=cfg.system.mask_size, dx=cfg.system.pixel_size_nm,
                device=device, dtype=dtype)
    pupil = Pupil(NA=cfg.pupil.na, wave_length=cfg.system.wavelength_nm,
                  refractive_index=cfg.pupil.n_image, grid=grid,
                  aberrations=cfg.pupil.aberration, defocus_nm=cfg.pupil.defocus_nm,
                  device=device)
    illum = Illumination(pupil=pupil, grid=grid,
                         source_type=SourceType(cfg.source.type),
                         sigma_in=cfg.source.sigma_in, sigma_out=cfg.source.sigma_out,
                         smoothing=cfg.source.smoothing, upsample=cfg.source.upsample)
    vt = VectorTransfer(pupil=pupil, grid=grid)
    abbe = VectorForwardImaging(grid=grid, pupil=pupil, illumination=illum,
                                vector_transfer=vt, weight_threshold=1e-6, chunk_size=64)
    return grid, pupil, illum, vt, abbe


def make_mask(N, dtype):
    m = torch.zeros((N, N), dtype=dtype)
    cy, cx = N // 2, N // 2
    m[cy-60:cy+60, cx-18:cx-4] = 1.0
    m[cy-60:cy+60, cx+4:cx+18] = 1.0
    return m


def main():
    device = torch.device("cpu")
    dtype = torch.float32
    cfg_path = project_root() / "configs" / "litho_system.yaml"
    cfg = SimulationConfig.from_yaml(cfg_path)

    print("=== 构建光学系统 ===")
    grid, pupil, illum, vt, abbe = build_system(cfg, device, dtype)
    N = cfg.system.mask_size
    mask = make_mask(N, dtype)

    # Abbe 参考
    print("\n=== Abbe 参考前向（非偏振）===")
    t0 = time.perf_counter()
    with torch.no_grad():
        I_ref = 0.5 * (abbe._aerial_for_jones(mask, 1.0, 0.0) +
                       abbe._aerial_for_jones(mask, 0.0, 1.0))
    t_abbe = time.perf_counter() - t0
    I_ref = 1.0 / (1.0 + torch.exp(-85.0 * (I_ref - 0.15)))

    print(f"    Abbe 耗时: {t_abbe*1000:.1f} ms  max={I_ref.max():.4f}")

    # 扫描 K 值
    K_list = [5, 10, 15, 20, 30]
    nmse_list, speed_list = [], []
    socs_images = {}

    for K in K_list:
        print(f"\n=== SOCS K={K} ===")
        socs = VectorSOCS(pupil=pupil, illumination=illum, vector_transfer=vt,
                          K=K, weight_threshold=1e-6, jones=(1.0, 0.0))
        ef = socs.energy_fraction()
        print(f"    能量保留比: {ef*100:.2f}%")

        socs_model = VectorSOCSImaging(socs=socs, grid=grid,
                                       sigmoid_tr=0.15, sigmoid_a=85.0)

        t0 = time.perf_counter()
        with torch.no_grad():
            I_socs = socs_model._aerial_socs(mask)
        t_socs = time.perf_counter() - t0
        I_socs = 1.0 / (1.0 + torch.exp(-85.0 * (I_socs - 0.15)))
        diff = I_socs - I_ref
        nmse = ((diff**2).mean() / (I_ref**2).mean().clamp_min(1e-12)).item()
        print(f"    SOCS 耗时: {t_socs*1000:.1f} ms  NMSE={nmse:.2e}  加速比={t_abbe/t_socs:.1f}×")

        nmse_list.append(nmse)
        speed_list.append(t_abbe / t_socs)
        if K in (10, 20):
            socs_images[K] = I_socs.numpy()

    # ─── 出图 ───
    I_ref_np = I_ref.detach().numpy()
    vmax = I_ref_np.max()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    def show(ax, img, title, cmap="inferno", vmin=0, vv=None):
        im = ax.imshow(img, cmap=cmap, origin="lower",
                       vmin=vmin, vmax=(vv or vmax))
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    show(axes[0, 0], I_ref_np, "Abbe 参考 (非偏振)")
    if 10 in socs_images:
        show(axes[0, 1], socs_images[10], "SOCS K=10")
        diff10 = socs_images[10] - I_ref_np
        vd = np.abs(diff10).max()
        show(axes[0, 2], diff10, "差值 (K=10 - Abbe)", cmap="seismic", vmin=-vd, vv=vd)
    if 20 in socs_images:
        show(axes[0, 3], socs_images[20], "SOCS K=20")

    # K-NMSE 曲线
    ax = axes[1, 0]
    ax.semilogy(K_list, nmse_list, "o-", color="royalblue")
    ax.set_xlabel("K (SOCS 项数)")
    ax.set_ylabel("NMSE")
    ax.set_title("NMSE vs K")
    ax.grid(True, which="both", alpha=0.4)

    # 加速比曲线
    ax = axes[1, 1]
    ax.plot(K_list, speed_list, "s-", color="tomato")
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xlabel("K")
    ax.set_ylabel("加速比 (Abbe / SOCS)")
    ax.set_title("加速比 vs K")
    ax.grid(True, alpha=0.4)

    # 中心剖面对比
    ax = axes[1, 2]
    cx = N // 2
    ax.plot(I_ref_np[cx, :], label="Abbe", lw=2)
    for K in [10, 20]:
        if K in socs_images:
            ax.plot(socs_images[K][cx, :], "--", label=f"SOCS K={K}")
    ax.set_title("水平中心剖面")
    ax.set_xlabel("pixel x")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    axes[1, 3].axis("off")

    plt.suptitle(
        f"矢量 SOCS 精度 vs 速度对比  "
        f"(N={N}, dx={cfg.system.pixel_size_nm}nm, NA={cfg.pupil.na})",
        fontsize=12,
    )
    plt.tight_layout()
    out = output_path("socs_demo.png")
    plt.savefig(out, dpi=140)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
