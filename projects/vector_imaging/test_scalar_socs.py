"""
快速验证 ScalarForwardImaging (Abbe) 与 ScalarSOCSImaging 的数值一致性。

对比指标:
    NMSE = mean((I_socs - I_abbe)²) / mean(I_abbe²)
    max_abs_err

合格标准: NMSE < 1e-3, max_abs_err < 1e-2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from litho_model.config import SimulationConfig
from litho_model.grid import Grid
from litho_model.pupil import Pupil
from litho_model.source import Illumination, SourceType
from litho_model.forwardImage import ScalarForwardImaging, ScalarSOCSImaging
from litho_model.socs import ScalarSOCS


def build_optics(cfg, device, dtype):
    grid = Grid(N=cfg.system.mask_size, dx=cfg.system.pixel_size_nm,
                device=device, dtype=dtype)
    pupil = Pupil(NA=cfg.pupil.na, wave_length=cfg.system.wavelength_nm,
                  refractive_index=cfg.pupil.n_image, grid=grid,
                  aberrations=cfg.pupil.aberration,
                  defocus_nm=cfg.pupil.defocus_nm, device=device)
    illum = Illumination(pupil=pupil, grid=grid,
                         source_type=SourceType(cfg.source.type),
                         sigma_in=cfg.source.sigma_in,
                         sigma_out=cfg.source.sigma_out,
                         smoothing=cfg.source.smoothing,
                         upsample=cfg.source.upsample)
    return grid, pupil, illum


def make_demo_mask(N, dtype):
    """两条对称竖直缝。"""
    m = torch.zeros((N, N), dtype=dtype)
    cy, cx = N // 2, N // 2
    half_h, w_half, gap = 70, 11, 16
    m[cy - half_h:cy + half_h, cx - gap - 2 * w_half:cx - gap] = 1.0
    m[cy - half_h:cy + half_h, cx + gap:cx + gap + 2 * w_half] = 1.0
    return m


def main():
    torch.manual_seed(0)
    dtype = torch.float32
    device = torch.device("cpu")

    cfg_path = PROJECT_ROOT / "configs" / "litho_system.yaml"
    cfg = SimulationConfig.from_yaml(cfg_path)
    print(f"[1] 配置: λ={cfg.system.wavelength_nm}nm, NA={cfg.pupil.na}, "
          f"N={cfg.system.mask_size}")

    grid, pupil, illum = build_optics(cfg, device, dtype)

    # ---- 1) Abbe 参考 ----
    print("[2] 构建 ScalarForwardImaging (Abbe) ...")
    abbe = ScalarForwardImaging(grid=grid, pupil=pupil, illumination=illum,
                                 weight_threshold=1e-6, chunk_size=64)
    print(f"    Ns_active = {abbe.fs_phys.numel()}")

    # ---- 2) SOCS ----
    print("[3] 构建 ScalarSOCS + ScalarSOCSImaging ...")
    K = 20
    socs = ScalarSOCS(pupil=pupil, illumination=illum, K=K,
                      weight_threshold=1e-6)
    socs_imaging = ScalarSOCSImaging(socs=socs, grid=grid)
    print(f"    K={socs.K}, energy_fraction={socs.energy_fraction():.4f}")

    # ---- 3) 单 mask 对比 (raw aerial) ----
    print("[4] 单 mask raw aerial 对比 ...")
    mask = make_demo_mask(grid.N, dtype)

    with torch.no_grad():
        I_abbe = abbe._aerial_abbe(mask)
        I_socs = socs_imaging._aerial_socs(mask)

    diff = I_socs - I_abbe
    nmse = (diff ** 2).mean() / (I_abbe ** 2).mean().clamp_min(1e-12)
    max_err = diff.abs().max()
    print(f"    NMSE          = {nmse.item():.4e}")
    print(f"    max_abs_err   = {max_err.item():.4e}")
    print(f"    I_abbe range  = [{I_abbe.min():.4e}, {I_abbe.max():.4e}]")
    print(f"    I_socs range  = [{I_socs.min():.4e}, {I_socs.max():.4e}]")

    # ---- 4) 批量前向对比 (sigmoid 后) ----
    print("[5] 批量 forward_batch 对比 (B=4) ...")
    masks = torch.stack([make_demo_mask(grid.N, dtype) for _ in range(4)],
                         dim=0)
    # 加点扰动
    masks = masks + 0.01 * torch.randn_like(masks)

    with torch.no_grad():
        wafer_abbe = abbe.forward_batch(masks)
        wafer_socs = socs_imaging.forward_batch(masks)

    diff_b = wafer_socs - wafer_abbe
    nmse_b = (diff_b ** 2).mean() / (wafer_abbe ** 2).mean().clamp_min(1e-12)
    max_err_b = diff_b.abs().max()
    print(f"    NMSE (batch)        = {nmse_b.item():.4e}")
    print(f"    max_abs_err (batch) = {max_err_b.item():.4e}")

    # ---- 5) 自动求导验证 ----
    print("[6] autograd 验证 (mask.requires_grad=True) ...")
    mask_g = make_demo_mask(grid.N, dtype).clone().requires_grad_(True)
    wafer = socs_imaging.forward(mask_g)
    target = make_demo_mask(grid.N, dtype) / 1.0
    loss = ((wafer - target) ** 2).mean()
    loss.backward()
    grad = mask_g.grad
    print(f"    loss = {loss.item():.6e}")
    print(f"    |grad|.max = {grad.abs().max().item():.4e}, "
          f"all_finite = {torch.isfinite(grad).all().item()}")

    # ---- 结论 ----
    print("\n" + "=" * 60)
    if nmse.item() < 1e-3 and max_err.item() < 1e-2:
        print("✅ ScalarSOCS 与 ScalarForwardImaging 数值一致")
    elif nmse.item() < 1e-2:
        print("⚠️  数值接近但有偏差: 检查 K 是否够大 / weight_threshold 设置")
    else:
        print("❌ 差异显著, 实现可能有 bug")
    print("=" * 60)


if __name__ == "__main__":
    main()
