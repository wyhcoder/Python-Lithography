"""
PyTorch 版前向仿真 + EPE/wEPE 计算（可微）。

数学等价于 ``litho_model/demo_compute_image.py::images_simulation_v3`` (SOCS 路径) 与
``op_model/demo_epe_wepe.py::caculate_epe / caculate_wepe``，但所有运算用
``torch.fft`` 和 tensor 运算，对 mask 完全可微。

注意：本模块**不修改** ``opt_cache``，只读取里面已经预计算好的：
    - H_k_list (空间域相干核)  或者
    - H_k_fft_stack (我们之前缓存的 batch FFT 形式) 优先使用
    - source_weights (奇异值平方)
    - norm_factor (SOCS 归一化因子)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# 工具：把 numpy 缓存搬到 torch (一次性，迭代里复用)
# -----------------------------------------------------------------------------

def opt_cache_to_torch(opt_cache: dict,
                       device: torch.device,
                       dtype_real: torch.dtype = torch.float64) -> dict:
    """
    把 simulator.opt_cache 里的 numpy 数组转成 torch tensor，缓存供
    ``forward_simulation_torch`` 使用。

    返回的 dict 字段：
    - H_k_fft_stack : (N, H, W) complex tensor
    - source_weights: (N,) real tensor
    - norm_factor   : scalar tensor (SOCS 时存在)
    """
    out = {}

    # 优先用 batch FFT stack；没有就从 H_k_list 现算一次
    if "H_k_fft_stack" in opt_cache:
        H_stack_np = np.asarray(opt_cache["H_k_fft_stack"]).astype(np.complex128)
        H_stack = torch.from_numpy(H_stack_np).to(device)
    else:
        H_list = opt_cache["H_k_list"]
        # H_k 是空间域复数核，stack 到 (N, H, W) 后做 fft2
        H_np = np.stack([np.asarray(h, dtype=np.complex128) for h in H_list], axis=0)
        H_stack_t = torch.from_numpy(H_np).to(device)
        H_stack = torch.fft.fft2(H_stack_t)
    out["H_k_fft_stack"] = H_stack

    sw_np = np.asarray(opt_cache["source_weights"], dtype=np.float64)
    out["source_weights"] = torch.from_numpy(sw_np).to(device=device, dtype=dtype_real)

    if "norm_factor" in opt_cache:
        out["norm_factor"] = torch.tensor(float(opt_cache["norm_factor"]),
                                          device=device, dtype=dtype_real)
    return out


# -----------------------------------------------------------------------------
# 前向仿真 (可微)
# -----------------------------------------------------------------------------

def forward_simulation_torch(mask_spatial: torch.Tensor,
                             cache_t: dict,
                             threshold: float,
                             alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    与 images_simulation_v3 (SOCS, batch FFT) 数学等价的可微版本。

    Parameters
    ----------
    mask_spatial : (H, W) tensor (real, requires_grad 视上游而定)
    cache_t      : ``opt_cache_to_torch`` 输出
    threshold    : resist 阈值
    alpha        : sigmoid 斜率

    Returns
    -------
    aerial_image : (H, W) tensor (real)
    wafer_image  : (H, W) tensor (real, ∈ [0, 1])
    """
    H_k_fft_stack = cache_t["H_k_fft_stack"]          # (N, H, W) complex
    source_weights = cache_t["source_weights"]        # (N,) real

    # 与 fftconvolve1 等价: ifftshift(ifft2(fft2(mask) * fft2(H_k)))
    mask_complex = mask_spatial.to(H_k_fft_stack.dtype)
    mask_fft = torch.fft.fft2(mask_complex)           # (H, W)
    spectra = mask_fft.unsqueeze(0) * H_k_fft_stack   # (N, H, W)
    E_stack = torch.fft.ifftshift(
        torch.fft.ifft2(spectra),
        dim=(-2, -1),
    )                                                 # (N, H, W) complex

    intensities = E_stack.real.pow(2) + E_stack.imag.pow(2)  # (N, H, W) real
    aerial_acc = (source_weights[:, None, None] * intensities).sum(dim=0)

    if "norm_factor" in cache_t:
        aerial = aerial_acc / cache_t["norm_factor"]
    else:
        total = source_weights.sum()
        aerial = aerial_acc / total.clamp_min(1e-9)

    wafer = torch.sigmoid(alpha * (aerial - threshold))
    return aerial, wafer


# -----------------------------------------------------------------------------
# EPE / wEPE (可微)
# -----------------------------------------------------------------------------

def caculate_epe_torch(aI: torch.Tensor,
                       threshold: float,
                       eps_yx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    与 ``op_model.demo_epe_wepe.caculate_epe`` 数学等价的可微版本。

    EPE_i = ((aI(p_i) - tr) / |∇aI(p_i)|) ^ 2

    其中 |∇aI| 用与 numpy.gradient(aI, 0.5) 一致的中心差分：
        dx = (aI[:, i+1] - aI[:, i-1]) / 1.0   (因为 spacing=0.5 双边总间距 1.0)

    Parameters
    ----------
    aI : (H, W) tensor
    threshold : float
    eps_yx : (K, 2) long tensor 采样点 (y, x) 整数像素坐标

    Returns
    -------
    epe_sum    : scalar tensor   (= sum(epe_vector))
    epe_vector : (K,) tensor
    """
    # 中心差分 (与 numpy.gradient(aI, 0.5, axis=...) 等价)
    # numpy.gradient 在内部点用 (a[i+1] - a[i-1]) / (2 * 0.5) = (a[i+1]-a[i-1])
    # 边界用单侧差分 / 0.5 → 这里我们也照搬
    H, W = aI.shape

    # 内部用 roll 做 (a[i+1] - a[i-1]) / 1.0
    dx_inner = (torch.roll(aI, -1, dims=1) - torch.roll(aI, 1, dims=1))  # (H, W)
    dy_inner = (torch.roll(aI, -1, dims=0) - torch.roll(aI, 1, dims=0))

    # 边界修正 (与 np.gradient 完全一致：只用单侧差分)
    # spacing=0.5 时单侧差分 = (a[1] - a[0]) / 0.5
    dx = dx_inner.clone()
    dx[:, 0] = (aI[:, 1] - aI[:, 0]) / 0.5
    dx[:, -1] = (aI[:, -1] - aI[:, -2]) / 0.5

    dy = dy_inner.clone()
    dy[0, :] = (aI[1, :] - aI[0, :]) / 0.5
    dy[-1, :] = (aI[-1, :] - aI[-2, :]) / 0.5

    G = torch.sqrt(dx.pow(2) + dy.pow(2)).clamp_min(1e-12)
    epe_field = ((aI - threshold) / G).pow(2)               # (H, W)

    # 在 eps_yx 采样
    rows = eps_yx[:, 0].to(torch.long)
    cols = eps_yx[:, 1].to(torch.long)
    epe_vector = epe_field[rows, cols]                       # (K,)
    return epe_vector.sum(), epe_vector


def caculate_wepe_torch(wepe_calculate: torch.Tensor,
                        epe_vector: torch.Tensor) -> torch.Tensor:
    """
    与 caculate_wepe 等价：(weight * epe).sum()
    """
    w = wepe_calculate.to(epe_vector.dtype).reshape(-1)
    e = epe_vector.reshape(-1)
    return (w * e).sum()
