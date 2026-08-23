"""
标量 SOCS 成像 — PyTorch 后端

与 numpy 后端物理模型完全一致，额外提供:
  - GPU 加速 (cuda / mps)
  - 自动微分支持 (mask.requires_grad=True)
  - 真批量仿真 (batch 维完全向量化)

核心公式:
  M = FFT2c(mask)                     # [B, N, N] 频谱
  MK = M[:, None, :, :] * φ_k         # [B, K, N, N] 频域滤波
  E = IFFT2c(MK)                      # [B, K, N, N] 电场
  I = Σ_k |E_k|²                      # [B, N, N] 强度
  wafer = sigmoid(I)                  # [B, N, N] 光刻像
"""
from __future__ import annotations

import time
from typing import Tuple, Optional, Dict, Any
import numpy as np

import torch
import torch.nn as nn

from .config import OpticalConfig
from .socs import SOCSDecomposition
from .pupil_source import compute_pupil, compute_source


class ScalarSOCSImagingTorch(nn.Module):
    """
    标量 SOCS 加速光刻成像 (PyTorch 后端)。

    作为 nn.Module 子类，支持:
      - model(mask)  前向传播
      - model.to('cuda')  GPU 迁移
      - loss.backward()  梯度反传

    Parameters
    ----------
    kernels      : [K, N, N] 复数 SOCS 核 (空间域)
    eigenvalues  : [K] 特征值
    norm_factor  : 归一化因子 (Σ λ_k)
    sigmoid_tr   : 显影阈值 (默认 0.15)
    sigmoid_a    : Sigmoid 斜率 (默认 85)

    Usage
    -----
        model = ScalarSOCSImagingTorch(kernels, eigenvalues, norm_factor)
        model = model.to('cuda')                    # GPU 加速

        # 单张
        wafer = model(mask)                         # [N, N]

        # 批量
        wafers = model.forward_batch(masks)         # [B, N, N]

        # 梯度
        mask.requires_grad = True
        loss = ((model(mask) - target) ** 2).mean()
        loss.backward()
        grad = mask.grad
    """
    def __init__(
        self,
        kernels: torch.Tensor,           # [K, N, N] complex
        eigenvalues: torch.Tensor,       # [K] real
        norm_factor: float,
        sigmoid_tr: float = 0.15,
        sigmoid_a: float = 85.0,
    ):
        super().__init__()

        K, N, _N = kernels.shape
        assert N == _N, "kernels must be square [K, N, N]"

        # 注册为 buffer (不参与梯度, 但跟随 .to(device))
        self.register_buffer("kernels", kernels)         # [K, N, N] complex
        self.register_buffer("eigenvalues", eigenvalues) # [K] real
        self.register_buffer("norm_factor", torch.tensor(norm_factor))

        self.sigmoid_tr = sigmoid_tr
        self.sigmoid_a = sigmoid_a
        self.N = N
        self.K = K

        # 预计算 FFT2 核，避免重复 FFT
        self.register_buffer(
            "kernels_fft",
            self._fft2c(kernels),  # [K, N, N] complex
        )

    # ---------- FFT 工具 (中心化) ----------

    @staticmethod
    def _fft2c(x: torch.Tensor) -> torch.Tensor:
        """中心化 2D FFT: fftshift(fft2(ifftshift(x)))."""
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )

    @staticmethod
    def _ifft2c(x: torch.Tensor) -> torch.Tensor:
        """中心化 2D IFFT: fftshift(ifft2(ifftshift(x)))."""
        return torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )

    # ---------- 核心前向 ----------

    def _aerial_socs(self, mask: torch.Tensor) -> torch.Tensor:
        """
        单张 mask SOCS 成像 → [N, N] aerial intensity。

        计算路径:
          M = FFT2c(mask)                          # [N, N]
          MK = M[None, :, :] * kernels_fft         # [K, N, N]
          E = IFFT2c(MK)                           # [K, N, N]
          I = Σ λ_k * |E_k|² / norm_factor          # [N, N]
        """
        # 确保 mask 是复数
        if not torch.is_complex(mask):
            mask = mask.to(dtype=self.kernels_fft.real.dtype).to(dtype=self.kernels_fft.dtype)

        M = self._fft2c(mask)                              # [N, N]
        MK = M[None, :, :] * self.kernels_fft           # [K, N, N]
        E = self._ifft2c(MK)                               # [K, N, N]

        # |E|² [K, N, N]
        I_k = E.real ** 2 + E.imag ** 2
        # 加权求和
        aerial = torch.einsum("k,khw->hw", self.eigenvalues, I_k)
        aerial = aerial 
        return aerial

    def _aerial_batch(self, masks: torch.Tensor) -> torch.Tensor:
        """
        批量 SOCS 成像: [B, N, N] → [B, N, N] aerial intensity。

        利用广播一次性完成 B×K 全部 FFT:
          M = FFT2c(masks)                                  # [B, N, N]
          MK = M[:, None, :, :] * kernels_fft[None, ...]    # [B, K, N, N]
          E = IFFT2c(MK)                                    # [B, K, N, N]
          I = Σ λ_k * |E_k|² / norm_factor                   # [B, N, N]
        """
        B = masks.shape[0]

        if not torch.is_complex(masks):
            masks = masks.to(dtype=self.kernels_fft.real.dtype).to(dtype=self.kernels_fft.dtype)

        M = self._fft2c(masks)                                      # [B, N, N]
        MK = M[:, None, :, :] * self.kernels_fft[None, :, :, :]    # [B, K, N, N]
        E = self._ifft2c(MK)                                        # [B, K, N, N]

        I_k = E.real ** 2 + E.imag ** 2                             # [B, K, N, N]
        aerial = torch.einsum("k,bkhw->bhw", self.eigenvalues, I_k) # [B, N, N]
        aerial = aerial 
        return aerial

    # ---------- Sigmoid ----------

    def _sigmoid(self, aerial: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-self.sigmoid_a * (aerial - self.sigmoid_tr)))

    # ---------- 公共接口 ----------

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        单张 mask 成像。

        Parameters
        ----------
        mask : [N, N] 实数或复数 tensor

        Returns
        -------
        wafer : [N, N] sigmoid 后的光刻图像
        """
        return self._sigmoid(self._aerial_socs(mask))

    def forward_batch(self, masks: torch.Tensor) -> torch.Tensor:
        """
        批量成像。

        Parameters
        ----------
        masks : [B, N, N] 实数或复数 tensor

        Returns
        -------
        wafers : [B, N, N] sigmoid 后的光刻图像
        """
        return self._sigmoid(self._aerial_batch(masks))

    def forward_aerial(self, mask: torch.Tensor) -> torch.Tensor:
        """单张 raw aerial intensity (不过 sigmoid)。"""
        return self._aerial_socs(mask)

    def forward_batch_aerial(self, masks: torch.Tensor) -> torch.Tensor:
        """批量 raw aerial intensity。"""
        return self._aerial_batch(masks)


class ScalarImagingTorch:
    """
    PyTorch 成像系统工厂: 从 OpticalConfig 构建 ScalarSOCSImagingTorch。

    用法:
        config = OpticalConfig.default()
        sim = ScalarImagingTorch(config)
        sim.prepare(device="cuda")
        model = sim.get_model()

        wafer = model(mask_tensor)
        wafers = model.forward_batch(masks_batch)

    Attributes
    ----------
    config  : 光学配置
    model   : ScalarSOCSImagingTorch 实例 (prepare 后可用)
    N       : 网格大小
    """
    def __init__(self, config: OpticalConfig, N: Optional[int] = None):
        self.config = config
        self.N = N or config.grid_n or 257
        self.model: Optional[ScalarSOCSImagingTorch] = None
        self._prepared = False

    def prepare(
        self,
        K: Optional[int] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        svd_method: str = "randomized",
    ) -> ScalarSOCSImagingTorch:
        """
        预计算 SOCS 核并构建 PyTorch 模型。

        Parameters
        ----------
        K          : SOCS 核数 (None 自动)
        device     : "cpu" / "cuda" / "mps"
        dtype      : torch.float32 或 torch.float64
        svd_method : "randomized" (快速) | "full" (精确)

        Returns
        -------
        model : ScalarSOCSImagingTorch 实例
        """
        if self._prepared and self.model is not None:
            print("[ScalarImagingTorch] Already prepared.")
            return self.model

        cfg = self.config
        K = K or cfg.socs_k
        if K is None:
            K = 50 if cfg.mask.pixel_size_nm == 4.0 else 60

        print("=" * 60)
        print(f"[ScalarImagingTorch] Preparing optical system (device={device})...")
        print(f"  Grid:          {self.N}×{self.N}")
        print(f"  Pixel size:    {cfg.mask.pixel_size_nm} nm")
        print(f"  NA:            {cfg.optics.na}")
        print(f"  Wavelength:    {cfg.optics.wavelength_nm} nm")
        print(f"  SOCS kernels:  {K}, SVD: {svd_method}")
        print(f"  Device:        {device}")
        print("-" * 60)

        # 用 numpy 后端计算光瞳+光源+SOCS (物理计算不分前后端)
        t0 = time.time()
        pixel_size = cfg.mask.pixel_size_nm

        pupil = compute_pupil(
            N=self.N, pixel_size_nm=pixel_size,
            na=cfg.optics.na, wavelength_nm=cfg.optics.wavelength_nm,
            aberrations=cfg.optics.aberrations,
        )
        print(f"  [1/3] Pupil computed. ({time.time()-t0:.2f}s)")

        t0 = time.time()
        source = compute_source(
            N=self.N, pixel_size_nm=pixel_size,
            na=cfg.optics.na, wavelength_nm=cfg.optics.wavelength_nm,
            source_cfg=cfg.source,
        )
        print(f"  [2/3] Source computed, {source.active_count} active points. ({time.time()-t0:.2f}s)")

        t0 = time.time()
        socs = SOCSDecomposition(pupil=pupil, source=source, K=K, method=svd_method)
        print(f"  [3/3] SOCS done. ({time.time()-t0:.2f}s)")

        # 转为 torch tensor
        kernels_np = socs.get_kernels()         # [K, N, N] complex128
        eigenvalues_np = socs.get_eigenvalues()  # [K] float64
        norm_factor = float(eigenvalues_np.sum())

        # 转换精度
        np_dtype = np.float32 if dtype == torch.float32 else np.float64
        kernels_t = torch.from_numpy(
            np.stack([kernels_np.real.astype(np_dtype), kernels_np.imag.astype(np_dtype)], axis=-1)
        )  # [K, N, N, 2]
        # 转为 complex tensor
        kernels_t = torch.view_as_complex(kernels_t.contiguous())

        eigenvalues_t = torch.from_numpy(eigenvalues_np.astype(np_dtype))

        # 构建模型
        self.model = ScalarSOCSImagingTorch(
            kernels=kernels_t,
            eigenvalues=eigenvalues_t,
            norm_factor=norm_factor,
            sigmoid_tr=cfg.resist.threshold,
            sigmoid_a=cfg.resist.alpha,
        )
        self.model = self.model.to(device=device, dtype=dtype)

        # 统计
        n_params = sum(p.numel() for p in self.model.parameters())
        n_buffers = sum(b.numel() for b in self.model.buffers())
        mem_mb = sum(
            b.element_size() * b.numel() for b in self.model.buffers()
        ) / (1024 * 1024)

        print(f"  Model: {n_params:,} params, {n_buffers:,} buffer elements")
        print(f"  Buffer memory: {mem_mb:.1f} MB")
        print("=" * 60)

        self._prepared = True
        return self.model

    def get_model(self) -> ScalarSOCSImagingTorch:
        """返回已构建的 PyTorch 模型 (必须先调用 prepare)。"""
        if self.model is None:
            raise RuntimeError("Call .prepare() first.")
        return self.model
