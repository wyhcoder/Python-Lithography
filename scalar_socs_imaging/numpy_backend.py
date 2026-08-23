"""
标量 SOCS 成像 — numpy 后端

核心流程:
  1) 加载/生成 mask
  2) 计算光瞳 + 光源
  3) SOCS 分解 → K 个相干核
  4) 逐核 FFT 卷积 → 加权求和 → aerial image
  5) Sigmoid 显影 → wafer image

支持两种仿真模式:
  - single:  单张 mask 成像
  - batch:   多张 mask 批量成像 (共享 SOCS 核)
"""
from __future__ import annotations

import time
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .config import OpticalConfig
from .pupil_source import compute_pupil, compute_source, PupilFunction, SourceMap
from .socs import SOCSDecomposition


class ScalarImagingNumpy:
    """
    标量 SOCS 加速光刻成像 (numpy 后端)。

    用法:
        config = OpticalConfig.default()
        sim = ScalarImagingNumpy(config)
        sim.prepare()                         # 预计算 SOCS 核
        aerial, wafer = sim.forward(mask)     # 单张成像
        aerials, wafers = sim.forward_batch(masks)  # 批量成像

    Attributes
    ----------
    config     : 光学配置
    N          : 网格大小
    pupil      : 光瞳函数
    source     : 光源分布
    socs       : SOCS 分解结果
    prepared   : 是否已预计算
    """
    def __init__(self, config: OpticalConfig, N: Optional[int] = None):
        """
        Parameters
        ----------
        config : 光学系统配置
        N      : 网格大小 (None 则从 config.grid_n 读取)
        """
        self.config = config
        self.N = N or config.grid_n or 257  # 默认 257×257
        self.pixel_size = config.mask.pixel_size_nm

        self.pupil: Optional[PupilFunction] = None
        self.source: Optional[SourceMap] = None
        self.socs: Optional[SOCSDecomposition] = None
        self._fft_kernels: Optional[np.ndarray] = None  # [K, N, N] 预计算 FFT2 核
        self._eigenvalues: Optional[np.ndarray] = None   # [K]
        self._norm_factor: float = 1.0
        self.prepared: bool = False

    # ---------- 预计算 ----------

    def prepare(self, K: Optional[int] = None, svd_method: str = "randomized") -> None:
        """
        预计算光瞳、光源、SOCS 核 (幂等: 重复调用自动跳过)。

        Parameters
        ----------
        K          : SOCS 核保留数 (None 则从 config.socs_k 读取)
        svd_method : "randomized" (快速) | "full" (精确)
        """
        if self.prepared:
            print("[ScalarImagingNumpy] Already prepared, skipping.")
            return

        cfg = self.config
        K = K or cfg.socs_k
        if K is None:
            K = 50 if self.pixel_size == 4.0 else 60

        print("=" * 60)
        print("[ScalarImagingNumpy] Preparing optical system...")
        print(f"  Grid:          {self.N}×{self.N}")
        print(f"  Pixel size:    {self.pixel_size} nm")
        print(f"  NA:            {cfg.optics.na}")
        print(f"  Wavelength:    {cfg.optics.wavelength_nm} nm")
        print(f"  Source type:   {cfg.source.type}")
        print(f"  Sigma:         {cfg.source.sigma_in} - {cfg.source.sigma_out}")
        print(f"  Resist thr:    {cfg.resist.threshold}, alpha: {cfg.resist.alpha}")
        print(f"  SOCS kernels:  {K}, SVD method: {svd_method}")
        print("-" * 60)

        # 1) 光瞳
        t0 = time.time()
        self.pupil = compute_pupil(
            N=self.N,
            pixel_size_nm=self.pixel_size,
            na=cfg.optics.na,
            wavelength_nm=cfg.optics.wavelength_nm,
            aberrations=cfg.optics.aberrations,
        )
        print(f"  [1/3] Pupil computed. ({time.time()-t0:.2f}s)")

        # 2) 光源
        t0 = time.time()
        self.source = compute_source(
            N=self.N,
            pixel_size_nm=self.pixel_size,
            na=cfg.optics.na,
            wavelength_nm=cfg.optics.wavelength_nm,
            source_cfg=cfg.source,
        )
        print(f"  [2/3] Source computed, {self.source.active_count} active points. ({time.time()-t0:.2f}s)")

        # 3) SOCS 分解
        t0 = time.time()
        self.socs = SOCSDecomposition(
            pupil=self.pupil,
            source=self.source,
            K=K,
            method=svd_method,
        )
        self._eigenvalues = self.socs.get_eigenvalues()
        self._norm_factor = self._eigenvalues.sum()
        self._fft_kernels = self.socs.get_fft_kernels()  # [K, N, N]
        print(f"  [3/3] SOCS decomposition done. ({time.time()-t0:.2f}s)")

        # 内存统计
        mem_mb = self._fft_kernels.nbytes / (1024 * 1024)
        print(f"  FFT kernel cache: {mem_mb:.1f} MB")
        print("=" * 60)

        self.prepared = True

    # ---------- 单张成像 ----------

    def forward(
        self,
        mask: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        单张 mask 的 SOCS 加速成像。

        Parameters
        ----------
        mask    : [N, N] 实数 mask (值域 [0, 1] 或 {0, 1})
        verbose : 是否打印计算统计

        Returns
        -------
        (aerial_image, wafer_image): 均为 [N, N] 实数
        """
        if not self.prepared:
            raise RuntimeError("Call .prepare() before .forward()")

        t0 = time.time()

        # 1) mask → 频谱
        mask_fft = np.fft.fft2(mask.astype(np.complex128))

        # 2) 频域点乘: [1, N, N] × [K, N, N] → [K, N, N]
        spectra = mask_fft[None, :, :] * self._fft_kernels

        # 3) IFFT → 空间域电场: [K, N, N]
        E_fields = np.fft.ifftshift(
            np.fft.ifft2(spectra, axes=(-2, -1)),
            axes=(-2, -1),
        )

        # 4) 强度 = Σ λ_k · |E_k|²
        intensities = E_fields.real**2 + E_fields.imag**2  # [K, N, N]
        aerial = np.einsum("k,khw->hw", self._eigenvalues, intensities)
        aerial /= self._norm_factor

        # 5) Sigmoid 显影
        wafer = self._sigmoid(aerial)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"  [forward] mask [{mask.shape}] → aerial [{aerial.shape}], "
                f"aerial max={aerial.max():.4f}, wafer max={wafer.max():.4f}, "
                f"time={elapsed:.3f}s"
            )

        return aerial, wafer

    # ---------- 批量成像 ----------

    def forward_batch(
        self,
        masks: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量 SOCS 成像: [B, N, N] 多张 mask 一次性计算。

        Parameters
        ----------
        masks   : [B, N, N] 实数 batch
        verbose : 是否打印统计

        Returns
        -------
        (aerials, wafers): [B, N, N] 各一张
        """
        if not self.prepared:
            raise RuntimeError("Call .prepare() before .forward_batch()")

        B = masks.shape[0]
        t0 = time.time()

        # 1) batch FFT2: [B, N, N] → [B, N, N]
        mask_fft = np.fft.fft2(masks.astype(np.complex128), axes=(-2, -1))

        # 2) 广播频域点乘: [B, 1, N, N] × [1, K, N, N] → [B, K, N, N]
        spectra = mask_fft[:, None, :, :] * self._fft_kernels[None, :, :, :]

        # 3) batch IFFT → [B, K, N, N]
        E_fields = np.fft.ifftshift(
            np.fft.ifft2(spectra, axes=(-2, -1)),
            axes=(-2, -1),
        )

        # 4) 强度: [B, K, N, N] → [B, N, N]
        intensities = E_fields.real**2 + E_fields.imag**2
        aerials = np.einsum("k,bkhw->bhw", self._eigenvalues, intensities)
        # aerials /= self._norm_factor

        # 5) Sigmoid
        wafers = self._sigmoid(aerials)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"  [forward_batch] {B} masks [{masks.shape}] → "
                f"{B} aerials/wafers, time={elapsed:.3f}s "
                f"({elapsed/B*1000:.1f} ms/mask)"
            )

        return aerials, wafers

    # ---------- 内部工具 ----------

    def _sigmoid(self, aerial: np.ndarray) -> np.ndarray:
        """Sigmoid 光刻胶模型: wafer = 1 / (1 + exp(-α*(aerial - thr)))."""
        cfg = self.config.resist
        return 1.0 / (1.0 + np.exp(-cfg.alpha * (aerial - cfg.threshold)))

    # ---------- 诊断 ----------

    def get_energy_spectrum(self) -> Dict[str, Any]:
        """返回 SOCS 能量谱摘要。"""
        if self.socs is None:
            return {}
        return self.socs.summary()

    def print_energy_report(self) -> None:
        """打印能量分布报告。"""
        if self.socs is None:
            print("[WARN] SOCS not computed yet.")
            return

        eig = self._eigenvalues
        total = eig.sum()
        cumsum = np.cumsum(eig) / total * 100

        print("\n--- SOCS Energy Spectrum ---")
        print(f"  Total kernels:  {len(eig)}")
        print(f"  Total energy:   {total:.6e}")
        print(f"  λ_max / λ_min:  {eig[0]:.2e} / {eig[-1]:.2e}")
        print("  Cumulative energy:")
        for pct in [50, 80, 90, 95, 99]:
            idx = np.searchsorted(cumsum, pct)
            print(f"    {pct:5.1f}% → {idx:3d} kernels, λ[{idx}]={eig[min(idx,len(eig)-1)]:.2e}")
        print("-" * 30)
