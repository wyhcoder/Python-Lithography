"""
LithographySimulator：把 SimulationConfig 转换成完整的仿真组件树。

持有：
    grid, mask, pupil, source (Illumination), vector_transfer (VectorTransfer)

并把 source 的有效点预算好（fs_phys / gs_phys / source_weights），
供后续 VectorForwardImaging 或自定义前向直接复用。
"""
from __future__ import annotations

import torch

from .config import SimulationConfig
from .grid import Grid
from .mask import Mask
from .pupil import Pupil
from .source import Illumination, SourceType
from .vector_transfer import VectorTransfer


class LithographySimulator:
    def __init__(
        self,
        config: SimulationConfig,
        weight_threshold: float = 0.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.weight_threshold = float(weight_threshold)
        self.device = device
        self.real_dtype = dtype

        print("[1/5] 构造 grid ...")
        self.grid = Grid(
            N=config.system.mask_size,
            dx=config.system.pixel_size_nm,
            device=device,
            dtype=dtype,
        )

        print("[2/5] 构造 mask ...")
        self.mask = Mask(
            mask_dir=config.mask.mask_dir,
            mask_name=config.mask.mask_name,   # 文件名，不是完整 Path
            mask_size=config.system.mask_size,
            normalize=config.mask.normalize,
        )

        print("[3/5] 构造 pupil ...")
        self.pupil = Pupil(
            NA=config.pupil.na,
            wave_length=config.system.wavelength_nm,
            refractive_index=config.pupil.n_image,
            grid=self.grid,
            aberrations=config.pupil.aberration,
            defocus_nm=config.pupil.defocus_nm,
            device=device,
        )

        print("[4/5] 构造 source ...")
        self.source = Illumination(
            pupil=self.pupil,
            grid=self.grid,
            source_type=SourceType(config.source.type),
            sigma_in=config.source.sigma_in,
            sigma_out=config.source.sigma_out,
            smoothing=config.source.smoothing,
            upsample=config.source.upsample,
        )

        print("[5/5] 构造 vector transfer ...")
        self.vector_transfer = VectorTransfer(
            pupil=self.pupil,
            grid=self.grid,
        )

        self._build_active_source_points()

    # ---------- source 点预算 ----------

    def _build_active_source_points(self) -> None:
        """把 Illumination 的 source 权重图拉平，筛出有效点，换算成物理频率。"""
        sigma_fx, sigma_fy = self.source.get_axes()
        weight_map = self.source.get_weights()  # [Ny, Nx]

        sy_grid, sx_grid = torch.meshgrid(sigma_fy, sigma_fx, indexing="ij")
        sx_flat = sx_grid.reshape(-1)
        sy_flat = sy_grid.reshape(-1)
        w_flat = weight_map.reshape(-1)

        keep = w_flat > self.weight_threshold
        sx_flat = sx_flat[keep]
        sy_flat = sy_flat[keep]
        w_flat = w_flat[keep]

        if w_flat.numel() == 0:
            raise ValueError("Illumination 中没有有效 source 点（权重全为 0）")

        f_max = self.pupil.NA / self.pupil.wave_length
        self.fs_phys = (sx_flat * f_max).to(self.real_dtype)   # [Ns_active]
        self.gs_phys = (sy_flat * f_max).to(self.real_dtype)   # [Ns_active]
        self.source_weights = w_flat.to(self.real_dtype)        # [Ns_active]

    # ---------- 便捷属性 ----------

    @property
    def mask_tensor(self) -> torch.Tensor:
        """把 Mask 的 numpy 数组转成 torch tensor（居中放置在仿真网格上）。"""
        return torch.from_numpy(self.mask.mask).to(
            device=self.device, dtype=self.real_dtype
        )
