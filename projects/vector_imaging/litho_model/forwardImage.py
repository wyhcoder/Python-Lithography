"""
矢量 Abbe 前向成像（vector aerial imaging）。

支持自动微分：
- 只要 `mask`（或下游传入的 mask 张量）`requires_grad=True`，
  最终 aerial intensity 即可对 mask 求梯度。
- pupil 中的 aberrations / defocus 若以可学习方式接入，也可一并求导
  （需要将 pupil 内部的相关属性改为 nn.Parameter / 可微参数；本文件
  对此保持开放，不做强约束）。

流程（Abbe 求和）：
对每个 source 点 (fs, gs)（单位 cycles/nm）：
    1) tilted_mask(x,y) = mask(x,y) * exp(j 2pi (fs x + gs y))
    2) U(fx,gy) = FFT2( tilted_mask )
    3) E_p(x,y) = IFFT2( U * T_p ),  p in {x,y,z}
    4) I_s     = |Ex|^2 + |Ey|^2 + |Ez|^2
最终 aerial = sum_s w_s * I_s

实现上把所有有效 source 点堆成一个 batch 维 Ns_active，避免 Python for。
"""
from __future__ import annotations

import math
from turtle import forward
from typing import Tuple

import torch
import torch.nn as nn

from tool.fft_tool import fft2c, ifft2c

from .grid import Grid
from .pupil import Pupil
from .source import Illumination
from .vector_transfer import VectorTransfer
from .socs import VectorSOCS
from .socs import ScalarSOCS


class VectorForwardImaging(nn.Module):
    """
    端到端矢量成像模型：mask -> aerial intensity。

    参数
    ----
    grid       : Mask，提供空间/频率坐标。
    pupil      : Pupil，含像差/离焦的复光瞳。
    illumination : Illumination，提供 source 权重图（归一化频率 sigma 上）。
    vector_transfer : VectorTransfer，提供 [3,2,N,N] 偏振变换矩阵。
    weight_threshold : 小于该阈值的 source 点不参与求和（节省计算）。
    chunk_size : 一次 batch 内处理多少个 source 点；None 表示一次处理全部。
                 显存紧张时可调小（如 32/64），不影响数值结果。
    """

    def __init__(
        self,
        grid: Grid,
        pupil: Pupil,
        illumination: Illumination,
        vector_transfer: VectorTransfer,
        weight_threshold: float = 0.0,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.grid = grid
        self.pupil = pupil
        self.illumination = illumination
        self.vector_transfer = vector_transfer

        self.weight_threshold = float(weight_threshold)
        self.chunk_size = chunk_size

        self.real_dtype = pupil.real_dtype
        self.complex_dtype = pupil.complex_dtype
        self.device = pupil.device

        self._build_active_source_points()

    # ---------- 准备有效 source 点列表 ----------

    def _build_active_source_points(self) -> None:
        """把 illumination 上的 (sigma_fx, sigma_fy, weight) 拉直成 1D，
        筛选权重大于阈值的点，并一次性转换为物理频率 (fs, gs)。"""
        sigma_fx, sigma_fy = self.illumination.get_axes()
        weight_map = self.illumination.get_weights()  # [Ny, Nx]

        # indexing="ij" 后 dim0=fy, dim1=fx
        sy_grid, sx_grid = torch.meshgrid(sigma_fy, sigma_fx, indexing="ij")
        sx_flat = sx_grid.reshape(-1)
        sy_flat = sy_grid.reshape(-1)
        w_flat = weight_map.reshape(-1)

        if self.weight_threshold > 0.0:
            keep = w_flat > self.weight_threshold
        else:
            keep = w_flat > 0.0

        sx_flat = sx_flat[keep]
        sy_flat = sy_flat[keep]
        w_flat = w_flat[keep]

        if w_flat.numel() == 0:
            raise ValueError("Illumination 中没有有效 source 点（权重全为 0）")

        # 转成物理频率 cycles/nm
        f_max = self.pupil.NA / self.pupil.wave_length
        fs_phys = (sx_flat * f_max).to(self.real_dtype)
        gs_phys = (sy_flat * f_max).to(self.real_dtype)
        weights = w_flat.to(self.real_dtype)

        self.fs_phys = fs_phys  # [Ns_active]
        self.gs_phys = gs_phys  # [Ns_active]
        self.source_weights = weights  # [Ns_active]

    # ---------- 单次前向 ----------

    def _aerial_for_jones(
        self,
        mask: torch.Tensor,
        Ex_in: complex | float,
        Ey_in: complex | float,
    ) -> torch.Tensor:
        """对给定单一 Jones 向量做 Abbe 求和，返回 [N, N] 实强度图。"""
        device = self.device
        cdtype = self.complex_dtype
        rdtype = self.real_dtype

        N = self.grid.N

        # ----- 1) 准备 mask（复数）-----
        if torch.is_complex(mask):
            mask_c = mask.to(device=device, dtype=cdtype)
        else:
            mask_c = mask.to(device=device, dtype=rdtype).to(cdtype)

        # ----- 2) 偏振传递函数 T(fx,fy) -----
        T = self.vector_transfer.transfer_for_jones(
            Ex_in=Ex_in,
            Ey_in=Ey_in,
            apply_pupil=True,
        )  # [3, N, N]

        # ----- 3) 空间坐标，需要中心化（与 grid 对齐）-----
        # 注意：grid.x/y 已经中心化（原点居中），与 fft2 直接配合时
        # 等价于忽略全局相位，对最终强度无影响。
        X = self.grid.x.to(rdtype)
        Y = self.grid.y.to(rdtype)

        fs = self.fs_phys  # [Ns]
        gs = self.gs_phys  # [Ns]
        ws = self.source_weights  # [Ns]
        Ns = fs.numel()

        chunk = self.chunk_size if self.chunk_size is not None else Ns
        chunk = max(1, int(chunk))

        aerial = torch.zeros((N, N), device=device, dtype=rdtype)

        two_pi = 2.0 * math.pi

        for start in range(0, Ns, chunk):
            end = min(start + chunk, Ns)
            fs_b = fs[start:end]  # [B]
            gs_b = gs[start:end]
            ws_b = ws[start:end]

            # ramp: [B, N, N], real phase -> complex 广播机制
            phase = two_pi * (fs_b[:,None,None] * X[None, :,:] + gs_b[:,None,None] * Y[None, :,:])
            ramp = torch.complex(torch.cos(phase), torch.sin(phase)).to(cdtype)
            # tilted mask: [B, N, N]
            u = mask_c.unsqueeze(0) * ramp

            # 中心化 FFT，与中心化的 T(fx,fy) 频率网格保持一致
            U = fft2c(u)  # [B, N, N]

            # 三分量逆变换：[B, 3, N, N] = U[:, None] * T[None]
            UT = U.unsqueeze(1) * T.unsqueeze(0)  # [B, 3, N, N]
            E = ifft2c(UT)  # [B, 3, N, N]

            I_s = (E.real * E.real + E.imag * E.imag).sum(dim=1)  # [B, N, N]

            # 加权求和：sum_s w_s I_s
            aerial = aerial + (ws_b[:, None, None] * I_s).sum(dim=0)

        return aerial   # 返回物理强度（未过 sigmoid）

    # ---------- 单 mask 接口 ----------

    def forward(
        self,
        mask: torch.Tensor,
        jones: Tuple[complex | float, complex | float] = (1.0, 0.0),
    ) -> torch.Tensor:
        """单 mask、单偏振成像，返回 [N, N] sigmoid 后的光刻图像。"""
        return self.sigmoid(self._aerial_for_jones(mask, jones[0], jones[1]))

    def forward_unpolarized(self, mask: torch.Tensor) -> torch.Tensor:
        """单 mask 非偏振成像 = sigmoid(0.5*(Ix+Iy))，返回 [N, N]。"""
        Ix = self._aerial_for_jones(mask, 1.0, 0.0)
        Iy = self._aerial_for_jones(mask, 0.0, 1.0)
        return self.sigmoid(0.5 * (Ix + Iy))

    # ---------- 批量 mask 接口 ----------

    def forward_batch(
        self,
        masks: torch.Tensor,
        jones: Tuple[complex | float, complex | float] = (1.0, 0.0),
    ) -> torch.Tensor:
        """
        批量单偏振成像。

        参数
        ----
        masks : [B, N, N] 实数或复数 tensor，B 为批量大小。
        jones : 入射偏振 Jones 向量 (Ex_in, Ey_in)。

        返回
        ----
        wafer : [B, N, N] 实数 tensor，经 sigmoid 后的光刻图像。
        """
        return self._aerial_batch(masks, jones[0], jones[1])

    def forward_batch_unpolarized(self, masks: torch.Tensor) -> torch.Tensor:
        """
        批量非偏振成像 = 0.5*(Ix+Iy)，返回 [B, N, N]。
        """
        Ix = self._aerial_batch(masks, 1.0, 0.0)
        Iy = self._aerial_batch(masks, 0.0, 1.0)
        return 0.5 * (Ix + Iy)

    def _aerial_batch(
        self,
        masks: torch.Tensor,
        Ex_in: complex | float,
        Ey_in: complex | float,
    ) -> torch.Tensor:
        """
        对 [B, N, N] 的 mask batch 批量做 Abbe 成像。

        实现方式：逐张调用 _aerial_for_jones（光学系统固定，不依赖 mask 形状），
        用 torch.stack 拼合成 [B, N, N] 后统一过 sigmoid 返回。
        内存受限时这样做最安全；若要更快可以在 _aerial_for_jones 内增加 mask_batch 维度，
        但内存占用会乘以 B，建议使用当前做法配合外部 DataLoader 分批调用。
        """
        device = self.device
        rdtype = self.real_dtype
        B = masks.shape[0]

        aerials = []
        for i in range(B):
            aerials.append(self._aerial_for_jones(masks[i], Ex_in, Ey_in))
        aerial_batch = torch.stack(aerials, dim=0)  # [B, N, N]
        return self.sigmoid(aerial_batch)

    # ---------- 工具 ----------

    def sigmoid(self, aerial: torch.Tensor) -> torch.Tensor:
        """sigmoid 光刻模型，threshold/alpha 从 config 读取（当前使用 hardcode 默认值）。"""
        tr = 0.15
        alpha = 85
        return 1 / (1 + torch.exp(-alpha * (aerial - tr)))


# ═══════════════════════════════════════════════════════════════
#  SOCS 加速版前向成像
# ═══════════════════════════════════════════════════════════════

class VectorSOCSImaging(nn.Module):
    """
    矢量 SOCS 加速前向成像：mask -> aerial intensity。

    与 VectorForwardImaging 接口完全相同，但用预计算的 SOCS 核函数替代
    Abbe source 求和，将 N_source 次 FFT 对压缩为 K 次（K ≪ N_source）。

    前向公式：
        I(x) = sum_k sum_p |IFFT{ M(f) * φ_k_p(f) }|²
             = sum_k ||IFFT{ M * φ_k }||²_Frobenius（分量已合并进 φ_k）

    其中 φ_k [3, N, N] = S_k * U_k（左奇异向量 × 奇异值）

    参数
    ----
    socs       : 预计算好的 VectorSOCS 实例
    grid       : Grid 实例（提供 N / x / y）
    sigmoid_tr : sigmoid 阈值（默认 0.15）
    sigmoid_a  : sigmoid 斜率（默认 85）
    """

    def __init__(
        self,
        socs: VectorSOCS,
        grid: Grid,
        sigmoid_tr: float = 0.15,
        sigmoid_a: float = 85.0,
    ):
        super().__init__()
        self.socs = socs
        self.grid = grid
        self.real_dtype = socs.real_dtype
        self.complex_dtype = socs.complex_dtype
        self.device = socs.device
        self.sigmoid_tr = sigmoid_tr
        self.sigmoid_a = sigmoid_a

    # ---------- 核心前向 ----------

    def _aerial_socs(self, mask: torch.Tensor) -> torch.Tensor:
        """
        单 mask 的 SOCS 前向，返回 [N, N] aerial intensity。
        支持 [K, 3, N, N]（单偏振）和 [K, 6, N, N]（非偏振，前3=x后3=y）两种核。
        """
        device = self.device
        cdtype = self.complex_dtype
        rdtype = self.real_dtype
        N = self.grid.N

        if torch.is_complex(mask):
            mask_c = mask.to(device=device, dtype=cdtype)
        else:
            mask_c = mask.to(device=device, dtype=rdtype).to(cdtype)
        M = fft2c(mask_c)  # [N, N]

        kernels = self.socs.get_kernels()  # [K, C, N, N], C=3(单偏振) or 6(非偏振)
        MK = M[None, None, :, :] * kernels   # [K, C, N, N]
        E = ifft2c(MK)                        # [K, C, N, N]
        aerial = (E.real * E.real + E.imag * E.imag).sum(dim=(0, 1))  # [N, N]

        # 非偏振 SOCS：A_total = [A_x; A_y]，前向得到的是 Ix+Iy，需除以 2 得非偏振均值
        C = kernels.shape[1]
        if C == 6:
            aerial = aerial * 0.5

        return aerial

    def sigmoid(self, aerial: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-self.sigmoid_a * (aerial - self.sigmoid_tr)))

    # ---------- 单 mask 接口 ----------

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """非偏振等效成像（偏振已合并进 SOCS 核），返回 [N, N]。"""
        aerial = self._aerial_socs(mask)
        return self.sigmoid(aerial)

    # ---------- 批量 mask 接口 ----------

    def forward_batch(self, masks: torch.Tensor) -> torch.Tensor:
        """
        批量 SOCS 成像。

        masks : [B, N, N]
        返回  : [B, N, N] sigmoid 后的光刻图像

        利用 M 的 batch 维一次性完成所有 mask 的频谱计算，
        再与 K 个核函数做广播乘法，真正做到 B×K 完全向量化。
        """
        device = self.device
        cdtype = self.complex_dtype
        rdtype = self.real_dtype
        B = masks.shape[0]

        if torch.is_complex(masks):
            masks_c = masks.to(device=device, dtype=cdtype)
        else:
            masks_c = masks.to(device=device, dtype=rdtype).to(cdtype)

        # [B, N, N] 频谱
        M_batch = fft2c(masks_c)                         # [B, N, N]

        kernels = self.socs.get_kernels()                # [K, 3, N, N]
        K = kernels.shape[0]

        # 广播：[B, 1, 1, N, N] × [1, K, C, N, N] → [B, K, C, N, N]
        MK = M_batch[:, None, None, :, :] * kernels[None, :, :, :, :]
        E = ifft2c(MK)                                   # [B, K, C, N, N]

        # sum over K and C → [B, N, N]
        I = (E.real * E.real + E.imag * E.imag).sum(dim=(1, 2))  # [B, N, N]

        return self.sigmoid(I)

    # ---------- 精度诊断 ----------

    def compare_with_abbe(
        self,
        mask: torch.Tensor,
        abbe_model: "VectorForwardImaging",
    ) -> dict:
        """
        与 Abbe 参考模型对比，返回 NMSE 和最大绝对误差。
        仅供调试，不用于生产路径。
        """
        with torch.no_grad():
            I_socs = self._aerial_socs(mask)
            I_abbe = abbe_model._aerial_for_jones(mask, 1.0, 0.0)
            I_abbe = I_abbe + abbe_model._aerial_for_jones(mask, 0.0, 1.0)
            I_abbe = 0.5 * I_abbe

            diff = I_socs - I_abbe
            nmse = (diff ** 2).mean() / (I_abbe ** 2).mean().clamp_min(1e-12)
            max_err = diff.abs().max()
        return {
            "nmse": nmse.item(),
            "max_abs_err": max_err.item(),
            "socs_max": I_socs.max().item(),
            "abbe_max": I_abbe.max().item(),
        }

class ScalarForwardImaging(nn.Module):
    '''标量前向成像：mask -> aerial intensity。'''
    def __init__(
        self,
        grid: Grid,
        pupil: Pupil,
        illumination: Illumination,
        weight_threshold: float = 0.0,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.grid = grid
        self.pupil = pupil
        self.illumination = illumination

        self.weight_threshold = float(weight_threshold)
        self.chunk_size = chunk_size
        
        self.real_dtype = grid.real_dtype
        self.complex_dtype = grid.complex_dtype
        self.device = grid.device

        self._build_active_source_points()
    def _build_active_source_points(self) -> None:
        """把 illumination 上的 (sigma_fx, sigma_fy, weight) 拉直成 1D，
        筛选权重大于阈值的点，并一次性转换为物理频率 (fs, gs)。"""
        sigma_fx, sigma_fy = self.illumination.get_axes()
        weight_map = self.illumination.get_weights()  # [Ny, Nx]

        # indexing="ij" 后 dim0=fy, dim1=fx
        sy_grid, sx_grid = torch.meshgrid(sigma_fy, sigma_fx, indexing="ij")
        sx_flat = sx_grid.reshape(-1)
        sy_flat = sy_grid.reshape(-1)
        w_flat = weight_map.reshape(-1)

        if self.weight_threshold > 0.0:
            keep = w_flat > self.weight_threshold
        else:
            keep = w_flat > 0.0

        sx_flat = sx_flat[keep]
        sy_flat = sy_flat[keep]
        w_flat = w_flat[keep]

        if w_flat.numel() == 0:
            raise ValueError("Illumination 中没有有效 source 点（权重全为 0）")

        # 转成物理频率 cycles/nm
        f_max = self.pupil.NA / self.pupil.wave_length
        fs_phys = (sx_flat * f_max).to(self.real_dtype)
        gs_phys = (sy_flat * f_max).to(self.real_dtype)
        weights = w_flat.to(self.real_dtype)

        self.fs_phys = fs_phys  # [Ns_active]
        self.gs_phys = gs_phys  # [Ns_active]
        self.source_weights = weights  # [Ns_active]

    def _aerial_abbe(self, mask: torch.Tensor) -> torch.Tensor:
        """
        单 mask 的标量 Abbe 前向，返回 [N, N] raw aerial intensity。

        与 VectorForwardImaging._aerial_for_jones 同结构，但用标量复光瞳 H(f)
        替代矢量传递矩阵 T(f)，只算单个电场分量:
            E_s(x) = IFFT2{ FFT2(mask · ramp_s) · H(f) }
            I_s    = |E_s|²
            aerial = Σ_s w_s · I_s
        """
        device = self.device
        cdtype = self.complex_dtype
        rdtype = self.real_dtype
        N = self.grid.N

        # 1) mask -> 复数
        if torch.is_complex(mask):
            mask_c = mask.to(device=device, dtype=cdtype)
        else:
            mask_c = mask.to(device=device, dtype=rdtype).to(cdtype)

        # 2) 复光瞳 H(f), [N, N]
        H = self.pupil.pupil.to(cdtype)

        X = self.grid.x.to(rdtype)
        Y = self.grid.y.to(rdtype)

        fs = self.fs_phys
        gs = self.gs_phys
        ws = self.source_weights
        Ns = fs.numel()
        chunk = self.chunk_size if self.chunk_size is not None else Ns
        chunk = max(1, int(chunk))

        aerial = torch.zeros((N, N), device=device, dtype=rdtype)
        two_pi = 2.0 * math.pi

        # 3) 分块累加
        for start in range(0, Ns, chunk):
            end = min(start + chunk, Ns)
            fs_b = fs[start:end]                                    # [B]
            gs_b = gs[start:end]
            ws_b = ws[start:end]

            # ramp_s(x): [B, N, N]
            phase = two_pi * (
                fs_b[:, None, None] * X[None, :, :]
                + gs_b[:, None, None] * Y[None, :, :]
            )
            ramp = torch.complex(torch.cos(phase), torch.sin(phase)).to(cdtype)

            # tilted mask: [B, N, N]
            u = mask_c.unsqueeze(0) * ramp

            # 中心化 FFT 与中心化的 H(f) 网格保持一致
            U = fft2c(u)                                            # [B, N, N]

            # 标量传递: E_s(x) = IFFT2{ U · H }
            E = ifft2c(U * H.unsqueeze(0))                          # [B, N, N]
            I_s = (E.real * E.real + E.imag * E.imag)               # [B, N, N]

            # 加权求和: Σ w_s · I_s
            aerial = aerial + (ws_b[:, None, None] * I_s).sum(dim=0)

        return aerial

    # ---------- 工具 ----------

    def sigmoid(self, aerial: torch.Tensor) -> torch.Tensor:
        """sigmoid resist 模型 (与 VectorForwardImaging 默认值保持一致)。"""
        tr = 0.15
        alpha = 85
        return 1.0 / (1.0 + torch.exp(-alpha * (aerial - tr)))

    # ---------- 单 mask 接口 ----------

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """单 mask 标量 Abbe 成像，返回 [N, N] sigmoid 后的光刻图像。"""
        return self.sigmoid(self._aerial_abbe(mask))

    # ---------- 批量 mask 接口 ----------

    def forward_batch(self, masks: torch.Tensor) -> torch.Tensor:
        """
        批量标量 Abbe 成像，逐 mask 调用 _aerial_abbe 后统一过 sigmoid。

        Parameters
        ----------
        masks : [B, N, N] real or complex tensor

        Returns
        -------
        wafer : [B, N, N] real tensor
        """
        B = masks.shape[0]
        aerials = []
        for i in range(B):
            aerials.append(self._aerial_abbe(masks[i]))
        return self.sigmoid(torch.stack(aerials, dim=0))


class ScalarSOCSImaging(nn.Module):
    """
    标量 SOCS 加速前向成像：mask -> aerial intensity -> wafer image。

    与 ScalarSOCS 配对使用，前向公式：
        I(x) = Σ_k |IFFT{ M(f) · φ_k }|²

    其中 φ_k [1, N, N] = S_k * U_k（左奇异向量 × 奇异值）。
    与 VectorSOCSImaging 的接口对齐，仅核 channel 维度从 3/6 退化为 1。

    Parameters
    ----------
    socs       : 预计算好的 ScalarSOCS 实例
    grid       : Grid 实例（提供 N / x / y）
    sigmoid_tr : sigmoid 阈值 (默认 0.15)
    sigmoid_a  : sigmoid 斜率 (默认 85)
    """

    def __init__(
        self,
        socs: ScalarSOCS,
        grid: Grid,
        sigmoid_tr: float = 0.15,
        sigmoid_a: float = 85.0,
    ):
        super().__init__()
        self.socs = socs
        self.grid = grid
        self.real_dtype = socs.real_dtype
        self.complex_dtype = socs.complex_dtype
        self.device = socs.device
        self.sigmoid_tr = sigmoid_tr
        self.sigmoid_a = sigmoid_a

    # ---------- 核心前向 ----------

    def _aerial_socs(self, mask: torch.Tensor) -> torch.Tensor:
        """
        单 mask 的标量 SOCS 前向，返回 [N, N] raw aerial intensity。
        """
        device = self.device
        cdtype = self.complex_dtype
        rdtype = self.real_dtype

        if torch.is_complex(mask):
            mask_c = mask.to(device=device, dtype=cdtype)
        else:
            mask_c = mask.to(device=device, dtype=rdtype).to(cdtype)
        M = fft2c(mask_c)                               # [N, N]

        kernels = self.socs.get_kernels()               # [K, 1, N, N]
        MK = M[None, None, :, :] * kernels              # [K, 1, N, N]
        E = ifft2c(MK)                                  # [K, 1, N, N]
        aerial = (E.real * E.real + E.imag * E.imag).sum(dim=(0, 1))  # [N, N]
        return aerial

    def sigmoid(self, aerial: torch.Tensor) -> torch.Tensor:
        """sigmoid resist 模型。"""
        return 1.0 / (1.0 + torch.exp(-self.sigmoid_a * (aerial - self.sigmoid_tr)))

    # ---------- 单 mask 接口 ----------

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """单 mask 标量成像，返回 [N, N] sigmoid 后的光刻图像。"""
        aerial = self._aerial_socs(mask)
        return self.sigmoid(aerial)

    # ---------- 批量 mask 接口 ----------

    def forward_batch(self, masks: torch.Tensor) -> torch.Tensor:
        """
        批量标量 SOCS 成像。

        Parameters
        ----------
        masks : [B, N, N] real or complex tensor

        Returns
        -------
        wafer : [B, N, N] real tensor (经 sigmoid 后)

        利用 M 的 batch 维一次性完成所有 mask 的频谱计算，与 K 个核函数广播
        相乘做到 B×K 完全向量化。
        """
        device = self.device
        cdtype = self.complex_dtype
        rdtype = self.real_dtype

        if torch.is_complex(masks):
            masks_c = masks.to(device=device, dtype=cdtype)
        else:
            masks_c = masks.to(device=device, dtype=rdtype).to(cdtype)

        # [B, N, N] 频谱
        M_batch = fft2c(masks_c)                        # [B, N, N]

        kernels = self.socs.get_kernels()               # [K, 1, N, N]

        # 广播：[B, 1, 1, N, N] × [1, K, 1, N, N] → [B, K, 1, N, N]
        MK = M_batch[:, None, None, :, :] * kernels[None, :, :, :, :]
        E = ifft2c(MK)                                  # [B, K, 1, N, N]

        # sum over K and the singleton channel → [B, N, N]
        I = (E.real * E.real + E.imag * E.imag).sum(dim=(1, 2))

        return self.sigmoid(I)

    def forward_batch_aerial(self, masks: torch.Tensor) -> torch.Tensor:
        """
        与 forward_batch 相同，但返回 raw aerial intensity (不过 sigmoid)。
        用于 MEEF 优化器中需要直接计算 EPE 的场景。
        """
        device = self.device
        cdtype = self.complex_dtype
        rdtype = self.real_dtype

        if torch.is_complex(masks):
            masks_c = masks.to(device=device, dtype=cdtype)
        else:
            masks_c = masks.to(device=device, dtype=rdtype).to(cdtype)

        M_batch = fft2c(masks_c)                        # [B, N, N]
        kernels = self.socs.get_kernels()               # [K, 1, N, N]
        MK = M_batch[:, None, None, :, :] * kernels[None, :, :, :, :]
        E = ifft2c(MK)
        return (E.real * E.real + E.imag * E.imag).sum(dim=(1, 2))

