"""
矢量 SOCS（Sum of Coherent Systems）加速器。

原理
----
将 Abbe 求和的 TCC（Transmission Cross Coefficient）做截断 SVD，
把 N_source 次 FFT 对压缩为 K 次（K ≪ N_source）。

矢量 TCC 定义：
    TCC_total(f, f') = sum_p sum_s J_s * T_p*(f) * T_p(f')
    T_p(f) = M0[p, :](f) · jones · H(f)      # [N, N]

TCC_total 是 (N², N²) 的 Hermitian 半正定矩阵，取前 K 个特征对即可。
在实际实现中 TCC 以频差形式 C(Δf) = IFFT(diag(TCC)) 表示，
可利用 FFT 高效计算（Hopkins 公式的离散版本），避免显式构造 N⁴ 矩阵。

本实现采用"显式但紧凑"方式：
    1. 把 T_p(f) 拉成向量 t = [T_x_flat, T_y_flat, T_z_flat]，长度 3N²
    2. 对所有 source 点加权外积：TCC = sum_s J_s * t_s ⊗ t_s*
    3. 截断 SVD，取前 K 个特征向量 phi_k，reshape 回 [3, N, N]
    4. 保存 eigenvalues + eigenvectors，供前向使用

内存：
    TCC 矩阵大小 = (3N²)² float32 ≈ 3×(257²)² × 4 B ≈ 20 GB ← 无法直接存储！

因此本模块改用"隐式 TCC + 随机 SVD"：
    - 不显式构造 TCC
    - 把"T_p × source_weight"堆成 [3×N×N, N_source] 的矩阵 A，
      使得 TCC = A @ A†
    - 对 A 做截断 SVD（thin SVD），左奇异向量即 SOCS 核函数 φ_k，
      奇异值平方即权重 λ_k
    - A 的大小 = 3×N²×N_source，N=257, N_s=100 → 约 20M float32 ≈ 80 MB，可以接受

参考：
    - Cobb et al., "Fast Fourier transform based simulation of partially coherent imaging..." (1995)
    - Erdmann et al., "Advanced Mask Simulation" (2019), Ch. 6
"""
from __future__ import annotations

from tkinter import Grid
from tracemalloc import start
from turtle import end_fill
from typing import Tuple
import time
from mpmath import ker
from sympy import im
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from .pupil import Pupil
from .source import Illumination
from .vector_transfer import VectorTransfer
from .grid import Grid


class VectorSOCS(nn.Module):
    """
    预计算矢量 SOCS 核函数。

    参数
    ----
    pupil           : Pupil 实例（提供复光瞳 H(f)）
    illumination    : Illumination 实例（提供 source 坐标 + 权重）
    vector_transfer : VectorTransfer 实例（提供 T_p(f)）
    K               : 保留的 SOCS 项数（默认 20）
    weight_threshold: source 权重筛选阈值（默认 0，全部保留）

    属性
    ----
    kernels  : [K, C, N, N] 复张量，SOCS 核函数 φ_k = s_k · u_k
               C=3（单偏振）或 C=6（非偏振，前3=x后3=y）。
               已包含奇异值 s_k 的缩放，前向公式为：
                   I = Σ_k ||φ_k · M||² = Σ_k s_k² ||u_k · M||²
               不需要在前向时再额外乘以特征值 λ_k = s_k²。
    singular_values : [K] 实张量，奇异值 s_k（非平方），
                      特征值 λ_k = s_k²，能量占比 = Σs_k² / Σ_all s²。
    """

    def __init__(
        self,
        pupil: Pupil,
        illumination: Illumination,
        vector_transfer: VectorTransfer,
        K: int = 10,
        weight_threshold: float = 0.0,
        unpolarized: bool = False,
        jones: Tuple[float, float] = (1.0, 0.0),
    ):
        """
        参数
        ----
        unpolarized : True（默认）→ 把 x 和 y 偏振的 A 矩阵竖向堆叠后一起 SVD，
                      等价于对 TCC_x + TCC_y 做特征分解，与 Abbe 非偏振成像匹配。
                      False → 只分解 jones 指定的单偏振 TCC。
        """
        super().__init__()
        self.K = K
        self.real_dtype = pupil.real_dtype
        self.complex_dtype = pupil.complex_dtype
        self.device = pupil.device

        if unpolarized:
            kernels, sv = self._build_socs_unpolarized(
                pupil, illumination, vector_transfer, K, weight_threshold
            )
        else:
            kernels, sv = self._build_socs(
                pupil, illumination, vector_transfer, K, weight_threshold, jones
            )
        # 存为 buffer（不参与梯度，但随模型 .to(device) 一起迁移）
        self.register_buffer("kernels", kernels)        # [K, N, N] complex
        self.register_buffer("singular_values", sv)     # [K] real

    # ---------- 构建 ----------

    @staticmethod
    def _build_socs(
        pupil: Pupil,
        illumination: Illumination,
        vector_transfer: VectorTransfer,
        K: int,
        weight_threshold: float,
        jones: Tuple[float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构造 A = [sqrt(w_s) * t_s]_{s=1..Ns}，截断 SVD 得到 SOCS 核。

        T_s(f) 的精确构造方式（等价于 Abbe 中的 ramp × mask → FFT）：
            T_s(f) = FFT{ t(x) * exp(j 2π (fs·x + gs·y)) }
                   = FFT{ IFFT{T0(f)} * ramp_s(x) }
        其中 t(x) = IFFT{T0(f)} 是中心传递函数的空域冲激响应，
        ramp_s(x) = exp(j 2π (fs·x + gs·y)) 是 source 点的倾斜平面波。
        这样避免了整像素近似，得到精确的亚像素平移传递函数。
        """
        import math as _math
        from tool.fft_tool import fft2c, ifft2c

        device = pupil.device
        cdtype = pupil.complex_dtype
        rdtype = pupil.real_dtype
        N = pupil.N

        # 有效 source 点
        sigma_fx, sigma_fy = illumination.get_axes()
        weight_map = illumination.get_weights()
        sy_grid, sx_grid = torch.meshgrid(sigma_fy, sigma_fx, indexing="ij")
        sx_flat = sx_grid.reshape(-1)
        sy_flat = sy_grid.reshape(-1)
        w_flat = weight_map.reshape(-1)
        keep = w_flat > weight_threshold
        sx_flat, sy_flat, w_flat = sx_flat[keep], sy_flat[keep], w_flat[keep]
        Ns = w_flat.numel()

        if Ns == 0:
            raise ValueError("没有有效 source 点，请检查 weight_threshold")

        K_use = min(K, Ns)

        # T0(f)：中心 Jones 传递函数，[3, N, N] 复数，中心化频域
        T0 = vector_transfer.transfer_for_jones(
            Ex_in=jones[0], Ey_in=jones[1], apply_pupil=True
        )

        # t0(x)：T0 的空域冲激响应，[3, N, N]，中心化空域
        t0 = ifft2c(T0)  # [3, N, N]

        # 空间坐标（中心化），单位 nm
        grid = vector_transfer.grid
        X = grid.x.to(rdtype)   # [N, N]
        Y = grid.y.to(rdtype)   # [N, N]

        # source 物理频率 cycles/nm
        f_max = pupil.NA / pupil.wave_length
        fs_phys = (sx_flat * f_max).to(rdtype)   # [Ns]
        gs_phys = (sy_flat * f_max).to(rdtype)   # [Ns]

        # 构造 A：[3N², Ns] complex
        two_pi = 2.0 * _math.pi
        flat_size = 3 * N * N
        A = torch.zeros((flat_size, Ns), dtype=cdtype, device=device)
        start_time = time.time()
        for s in range(Ns):
            # ramp_s(x, y) = exp(j 2π (fs·x + gs·y))
            phase = two_pi * (fs_phys[s] * X + gs_phys[s] * Y)
            ramp = torch.complex(torch.cos(phase), torch.sin(phase)).to(cdtype)

            # T_s(f) = FFT{ t0(x) * ramp_s(x) }  [3, N, N]
            t_s = fft2c(t0 * ramp.unsqueeze(0))  # [3, N, N]

            # 按 w_s 加权后展平成列
            A[:, s] = t_s.reshape(-1) * w_flat[s].to(cdtype).sqrt()

        # 截断 thin SVD：A = U S V†
        print(
            f"    [SOCS] 对 A ({flat_size} × {Ns}) 做截断 SVD，保留 K={K_use} 项 ..."
        )
        U, S, _Vh = torch.linalg.svd(A, full_matrices=False)
        U_k = U[:, :K_use]   # [3N², K_use]
        S_k = S[:K_use]       # [K_use]

        # 核函数 φ_k = S_k * U_k，reshape 为 [K, 3, N, N]
        kernels_3p = (U_k * S_k.unsqueeze(0)).T.reshape(K_use, 3, N, N)
        end_time = time.time()
        print(f"    [SOCS] 构建 SOCS 核函数耗时 {end_time - start_time:.2f} 秒")
        return kernels_3p.contiguous(), S_k.to(rdtype)

    @classmethod
    def _build_socs_unpolarized(
        cls,
        pupil: Pupil,
        illumination: Illumination,
        vector_transfer: VectorTransfer,
        K: int,
        weight_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        非偏振 SOCS：把 x 偏振和 y 偏振的 A 矩阵竖向堆叠，
        A_total = [A_x ; A_y]（shape [6N², Ns]），再一起做 SVD。
        这等价于对 TCC_x + TCC_y 同时分解，与 0.5*(Ix+Iy) 的 Abbe 参考一致。
        """
        def _build_A(jones):
            return cls._build_socs(
                pupil, illumination, vector_transfer, K, weight_threshold, jones
            )

        print("    [SOCS] 构建 x 偏振 A ...")
        # 临时调用 _build_socs 只是为了拿到 A 矩阵，不做 SVD
        # 为此我们把 _build_socs 内部 A 的构造提取出来复用
        import math as _math
        from tool.fft_tool import fft2c, ifft2c

        device = pupil.device
        cdtype = pupil.complex_dtype
        rdtype = pupil.real_dtype
        N = pupil.N

        sigma_fx, sigma_fy = illumination.get_axes()
        weight_map = illumination.get_weights()
        sy_grid, sx_grid = torch.meshgrid(sigma_fy, sigma_fx, indexing="ij")
        w_flat_all = weight_map.reshape(-1)
        sx_all = sx_grid.reshape(-1)
        sy_all = sy_grid.reshape(-1)
        keep = w_flat_all > weight_threshold
        sx_flat = sx_all[keep]; sy_flat = sy_all[keep]; w_flat = w_flat_all[keep]
        Ns = w_flat.numel()
        K_use = min(K, Ns)

        f_max = pupil.NA / pupil.wave_length
        fs_phys = (sx_flat * f_max).to(rdtype)
        gs_phys = (sy_flat * f_max).to(rdtype)
        X = vector_transfer.grid.x.to(rdtype)
        Y = vector_transfer.grid.y.to(rdtype)
        two_pi = 2.0 * _math.pi

        A_list = []
        for jones_vec in [(1.0, 0.0), (0.0, 1.0)]:
            T0 = vector_transfer.transfer_for_jones(
                Ex_in=jones_vec[0], Ey_in=jones_vec[1], apply_pupil=True
            )
            t0 = ifft2c(T0)
            A = torch.zeros((3 * N * N, Ns), dtype=cdtype, device=device)
            for s in range(Ns):
                phase = two_pi * (fs_phys[s] * X + gs_phys[s] * Y)
                ramp = torch.complex(torch.cos(phase), torch.sin(phase)).to(cdtype)
                t_s = fft2c(t0 * ramp.unsqueeze(0))
                A[:, s] = t_s.reshape(-1) * w_flat[s].to(cdtype).sqrt()
            A_list.append(A)

        # A_total = [A_x; A_y]，直接拼接
        # 前向时 I = sum_k |E_k|^2 = Ix + Iy（双偏振总强度），需在前向除以 2
        A_total = torch.cat(A_list, dim=0)

        flat_total = A_total.shape[0]
        print(
            f"    [SOCS] 非偏振：对 A ({flat_total} × {Ns}) 做截断 SVD，保留 K={K_use} 项 ..."
        )
        U, S, _Vh = torch.linalg.svd(A_total, full_matrices=False)
        U_k = U[:, :K_use]   # [6N², K_use]
        S_k = S[:K_use]

        # 把 U_k 前半段（对应 x 偏振）和后半段（y 偏振）各 reshape 为 [K, 3, N, N]
        # 前向时两部分的强度贡献要分别计算再加
        half = 3 * N * N
        Ux_k = (U_k[:half, :] * S_k.unsqueeze(0)).T.reshape(K_use, 3, N, N)
        Uy_k = (U_k[half:, :] * S_k.unsqueeze(0)).T.reshape(K_use, 3, N, N)

        # 拼成 [K, 6, N, N]（前 3 通道=x 偏振，后 3 通道=y 偏振）
        kernels = torch.cat([Ux_k, Uy_k], dim=1).contiguous()   # [K, 6, N, N]
        return kernels, S_k.to(rdtype)


    # ---------- 对外接口 ----------

    def get_kernels(self) -> torch.Tensor:
        """返回 [K, 3, N, N] 复张量（含奇异值缩放）。"""
        return self.kernels

    def get_singular_values(self) -> torch.Tensor:
        """返回 [K] 奇异值（仅供诊断，kernels 已含缩放）。"""
        return self.singular_values

    def energy_fraction(self) -> float:
        """返回保留 K 项的能量占比（奇异值平方之和 / 总能量）。"""
        sv = self.singular_values
        return (sv[:self.K] ** 2).sum().item() / ((sv ** 2).sum().item() + 1e-12)


class ScalarSOCS(nn.Module):
    """
    标量 SOCS：忽略偏振，用复光瞳 H(f) 替代矢量传递矩阵 T0。

    构造方式与 VectorSOCS 平行，仅把 T0 从 [3, N, N] 退化为 [N, N]:
        H(f) = pupil.pupil                         # [N, N] complex
        h(x) = ifft2c(H)                            # 中心 PSF (空域冲激响应)
        T_s(f) = fft2c(h(x) * ramp_s(x))            # source 点 s 的频域传递函数
        ramp_s(x) = exp(j 2π (fs·x + gs·y))         # source 点 s 倾斜平面波
        A[:, s] = sqrt(w_s) * T_s.flatten()         # [N², Ns] complex

    A = U S V†，截断前 K 项:
        φ_k = S_k * U_k, reshape 为 [K, 1, N, N]   (单分量)

    前向公式 (在 ScalarSOCSImaging 内):
        I(x) = Σ_k |IFFT{ M(f) · φ_k }|²

    Parameters
    ----------
    pupil : Pupil
    illumination : Illumination
    K : int
        保留的 SOCS 项数
    weight_threshold : float
        小于该阈值的 source 点忽略
    """

    def __init__(
        self,
        pupil: Pupil,
        illumination: Illumination,
        K: int,
        weight_threshold: float,
    ) -> None:
        super().__init__()
        self.pupil = pupil
        self.illumination = illumination
        self.K = K
        self.weight_threshold = weight_threshold

        # 数据类型/设备 (供 ScalarSOCSImaging 复用)
        self.real_dtype = pupil.real_dtype
        self.complex_dtype = pupil.complex_dtype
        self.device = pupil.device

        kernels, singular_values = self._build_socs_kernel(
            pupil, illumination, pupil.grid, K, weight_threshold,
        )
        # 存为 buffer (不参与梯度，但随模型 .to(device) 一起迁移)
        self.register_buffer("kernels", kernels)            # [K, 1, N, N] complex
        self.register_buffer("singular_values", singular_values)  # [K] real

    # ---------- 构建 ----------

    @staticmethod
    def _build_socs_kernel(
        pupil: Pupil,
        illumination: Illumination,
        grid: Grid,
        K: int,
        weight_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建 SOCS 核：返回 (kernels [K, 1, N, N] complex, S [K] real)。
        """
        import math as _math
        from tool.fft_tool import fft2c, ifft2c

        device = pupil.device
        cdtype = pupil.complex_dtype
        rdtype = pupil.real_dtype
        N = pupil.N

        # 1) 有效 source 点 (与 VectorSOCS 一致)
        sigma_fx, sigma_fy = illumination.get_axes()
        weight_map = illumination.get_weights()
        sy_grid, sx_grid = torch.meshgrid(sigma_fy, sigma_fx, indexing="ij")
        sx_flat = sx_grid.reshape(-1)
        sy_flat = sy_grid.reshape(-1)
        w_flat = weight_map.reshape(-1)
        keep = w_flat > weight_threshold
        sx_flat = sx_flat[keep]; sy_flat = sy_flat[keep]; w_flat = w_flat[keep]
        Ns = w_flat.numel()
        if Ns == 0:
            raise ValueError("没有有效 source 点，请检查 weight_threshold")
        K_use = min(K, Ns)

        # 2) 复光瞳 H(f) 与中心 PSF h(x) (关键: 必须含 pupil.pupil 的限频效应)
        H = pupil.pupil.to(cdtype)                                # [N, N] complex
        # print(H.cpu().numpy())
        h0 = ifft2c(H)                                             # [N, N] complex
        # h0 = H
        # 3) 空间坐标 (中心化), 单位 nm
        X = grid.x.to(rdtype)
        Y = grid.y.to(rdtype)

        # 4) source 物理频率 cycles/nm
        f_max = pupil.NA / pupil.wave_length
        fs_phys = (sx_flat * f_max).to(rdtype)                    # [Ns]
        gs_phys = (sy_flat * f_max).to(rdtype)                    # [Ns]

        # 5) 构造 A: [N², Ns] complex
        two_pi = 2.0 * _math.pi
        flat_size = N * N
        A = torch.zeros((flat_size, Ns), dtype=cdtype, device=device)
        t0 = time.time()
        print(
            f"    [ScalarSOCS] 构建 A ({flat_size} × {Ns}), K={K}, K_use={K_use} ..."
        )
        for s in range(Ns):
            phase = two_pi * (fs_phys[s] * X + gs_phys[s] * Y)
            ramp = torch.complex(torch.cos(phase), torch.sin(phase)).to(cdtype)
            # T_s(f) = FFT{ h(x) * ramp_s(x) }
            t_s = fft2c(h0 * ramp)  
            # t_s = h0 * ramp                           # [N, N]
            A[:, s] = t_s.reshape(-1) * w_flat[s].to(cdtype).sqrt()

        # 6) 截断 thin SVD: A = U S V†
        print(f"    [ScalarSOCS] 对 A 做截断 SVD，保留 K={K_use} 项 ...")
        U, S, _Vh = torch.linalg.svd(A, full_matrices=False)
        U_k = U[:, :K_use]                                         # [N², K_use]
        S_k = S[:K_use]     
        
        # 7) 核函数 φ_k = S_k * U_k，reshape 为 [K, 1, N, N]
        kernels = (U_k * S_k.unsqueeze(0)).T.reshape(K_use, 1, N, N)
        print(f"    [ScalarSOCS] 构建完成，耗时 {time.time() - t0:.2f} s")
        return kernels.contiguous(), S_k.to(rdtype)

    # ---------- 对外接口 ----------

    def get_kernels(self) -> torch.Tensor:
        """返回 [K, 1, N, N] 复张量 (含奇异值缩放)。"""
        return self.kernels

    def get_singular_values(self) -> torch.Tensor:
        """返回 [K] 奇异值 (诊断用)。"""
        return self.singular_values

    def energy_fraction(self) -> float:
        """返回保留 K 项的能量占比 (奇异值平方比)。"""
        sv = self.singular_values
        return (sv[:self.K] ** 2).sum().item() / ((sv ** 2).sum().item() + 1e-12)

