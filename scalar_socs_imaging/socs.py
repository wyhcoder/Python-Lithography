"""
SOCS (Sum of Coherent Systems) 分解 — 将 Hopkins 部分相干成像压缩为 K 个相干核。

物理原理:
  Hopkins 成像的互强度函数 J 可由 SVD 分解:
    J(f1, f2) = Σ_k λ_k · φ_k(f1) · φ_k*(f2)

  SVD 作用于矩阵 A, 其中 A 的每列对应一个光源点:
    A_col = sqrt(w_s) · P(f - k_s)

  A = U Σ V^H  →  λ_k = σ_k²  →  φ_k(f) = U_·k(f) / sqrt(Σ w_s)

加速比:
  - Abbe:  每光源点一次 FFT 卷积 (~200-400 点)
  - SOCS:  前 K 个核一次 FFT 卷积 (K≈50)
  → 约 4-8x 加速，能量保留 > 99%
"""
import numpy as np
import time
from typing import Tuple, List
from .pupil_source import PupilFunction, SourceMap


class SOCSDecomposition:
    """
    SOCS 分解器: 将光源 + 光瞳系统压缩为 K 个相干核。

    支持两种 SVD 策略:
      - "randomized": 随机化 SVD，只计算前 K 个分量 (快，默认)
      - "full":       完整 np.linalg.svd (慢，用于精度对比)

    Attributes
    ----------
    kernels         : [K, N, N] 复数相干核 (空间域)
    eigenvalues     : [K] 特征值 λ_k (降序)
    total_energy    : Σλ (所有特征值之和)
    retained_energy : Σ_{k<K} λ_k (保留能量)
    energy_ratio    : 保留能量占比
    K               : 实际保留的核数
    """

    def __init__(
        self,
        pupil: PupilFunction,
        source: SourceMap,
        K: int = 50,
        method: str = "randomized",
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        pupil        : 光瞳函数
        source       : 光源分布
        K            : 最大保留核数
        method       : "randomized" | "full"
        random_state : 随机种子 (randomized 模式)
        """
        self.pupil = pupil
        self.source = source
        self.K = K
        self.method = method
        self.random_state = random_state

        self.kernels: List[np.ndarray] = []
        self.eigenvalues: np.ndarray = np.array([])

        self._decompose()

    # ================================================================
    #  核心分解
    # ================================================================

    def _decompose(self) -> None:
        pupil_data = self.pupil.data
        source_data = self.source.data

        sy, sx = np.nonzero(source_data > 1e-9)
        weights = source_data[sy, sx]

        if len(sx) == 0:
            raise RuntimeError("光源中没有有效发光点")

        N = pupil_data.shape[0]
        M = len(weights)

        pupil_k_step = self.pupil.frequency_coords[1] - self.pupil.frequency_coords[0]
        source_k_coords_x = self.source.coords[sx]
        source_k_coords_y = self.source.coords[sy]
        shifts_x = np.round(source_k_coords_x / pupil_k_step).astype(int)
        shifts_y = np.round(source_k_coords_y / pupil_k_step).astype(int)

        print(f"  [SOCS] Grid: {N}×{N}, Source points: {M}, Target K: {self.K}")
        print(f"  [SOCS] A matrix: {N*N:,} rows × {M} cols, "
              f"mem={N*N*M*16/(1024**2):.1f} MB")
        print(f"  [SOCS] Method: {self.method}")

        # 构建 A 矩阵
        t0 = time.time()
        A = self._build_A(pupil_data, weights, shifts_x, shifts_y, N, M)
        t_build = time.time() - t0

        # SVD
        t0 = time.time()
        if self.method == "randomized":
            U, S = self._randomized_svd(A, self.K)
        else:
            U, S, _ = np.linalg.svd(A, full_matrices=False)
        t_svd = time.time() - t0

        eigenvalues = S ** 2
        total_energy = np.sum(eigenvalues)

        # 用所有特征值估计总能量 (随机化 SVD 只算了前 K 个)
        # 假设 λ_{K+1}...λ_M 很小，用 A 的 Frobenius 范数补全
        if self.method == "randomized":
            # Frobenius 范数 = Σσ² = Σλ，包含所有奇异值
            total_fro = float(np.linalg.norm(A, "fro") ** 2)
            # 前 K 个的和 vs 全部
            total_energy = max(total_fro, eigenvalues.sum())

        actual_K = min(self.K, M)

        print(f"  [SOCS] Build A: {t_build:.1f}s, SVD: {t_svd:.1f}s, "
              f"Total: {t_build+t_svd:.1f}s")

        # 能量统计
        for k_check in [5, 10, 20, 50, min(100, M)]:
            if k_check <= actual_K:
                ratio = np.sum(eigenvalues[:k_check]) / total_energy * 100
                print(f"  [SOCS]   Top {k_check:3d} kernels: {ratio:.2f}% energy")

        # 生成空间域核
        self.eigenvalues = eigenvalues[:actual_K]
        self.total_energy = total_energy
        self.retained_energy = np.sum(self.eigenvalues)
        self.energy_ratio = self.retained_energy / total_energy

        self.kernels = []
        for k in range(actual_K):
            kernel_freq = U[:, k].reshape((N, N))
            kernel_spatial = np.fft.fftshift(
                np.fft.ifft2(np.fft.ifftshift(kernel_freq))
            )
            
            self.kernels.append(kernel_spatial.astype(np.complex128))

        print(f"  [SOCS] Done. {actual_K} kernels, "
              f"energy={self.energy_ratio*100:.2f}%")

    # ================================================================
    #  构建 A 矩阵
    # ================================================================

    @staticmethod
    def _build_A(
        pupil: np.ndarray,
        weights: np.ndarray,
        shifts_x: np.ndarray,
        shifts_y: np.ndarray,
        N: int,
        M: int,
    ) -> np.ndarray:
        """
        构建 A 矩阵，列 = sqrt(w_s) · P(f - k_s) 展平。
        使用向量化 np.roll 提升构建速度。
        """
        A = np.zeros((N * N, M), dtype=np.complex128)
        sqrt_weights = np.sqrt(weights)
        for i in range(M):
            shifted = np.roll(pupil, (shifts_y[i], shifts_x[i]), axis=(0, 1))
            A[:, i] = np.ravel(sqrt_weights[i] * shifted)
        return A

    # ================================================================
    #  随机化 SVD (Halko-Martinsson-Tropp 算法)
    # ================================================================

    @staticmethod
    def _randomized_svd(
        A: np.ndarray,
        K: int,
        n_oversamples: int = 10,
        n_power_iter: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机化 SVD: 只计算前 K 个奇异值/向量。

        算法 (Halko et al., 2011):
          1) 生成随机矩阵 Ω ∈ R^{M × (K+p)}
          2) Y = A Ω              (子空间采样)
          3) Q = QR(Y)            (正交基)
          4) B = Q^H A            (投影到小矩阵)
          5) SVD(B) = U_B Σ V^H
          6) U = Q U_B            (恢复左奇异向量)

        Parameters
        ----------
        A              : (N², M) 复数矩阵
        K              : 目标秩
        n_oversamples  : 过采样数 (推荐 5-10)
        n_power_iter   : 幂迭代次数 (2-3 次提高精度)

        Returns
        -------
        U : (N², K) 左奇异向量
        S : (K,)    奇异值
        """
        M = A.shape[1]
        p = n_oversamples
        target_rank = min(K + p, M)

        rng = np.random.default_rng(42)

        # 1) 随机采样矩阵
        Omega = rng.standard_normal((M, target_rank)).astype(A.real.dtype)

        # 2) 子空间采样: Y = A Ω
        Y = A @ Omega

        # 3) 幂迭代: 增强奇异值衰减 (A A^H)^q A Ω
        for _ in range(n_power_iter):
            Y = A @ (A.conj().T @ Y)

        # 4) QR 分解: Y = Q R
        Q, _ = np.linalg.qr(Y)

        # 5) 投影: B = Q^H A  (小矩阵)
        B = Q.conj().T @ A  # (target_rank, M)

        # 6) 对小矩阵做完整 SVD
        Ub, S, _ = np.linalg.svd(B, full_matrices=False)

        # 7) 恢复: U = Q Ub
        U = Q @ Ub  # (N², target_rank)

        # 只取前 K 个
        return U[:, :K], S[:K]

    # ================================================================
    #  公共接口
    # ================================================================

    def get_kernels(self) -> np.ndarray:
        """返回 [K, N, N] 复数核堆叠数组。"""
        return np.stack(self.kernels, axis=0)

    def get_eigenvalues(self) -> np.ndarray:
        """返回特征值 [K]。"""
        return self.eigenvalues.copy()

    def get_fft_kernels(self) -> np.ndarray:
        """返回 [K, N, N] 预计算好的 FFT2 核。"""
        kernels_stack = self.get_kernels()
        return np.fft.fft2(kernels_stack, axes=(-2, -1)).astype(np.complex128)

    def summary(self) -> dict:
        """返回分解摘要。"""
        return {
            "K": len(self.kernels),
            "grid_size": self.pupil.data.shape[0],
            "method": self.method,
            "total_energy": float(self.total_energy),
            "retained_energy": float(self.retained_energy),
            "energy_ratio": float(self.energy_ratio),
            "eigenvalues": self.eigenvalues.tolist(),
        }
