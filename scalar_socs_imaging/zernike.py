"""
Zernike 多项式生成器 (NOLL 索引约定)

在光瞳圆域内生成归一化 Zernike 多项式 Z_noll(rho, theta)。
rho ∈ [0, 1] 为归一化频率半径，theta 为方位角。
"""
import numpy as np


# NOLL 索引到 (n, m) 的映射表 (支持到 Z37)
_NOLL_TO_NM = {
    1: (0, 0),   2: (1, 1),   3: (1, -1),  4: (2, 0),
    5: (2, -2),  6: (2, 2),   7: (3, -1),  8: (3, 1),
    9: (3, -3), 10: (3, 3),  11: (4, 0),  12: (4, 2),
    13: (4, -2), 14: (4, 4),  15: (4, -4), 16: (5, 1),
    17: (5, -1), 18: (5, 3),  19: (5, -3), 20: (5, 5),
    21: (5, -5), 22: (6, 0),  23: (6, -2), 24: (6, 2),
    25: (6, -4), 26: (6, 4),  27: (6, -6), 28: (6, 6),
    29: (7, 1),  30: (7, -1), 31: (7, 3),  32: (7, -3),
    33: (7, 5),  34: (7, -5), 35: (7, 7),  36: (7, -7),
    37: (8, 0),
}


def zernike_polynomial(
    N: int,
    noll_index: int,
    rho: np.ndarray,
    theta: np.ndarray,
    f_cutoff: float,
    freq_radius: np.ndarray,
) -> np.ndarray:
    """
    计算 NOLL 索引对应的 Zernike 多项式。

    多项式在光瞳圆域内 (rho <= 1) 有效，域外强制为 0。

    Parameters
    ----------
    N           : 网格大小
    noll_index  : NOLL 索引 (1-based)
    rho         : [N, N] 归一化频率半径
    theta       : [N, N] 方位角 [rad]
    f_cutoff    : 截止频率
    freq_radius : [N, N] 频率半径 (未归一化)

    Returns
    -------
    Z : [N, N] 多项式值
    """
    if noll_index not in _NOLL_TO_NM:
        return np.zeros((N, N), dtype=np.float64)

    n, m = _NOLL_TO_NM[noll_index]

    # 径向多项式 R_n^|m|(rho)
    R = _radial_poly(n, abs(m), rho)

    # 角向部分
    if m >= 0:
        angular = np.cos(m * theta)
    else:
        angular = np.sin(abs(m) * theta)

    Z = R * angular

    # 域外设零
    Z[freq_radius > f_cutoff] = 0.0

    return Z


def _radial_poly(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Zernike 径向多项式 R_n^m(rho)。"""
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)

    result = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coeff = (
            ((-1) ** k)
            * np.math.factorial(n - k)
            / (
                np.math.factorial(k)
                * np.math.factorial((n + m) // 2 - k)
                * np.math.factorial((n - m) // 2 - k)
            )
        )
        result += coeff * rho ** (n - 2 * k)

    return result
