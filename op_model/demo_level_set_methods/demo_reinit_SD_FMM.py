# 文件名: reinit_SD_FMM.py
import numpy as np

try:
    import skfmm  # type: ignore
    _HAS_SKFMM = True
except Exception:
    skfmm = None
    _HAS_SKFMM = False


def _reinit_with_edt(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """无 skfmm 时的降级实现：用 EDT 构造符号距离函数。"""
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as e:
        raise ImportError(
            "reinit_SD_FMM 需要 `scikit-fmm` 或 `scipy` 之一。当前环境未找到可用实现。"
        ) from e

    # 约定：phi < 0 为内部，phi >= 0 为外部
    inside = (phi < 0)

    # EDT 返回到“零像素”的距离，使用互补掩码得到内外距离
    dist_out = distance_transform_edt(~inside, sampling=(dy, dx)).astype(np.float64)
    dist_in = distance_transform_edt(inside, sampling=(dy, dx)).astype(np.float64)

    # 外部为正、内部为负
    return dist_out - dist_in


def reinit_SD_FMM(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    将水平集函数重初始化为符号距离函数(SDF)。

    优先使用 scikit-fmm；若其不可用，则自动降级到 scipy EDT 实现，
    从而避免在无 skfmm 环境下 import 失败。
    """
    if _HAS_SKFMM:
        # scikit-fmm 的 2D 网格间距顺序为 (dy, dx)
        return skfmm.distance(phi, dx=(dy, dx))

    return _reinit_with_edt(phi, dx, dy)
 