"""
SDF + Sigmoid 渲染器（方案 A，纯 NumPy，零外部依赖）

设计目标
--------
1) 提供一个与 ``utils_model.demo_MSAA.AntiAliasRenderer`` 接口完全兼容的渲染器：
        renderer.MSAA(all_polygons, mask, type='gray') -> (H, W) float32

   这样在 MEEF / SRAF / OPC 流程里，只需把
        self.render = AntiAliasRenderer(msaa_level=16)
   换成
        self.render = SDFRenderer(beta=1.0)
   即可零侵入地切换到 SDF 渲染。

2) 与 MSAA 本质区别：
   - MSAA 的 coverage 是亚像素采样比例，**关于多边形位置是阶梯函数**，量化精度 1/N。
   - SDF 是把"像素 p 到多边形边界的有符号距离 d(p)"通过 sigmoid 软化为 (0,1)：
            coverage(p) = σ(-d(p) / β)
     coverage 关于控制点位置是**处处光滑**的，没有 1/N 量化噪声。
     这对基于亚像素中心差分的 MEEF 矩阵来说，能让梯度更密集、更稳定。

3) 坐标约定与 MSAA 完全一致：多边形点是 (y, x) 顺序，即 polygon[:, 0]=y, polygon[:, 1]=x。

性能说明
--------
对每个像素 p、每条边 e，计算"点到线段距离"是 O(H*W*E)。
使用 NumPy 广播一次性向量化，单次渲染（512x512, 多边形总边数 ~200）大致 30-150 ms。
比 16x MSAA 慢，但梯度精度提升能换来 MEEF 整体迭代次数减少，整体不亏。

对外暴露
--------
- ``SDFRenderer``: 主类，``MSAA(...)`` 方法签名兼容现有调用方。
- ``polygon_signed_distance(...)``: 工具函数，可单独使用。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _segment_distance_field(pixels_yx: np.ndarray,
                            polygon_yx: np.ndarray) -> np.ndarray:
    """
    计算每个像素到多边形所有边的最小距离（无符号）。

    Parameters
    ----------
    pixels_yx : (H, W, 2) float
        像素中心坐标，pixels_yx[i, j] = [i + 0.5, j + 0.5]（也可不加 0.5，与 MSAA 一致即可）。
        与 MSAA 保持一致：MSAA 中 grid_y/grid_x 为整数像素索引，所以这里我们也不加 0.5。
    polygon_yx : (N, 2) float
        闭合多边形顶点（顺序连接，最后一点与第一点连成一条边）。

    Returns
    -------
    dist : (H, W) float64
        每个像素到多边形（所有边）的最小欧氏距离。
    """
    # 端点 a, b: (N, 2)
    a = polygon_yx
    b = np.roll(polygon_yx, -1, axis=0)

    # 边向量 ab: (N, 2)
    ab = b - a
    ab_sq = (ab * ab).sum(axis=-1)                    # (N,) 每条边长度的平方
    ab_sq = np.maximum(ab_sq, 1e-12)                  # 防 0 (退化的"零长度边")

    # 每个像素 p 到每条边起点 a 的向量 ap: (H, W, N, 2)
    # pixels: (H, W, 1, 2);  a: (N, 2) → (1, 1, N, 2)
    ap = pixels_yx[:, :, None, :] - a[None, None, :, :]

    # 投影系数 t = (ap · ab) / |ab|^2, 再 clip 到 [0,1]
    # (H, W, N)
    t = (ap * ab[None, None, :, :]).sum(axis=-1) / ab_sq[None, None, :]
    t = np.clip(t, 0.0, 1.0)

    # 最近点: a + t * ab, shape (H, W, N, 2)
    closest = a[None, None, :, :] + t[..., None] * ab[None, None, :, :]

    # 像素到最近点距离: (H, W, N)
    diff = pixels_yx[:, :, None, :] - closest
    dist_per_edge = np.sqrt((diff * diff).sum(axis=-1))

    # 取所有边的最小距离: (H, W)
    return dist_per_edge.min(axis=-1)


def _winding_inside(pixels_yx: np.ndarray,
                    polygon_yx: np.ndarray) -> np.ndarray:
    """
    用射线法（even-odd rule）判断每个像素是否落在多边形内部。

    与 utils_model/demo_MSAA.py 中 ``vectorized_is_inside`` 算法一致，但接收已经
    展开成 (H, W, 2) 的像素坐标，返回 (H, W) bool。

    备注：返回值是离散的 ±1（外部为 False/+1, 内部为 True/-1），
    在 SDF 上下文里只贡献"符号"，距离的连续性由 _segment_distance_field 保证。
    """
    H, W, _ = pixels_yx.shape
    a = polygon_yx                               # (N, 2)
    b = np.roll(polygon_yx, -1, axis=0)          # (N, 2)
    N = a.shape[0]
    if N < 3:
        return np.zeros((H, W), dtype=bool)

    # 像素 y, x: (H, W) → (H, W, 1) 与 (N,) 广播
    py = pixels_yx[..., 0:1]                     # (H, W, 1)
    px = pixels_yx[..., 1:2]                     # (H, W, 1)
    ay, ax = a[:, 0], a[:, 1]                    # (N,)
    by, bx = b[:, 0], b[:, 1]                    # (N,)

    # half-open: 边的 y 范围跨过像素 y 才算
    cond1 = (ay > py) != (by > py)               # (H, W, N)

    # 求边在像素 y 处的 x 交点
    denom = (by - ay) + 1e-12                    # (N,)
    t = (py - ay) / denom                        # (H, W, N)
    x_hit = ax + t * (bx - ax)                   # (H, W, N)

    cond2 = x_hit > px                           # (H, W, N) 交点在像素右边
    crossings = (cond1 & cond2).sum(axis=-1)     # (H, W) 每个像素的射线穿越次数

    inside = (crossings % 2) == 1
    return inside


def polygon_signed_distance(pixels_yx: np.ndarray,
                            polygon_yx: np.ndarray) -> np.ndarray:
    """
    计算多边形的有符号距离场。

    约定
    ----
    - 多边形外部: d > 0
    - 多边形内部: d < 0
    - 边界:        d ≈ 0

    Parameters
    ----------
    pixels_yx : (H, W, 2) float
        像素中心坐标 (y, x)。
    polygon_yx : (N, 2) float
        闭合多边形顶点。

    Returns
    -------
    sdf : (H, W) float64
    """
    unsigned = _segment_distance_field(pixels_yx, polygon_yx)
    inside = _winding_inside(pixels_yx, polygon_yx)
    sign = np.where(inside, -1.0, 1.0)
    return sign * unsigned


def _make_pixel_grid(H: int, W: int) -> np.ndarray:
    """
    构造 (H, W, 2) 的像素坐标网格，与 MSAA 一致使用 [y=row, x=col]，整数索引（不加 0.5）。
    这样像素 (i, j) 的坐标就是 (i, j)，方便和现有多边形顶点直接比较。
    """
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return np.stack([yy, xx], axis=-1).astype(np.float64)


class SDFRenderer:
    """
    SDF + Sigmoid 渲染器，接口与 ``AntiAliasRenderer`` 兼容。

    Parameters
    ----------
    beta : float, default 1.0
        sigmoid 软化宽度（像素）。
        - β → 0   : 退化为二值 mask（硬边）
        - β = 0.5 : 边界过渡 ~1 像素，最接近真实光刻边
        - β = 1.0 : 边界过渡 ~2 像素，梯度更平滑（推荐起步）
        - β = 2.0 : 较糊，适合早期粗优化
    clip_dist : float, default 8.0
        距离截断阈值。|d| 超过此值的像素直接判 0/1，避免计算无意义的 sigmoid。
        默认 8 像素，对 β ≤ 2 时 sigmoid 已经饱和到 1e-4 以下。

    Notes
    -----
    - 与 ``AntiAliasRenderer.MSAA`` 一致，多边形顶点格式为 (y, x)。
    - 多多边形合并策略与 MSAA 'gray' 模式保持一致：取 element-wise max。
      对互不重叠的多边形是正确的；对嵌套/相交多边形请走 'binary' 路径或自行 XOR。
    """

    def __init__(self, beta: float = 1.0, clip_dist: float = 8.0):
        if beta <= 0:
            raise ValueError("beta 必须 > 0；如需硬边请用 MSAA binary 模式")
        self.beta = float(beta)
        self.clip_dist = float(clip_dist)

        # 像素网格按 (H, W) 缓存，避免每次重建
        self._cached_grid_shape: Optional[Tuple[int, int]] = None
        self._cached_grid: Optional[np.ndarray] = None

    # ----------------------------- 内部工具 -----------------------------

    def _get_grid(self, H: int, W: int) -> np.ndarray:
        if self._cached_grid_shape != (H, W):
            self._cached_grid = _make_pixel_grid(H, W)
            self._cached_grid_shape = (H, W)
        return self._cached_grid  # type: ignore[return-value]

    def _coverage_from_sdf(self, sdf: np.ndarray) -> np.ndarray:
        """
        coverage = sigmoid(-d / β)
        - 内部 d < 0  → coverage → 1
        - 外部 d > 0  → coverage → 0
        在 |d| > clip_dist 处直接置 0/1，避免 exp 溢出 / 无意义计算。
        """
        beta = self.beta
        clip = self.clip_dist
        coverage = np.empty_like(sdf, dtype=np.float32)

        # 远离边界的部分硬切到 0/1（数值稳定 + 加速）
        far_outside = sdf > clip
        far_inside = sdf < -clip
        near = ~(far_outside | far_inside)

        coverage[far_outside] = 0.0
        coverage[far_inside] = 1.0

        # 近边界用 sigmoid 软化
        x = -sdf[near] / beta
        # 数值稳定的 sigmoid: 直接用 1/(1+exp(-x))，由 clip 控制 |x| ≤ clip/β
        coverage[near] = 1.0 / (1.0 + np.exp(-x))
        return coverage

    # ----------------------------- 对外接口 -----------------------------

    def MSAA(self,
             all_polygons,
             mask: np.ndarray,
             type: str = 'gray',
             batch_size=None,
             max_mem_fraction: float = 0.2) -> np.ndarray:
        """
        与 ``AntiAliasRenderer.MSAA`` 接口兼容的渲染入口。

        Parameters
        ----------
        all_polygons : list of (N_i, 2) array-like
            一组闭合多边形（每个 polygon 顶点为 [y, x]）。
        mask : np.ndarray
            只用 mask.shape 取出画布尺寸 (H, W)，与 MSAA 一致。
        type : {'gray', 'binary'}
            - 'gray'   : 返回 sigmoid 软覆盖率，元素 ∈ [0, 1]，float32
            - 'binary' : 返回硬阈值 (coverage >= 0.5)，float32 取值 0/1
        batch_size, max_mem_fraction : 兼容签名，**当前实现不使用**（保留以便上层无感切换）

        Returns
        -------
        coverage : (H, W) float32
        """
        H, W = int(mask.shape[0]), int(mask.shape[1])
        pixels = self._get_grid(H, W)

        # 多多边形 → 取每个多边形的 SDF，coverage 后做 element-wise max（与 MSAA gray 一致）
        final_coverage = np.zeros((H, W), dtype=np.float32)

        for poly in all_polygons:
            poly_arr = np.asarray(poly, dtype=np.float64)
            if poly_arr.ndim != 2 or poly_arr.shape[0] < 3 or poly_arr.shape[1] != 2:
                # 与 MSAA rasterize_polygon 行为一致：少于 3 个点直接跳过
                continue
            sdf = polygon_signed_distance(pixels, poly_arr)
            coverage = self._coverage_from_sdf(sdf)
            np.maximum(final_coverage, coverage, out=final_coverage)

        if type == 'gray':
            return final_coverage
        elif type == 'binary':
            return (final_coverage >= 0.5).astype(np.float32)
        else:
            raise ValueError("type must be 'gray' or 'binary'")
