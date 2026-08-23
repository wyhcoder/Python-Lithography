"""SRAF 渲染器.

把每个 SRAF 连通块的闭合 CP 用 B-spline (周期) 拟合, 然后送入 MSAA 渲染器,
得到一张和 mask 同尺寸的灰度图. 由于优化变量只是主图形 CP, SRAF CP 全程
冻结, 因此这张 sraf_mask 只需在初始化时渲染一次, 之后作为常量参与所有
forward (MEEF_Optimizer 已经按此结构使用 sraf_mask).
"""

from __future__ import annotations

from typing import List, Sequence
import numpy as np
from scipy.interpolate import splprep, splev

from utils_model.demo_MSAA import AntiAliasRenderer


def fit_closed_bspline(
    cp_block: np.ndarray,
    smoothing: float = 0.3,
    num_points: int = 200,
) -> np.ndarray:
    """对一组闭合 CP 做周期 B-spline 拟合, 返回 (num_points, 2) 的 [y, x] 曲线.

    与 utils_model.demo_parametric.ParametricDemo.b_spile 单段调用等价, 但不
    要求 splprep 的 per=True 在所有 scipy 版本上行为一致 -- 这里显式做闭合.
    """
    pts = np.asarray(cp_block, dtype=np.float64)
    if len(pts) < 4:
        # B-spline 拟合至少要 4 个点 (degree=3); 不够就直接返回原 CP 当多边形
        return pts

    y, x = pts[:, 0], pts[:, 1]
    # 周期 B-spline: per=True 要求首末点重合, 这里 pts 不含重复首点, 显式 wrap
    tck, _ = splprep([x, y], s=smoothing, per=True)
    u_fine = np.linspace(0.0, 1.0, num_points)
    x_fit, y_fit = splev(u_fine, tck)
    return np.stack([y_fit, x_fit], axis=1)


def render_sraf_mask(
    sraf_cps: Sequence[np.ndarray],
    mask_shape: tuple,
    smoothing: float = 0.3,
    num_points: int = 200,
    msaa_level: int = 16,
) -> np.ndarray:
    """SRAF CP 列表 -> 闭合 B-spline -> MSAA 渲染 -> 灰度图.

    Args:
        sraf_cps:   list[ ndarray(K_i, 2) ] 每个连通块的闭合 CP.
        mask_shape: (H, W) 输出灰度图尺寸, 与原始 mask 一致.
        smoothing:  B-spline 平滑系数, 0.3 与项目主图形拟合保持一致.
        num_points: 每条曲线离散到多少采样点送 MSAA 多边形填充.
        msaa_level: AntiAliasRenderer 的 msaa_level (16x = 4x4 网格, 默认).
    Returns:
        sraf_mask: (H, W) float32, 范围约 [0, 1] 的灰度覆盖率.
    """
    H, W = mask_shape
    if not sraf_cps:
        return np.zeros((H, W), dtype=np.float32)

    # 每个块拟合一条闭合曲线, 用列表传入 MSAA (它内部按多边形 OR 叠加)
    polygons: List[np.ndarray] = []
    for cp_block in sraf_cps:
        if cp_block is None or len(cp_block) < 3:
            continue
        curve = fit_closed_bspline(cp_block, smoothing=smoothing, num_points=num_points)
        polygons.append(curve)

    if not polygons:
        return np.zeros((H, W), dtype=np.float32)

    # 用一个空模板传形状即可; AntiAliasRenderer 只读 .shape
    template = np.zeros((H, W), dtype=np.float32)
    renderer = AntiAliasRenderer(msaa_level=msaa_level)
    return renderer.MSAA(polygons, template, type="gray")
