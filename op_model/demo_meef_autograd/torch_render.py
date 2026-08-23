"""
基于 PyTorch 的 SDF + sigmoid 多边形渲染（可微）。

与 ``utils_model/demo_sdf_renderer.py`` (NumPy 版) 的设计保持一致，
但全部用 ``torch.Tensor`` 实现，因此关于多边形顶点的梯度可以通过
autograd 自动反传。

关键约定
--------
- 多边形顶点格式 (N, 2) 是 [y, x] —— 与现有 MSAA / SDF 渲染器一致；
- 像素网格 (H, W, 2) 也是 [y, x]，整数索引 (不加 0.5)；
- coverage = sigmoid(-d / β)，d>0 表示外部，d<0 表示内部。

只对"内外判定的符号"做 detach (sign 是 ±1 的离散值，本身不可导，
但当扰动远小于 β 时 sign 实际不变化，detach 不影响小步长内的梯度精度)。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _make_pixel_grid(H: int, W: int,
                     dtype: torch.dtype = torch.float64,
                     device: Optional[torch.device] = None) -> torch.Tensor:
    """构造 (H, W, 2) 的像素坐标网格 (y, x)，整数索引。"""
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing='ij',
    )
    return torch.stack([yy, xx], dim=-1)


def _segment_distance_field(pixels_yx: torch.Tensor,
                            polygon_yx: torch.Tensor) -> torch.Tensor:
    """
    每个像素到多边形所有边的最小欧氏距离 (无符号)。

    Parameters
    ----------
    pixels_yx : (H, W, 2)  浮点
    polygon_yx : (N, 2)    浮点 (requires_grad=True 才能反传)

    Returns
    -------
    dist : (H, W)
    """
    a = polygon_yx                                       # (N, 2)
    b = torch.roll(polygon_yx, shifts=-1, dims=0)        # (N, 2)
    ab = b - a                                           # (N, 2)
    ab_sq = (ab * ab).sum(dim=-1).clamp_min(1e-12)       # (N,)

    # 像素到每条边起点的向量: (H, W, N, 2)
    ap = pixels_yx[:, :, None, :] - a[None, None, :, :]
    # 投影系数 t: (H, W, N), clip 到 [0, 1] 落到端点闭线段
    t = (ap * ab[None, None, :, :]).sum(dim=-1) / ab_sq[None, None, :]
    t = t.clamp(0.0, 1.0)
    # 最近点: (H, W, N, 2)
    closest = a[None, None, :, :] + t[..., None] * ab[None, None, :, :]
    # 距离: (H, W, N)
    diff = pixels_yx[:, :, None, :] - closest
    dist_per_edge = torch.sqrt((diff * diff).sum(dim=-1).clamp_min(1e-30))
    # Softmin 替代 hard min: 避免 "最近边切换" 导致的梯度丢失
    #   当两条边距离接近时，梯度平滑分配而非硬切到单条边
    #   tau 控制软化程度: 越小越接近 hard min，但梯度越容易丢失;
    #   tau=0.5 在亚像素精度和梯度平滑间取得平衡
    tau = 0.5
    dist_shifted = dist_per_edge - dist_per_edge.min(dim=-1, keepdim=True).values
    weights = torch.softmax(-dist_shifted / tau, dim=-1)
    return (dist_per_edge * weights).sum(dim=-1)


def _segment_distance_field_bbox(pixels_yx: torch.Tensor,
                                 polygon_yx: torch.Tensor,
                                 clip_dist: float) -> Tuple[torch.Tensor,
                                                            Tuple[int, int, int, int]]:
    """
    bbox 加速版: 仅在 polygon 包围盒 + clip_dist 边界内算精确距离场,
    其他像素直接置为 ``clip_dist + 1``  (远离边界, 后面会被 sigmoid 饱和).

    这是性能关键优化: 朴素全图 (H, W, N, 2) 的中间张量在 H=W=512, N=200
    时约 1.6 GB; bbox 后通常窗口只有 ~64x64, 总计算量降低 50-100 倍.

    Returns
    -------
    dist_full : (H, W) 完整距离场 (bbox 外为 clip_dist + 1)
    bbox      : (y0, y1, x0, x1) 真正算了精确距离的窗口
    """
    H, W = pixels_yx.shape[:2]
    device = pixels_yx.device
    dtype = pixels_yx.dtype

    # 用 detach 算 bbox: bbox 边界本身关于 cps 不可导, 但只要 cps 移动 < clip_dist,
    # 边界上像素就一直在窗口内, 不影响梯度精度.
    with torch.no_grad():
        ymin = polygon_yx[:, 0].min().item()
        ymax = polygon_yx[:, 0].max().item()
        xmin = polygon_yx[:, 1].min().item()
        xmax = polygon_yx[:, 1].max().item()

    pad = clip_dist + 2.0  # 多 2 像素缓冲
    y0 = max(0, int(ymin - pad))
    y1 = min(H, int(ymax + pad) + 1)
    x0 = max(0, int(xmin - pad))
    x1 = min(W, int(xmax + pad) + 1)

    if y1 <= y0 or x1 <= x0:
        # 退化情况, 直接走全图
        full = _segment_distance_field(pixels_yx, polygon_yx)
        return full, (0, H, 0, W)

    # 仅在窗口里算精确距离场
    pixels_window = pixels_yx[y0:y1, x0:x1]                # (h, w, 2)
    dist_window = _segment_distance_field(pixels_window, polygon_yx)  # (h, w)

    # 拼回全图: 窗口外填 clip_dist + 1, 让外面的 sigmoid 饱和到 0
    far = torch.full((H, W), float(clip_dist) + 1.0,
                     dtype=dtype, device=device)
    # 用 index 写入, 不破坏 dist_window 的梯度
    full = far.clone()
    full[y0:y1, x0:x1] = dist_window
    return full, (y0, y1, x0, x1)


def _winding_inside_detached(pixels_yx: torch.Tensor,
                             polygon_yx: torch.Tensor,
                             bbox: Optional[Tuple[int, int, int, int]] = None
                             ) -> torch.Tensor:
    """
    用射线法 (even-odd) 判像素是否在多边形内部，返回 (H, W) bool。

    这一步**不参与梯度**：sign 是 ±1 的离散量，其偏导几乎处处为 0。
    在 |扰动| << β 的范围内 sign 不变化，detach 不会损失梯度精度。

    bbox 加速: 如果传入 bbox=(y0,y1,x0,x1), 仅在该窗口内算 (H,W,N) 张量,
    bbox 外像素假定为外部 (False). 这与距离场的 bbox 裁剪策略一致,
    显著减少 (H, W, N) 中间张量的内存与算力开销.
    """
    with torch.no_grad():
        a = polygon_yx                                       # (N, 2)
        b = torch.roll(polygon_yx, shifts=-1, dims=0)
        N = a.shape[0]
        H, W, _ = pixels_yx.shape
        if N < 3:
            return torch.zeros((H, W), dtype=torch.bool, device=pixels_yx.device)

        if bbox is not None:
            y0, y1, x0, x1 = bbox
            window = pixels_yx[y0:y1, x0:x1]
        else:
            y0, y1, x0, x1 = 0, H, 0, W
            window = pixels_yx

        py = window[..., 0:1]                                 # (h, w, 1)
        px = window[..., 1:2]
        ay, ax = a[:, 0], a[:, 1]                             # (N,)
        by, bx = b[:, 0], b[:, 1]

        cond1 = (ay > py) != (by > py)                        # (h, w, N)
        denom = (by - ay) + 1e-12
        t = (py - ay) / denom
        x_hit = ax + t * (bx - ax)
        cond2 = x_hit > px
        crossings = (cond1 & cond2).to(torch.int64).sum(dim=-1)  # (h, w)
        inside_window = (crossings % 2) == 1

        if bbox is None:
            return inside_window
        # 拼回全图
        inside_full = torch.zeros((H, W), dtype=torch.bool,
                                  device=pixels_yx.device)
        inside_full[y0:y1, x0:x1] = inside_window
        return inside_full


def render_polygon_sdf(polygon_yx: torch.Tensor,
                       H: int, W: int,
                       beta: float = 1.0,
                       clip_dist: float = 8.0,
                       pixel_grid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    把单个多边形渲染成 (H, W) 的 soft mask，关于多边形顶点可微。

    Parameters
    ----------
    polygon_yx : (N, 2)
        多边形闭合顶点 (一般 requires_grad=True)。
    H, W : int
        画布尺寸。
    beta : float
        sigmoid 软化宽度 (像素)。
    clip_dist : float
        |d| 超过此值的像素 sigmoid 已饱和到 ~0/1，硬切以提升稳定性。
        注意：clip 后 *远离边界* 的像素梯度严格为 0，这对优化无害
        (那些像素本来就不该响应)。
    pixel_grid : (H, W, 2) optional
        预生成的像素网格，避免每次重建。

    Returns
    -------
    coverage : (H, W) float
    """
    if pixel_grid is None:
        pixel_grid = _make_pixel_grid(H, W,
                                      dtype=polygon_yx.dtype,
                                      device=polygon_yx.device)

    # bbox 加速: 仅在 polygon bbox + clip_dist 边界内算精确距离场
    unsigned, bbox = _segment_distance_field_bbox(pixel_grid, polygon_yx,
                                                  clip_dist=clip_dist)
    # winding 判内外也走同一个 bbox, 减少 (H, W, N) 中间张量
    inside = _winding_inside_detached(pixel_grid, polygon_yx, bbox=bbox)

    # signed distance: 内部 d<0, 外部 d>0
    sign = torch.where(
        inside,
        torch.full_like(unsigned, -1.0),
        torch.full_like(unsigned, +1.0),
    )
    sdf = sign * unsigned

    # 远离边界硬切, 加速 + 数值稳定
    near = sdf.abs() < clip_dist
    coverage = torch.zeros_like(sdf)
    # 内部远区: coverage=1
    coverage = torch.where(sdf < -clip_dist,
                           torch.ones_like(coverage), coverage)
    # 近边界: sigmoid 软化
    soft = torch.sigmoid(-sdf / beta)
    coverage = torch.where(near, soft, coverage)
    return coverage


def render_polygons_union_sdf(polygons_yx,
                              H: int, W: int,
                              beta: float = 1.0,
                              clip_dist: float = 8.0,
                              pixel_grid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    把一组多边形渲染并合并成一张 mask (取 element-wise max)。

    与 ``AntiAliasRenderer.MSAA(..., type='gray')`` 行为一致：
    互不重叠的多边形等价于"贴片合并"。

    Parameters
    ----------
    polygons_yx : list of (N_i, 2) tensor
        每个多边形单独渲染再 max；polygon 张量 requires_grad 各自决定。

    Returns
    -------
    coverage : (H, W) tensor
    """
    if pixel_grid is None:
        # 用第一个多边形的 dtype/device 创建网格
        ref = polygons_yx[0]
        pixel_grid = _make_pixel_grid(H, W, dtype=ref.dtype, device=ref.device)

    final = None
    for poly in polygons_yx:
        if poly.shape[0] < 3:
            continue
        cov = render_polygon_sdf(poly, H, W,
                                 beta=beta, clip_dist=clip_dist,
                                 pixel_grid=pixel_grid)
        final = cov if final is None else torch.maximum(final, cov)

    if final is None:
        # 没有合法多边形 → 返回零图
        return torch.zeros((H, W),
                           dtype=pixel_grid.dtype,
                           device=pixel_grid.device)
    return final


# =============================================================================
# B-spline (周期闭合) 的可微采样
# =============================================================================

def bspline_sample_periodic(cps: torch.Tensor,
                            num_points: int = 200,
                            degree: int = 3) -> torch.Tensor:
    """
    用周期均匀 B-spline 把控制点采样成 (num_points, 2) 的密集轮廓。

    与 ``utils_model.demo_parametric.ParametricDemo.b_spile`` 的语义近似：
    输入控制点 cps (N, 2) [y, x]，返回闭合的 B-spline 曲线点。

    实现为纯 tensor 运算，因此对 cps 完全可微。
    采用 *uniform* B-spline (周期 knot)，与 scipy.splprep 的 ``per=True``
    在视觉上几乎一致 (smoothing s 设为 0 时完全一致)。

    Notes
    -----
    - degree 默认 3 (cubic)，与 OPC 工程惯例一致。
    - 对于 OA (折线)，请直接传 cps 给 render，无需经过 B-spline。
    """
    N = cps.shape[0]
    if N < degree + 1:
        # 控制点不足以拟合，直接返回原 cps (退化为折线)
        return cps

    # 周期 B-spline: 在 cps 末尾循环 degree 个点
    pad = cps[:degree]                              # (degree, 2)
    cps_ext = torch.cat([cps, pad], dim=0)          # (N+degree, 2)

    # 参数 u ∈ [degree, N+degree]，均匀间隔
    # 注意 N 个段，每段 num_points/N 个采样点 (向上取整)
    n_per_seg = max(1, num_points // N)
    total = n_per_seg * N

    # 生成均匀参数 u: 长度 total
    u = torch.linspace(0.0, float(N), total + 1,
                       dtype=cps.dtype, device=cps.device)[:-1]

    # de Boor 算法 (uniform knot, degree=k)
    # uniform B-spline 的 basis 函数是 piecewise polynomial of u%1
    # 这里直接用 de Boor 递推，对 cubic 就是 4-point Catmull-like 卷积
    # 但 uniform B-spline 的解析形式更简洁:
    #
    #   B(u) = sum_{i} N_{i,k}(u) * P_i
    #
    # 对于 uniform cubic (k=3), 每个段只用 4 个连续控制点:
    #
    #   P(t) = 1/6 * [(1-t)^3 * P_{i-1}
    #               + (3t^3 - 6t^2 + 4) * P_i
    #               + (-3t^3 + 3t^2 + 3t + 1) * P_{i+1}
    #               + t^3 * P_{i+2}]   for t in [0, 1]
    #
    # 对应段 i = floor(u), t = u - i
    if degree != 3:
        raise NotImplementedError(
            "目前只实现了 cubic uniform B-spline (degree=3); "
            "如需其它阶可扩展。"
        )

    seg = torch.floor(u).to(torch.int64) % N         # (total,) 段索引
    t = (u - torch.floor(u))                         # (total,) ∈ [0, 1)

    # 取 4 个控制点 (周期闭合，自动从 cps_ext 借位):
    # i-1, i, i+1, i+2  →  在 cps_ext 中对应 idx, idx+1, idx+2, idx+3
    # 因为 cps_ext = cps ++ cps[:3]，且 i=seg, 所以:
    #   P_{-1} ↔ cps[(seg-1) mod N]
    #   P_{ 0} ↔ cps[seg]
    #   P_{+1} ↔ cps[(seg+1) mod N]  → cps_ext[seg+1]
    #   P_{+2} ↔ cps[(seg+2) mod N]  → cps_ext[seg+2]
    pm1 = cps[(seg - 1) % N]                         # (total, 2)
    p0 = cps[seg]
    p1 = cps_ext[(seg + 1)]                          # 对应 cps[(seg+1)%N]
    p2 = cps_ext[(seg + 2)]

    t2 = t * t
    t3 = t2 * t
    one_m_t = 1.0 - t
    b_m1 = (one_m_t ** 3) / 6.0
    b_0 = (3 * t3 - 6 * t2 + 4) / 6.0
    b_1 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
    b_2 = t3 / 6.0

    curve = (b_m1[:, None] * pm1
             + b_0[:, None] * p0
             + b_1[:, None] * p1
             + b_2[:, None] * p2)
    return curve  # (total, 2)
