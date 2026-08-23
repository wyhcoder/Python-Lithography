"""控制点 (Control Points, CP) 提取.

主图形 CP:
    沿用 utils_model.select_eps.DemoSelectEps_new.extract_mask_control_points,
    支持 left-right / diagonal / center 对称.
SRAF CP:
    对 sraf 二值图做 cv2.findContours 取每个连通块的外轮廓, 然后沿弧长每隔
    `arclen_step` 个像素采样一个 CP. 不足 `min_cps` 个时按弧长等分采样到
    `min_cps` 个 (B-spline 闭合拟合下限).

对称约束 (与主图形保持一致):
    若指定 symmetry, 则一半 SRAF CP 由原 LSM 实测取得, 另一半通过几何对称
    变换镜像生成, 保证两组 CP 严格对称.

返回的数据结构:
    main_cps: list[ list[ [y, x] ] ]
        主图形 CP, 与 set_meef.MEEF.cps 同构. 通常长度为 1 (单连通主图形)
        或 2 (对称图形拆分成两组).
    sraf_cps: list[ list[ [y, x] ] ]
        每个 SRAF 连通块对应一组闭合 CP. 长度 = SRAF 连通块数.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np
import cv2
from skimage.measure import label, regionprops

from utils_model.select_eps import DemoSelectEps_new


# ----------------------------------------------------------------------
# 主图形 CP
# ----------------------------------------------------------------------
def extract_main_cps(
    target_mask: np.ndarray,
    interval_k: int,
    symmetry: Optional[str] = None,
    pattern_name: str = "",
    mid_weight: float = 4.0,
    other_weight: float = 1.0,
) -> Tuple[list, DemoSelectEps_new]:
    """提取主图形控制点 (沿用项目内现有逻辑).

    Args:
        target_mask:  二值主图形.
        interval_k:   extract_mask_control_points 的 k, 实际间隔 = k+1 像素.
        symmetry:     None / "left-right" / "diagonal" / "center".
        pattern_name: 仅用于内部 DemoSelectEps_new 构造, 不影响 CP 结果.
    Returns:
        (main_cps, eps_helper)
        main_cps: list[list[[y, x]]] 与 set_meef.MEEF.cps 同构.
        eps_helper: DemoSelectEps_new 实例, 后续 EP 选取也复用它.
    """
    eps_helper = DemoSelectEps_new(target_mask, mid_weight, other_weight, pattern_name)
    sym = None if (symmetry is None or symmetry.lower() == "none") else symmetry
    main_cps = eps_helper.extract_mask_control_points(interval_k, symmetry=sym)
    # 与 set_meef.MEEF 一致: 清洗过近的相邻控制点, 防优化震荡
    main_cps = eps_helper.remove_adjacent_close_points(main_cps, threshold=2)
    return main_cps, eps_helper


# ----------------------------------------------------------------------
# SRAF CP
# ----------------------------------------------------------------------
def _resample_contour_by_arclen(
    contour_yx: np.ndarray,
    arclen_step: float,
    min_cps: int,
) -> np.ndarray:
    """沿闭合轮廓按弧长等距重采样, 返回 (N, 2) 的 [y, x] 数组.

    - 闭合处理: 自动把首点接到末尾计算总弧长, 但输出不含重复首点.
    - 当总周长 / arclen_step < min_cps 时, 改用 min_cps 个点等分.
    """
    pts = np.asarray(contour_yx, dtype=np.float64)
    if len(pts) < 3:
        return pts.copy()

    # 累积弧长 (闭合)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return pts[:1].copy()

    # 决定采样个数
    n_by_step = int(np.floor(total / max(arclen_step, 1e-3)))
    n = max(min_cps, n_by_step)

    # 在 [0, total) 上等距采样 n 个 (闭合, 不含 total)
    targets = np.linspace(0.0, total, n, endpoint=False)
    # 在 cum 上插值 y, x
    y_interp = np.interp(targets, cum, closed[:, 0])
    x_interp = np.interp(targets, cum, closed[:, 1])
    return np.stack([y_interp, x_interp], axis=1)


def _apply_symmetry(pts: np.ndarray, shape: tuple, symmetry: str) -> np.ndarray:
    """对一组 [y, x] 点应用几何对称变换 (与 extract_mask_control_points 一致).

    - center:     绕图像中心 180 度旋转
    - left-right: 关于竖直中线左右镜像 (y 不变, x 镜像)
    - diagonal:   关于反对角线翻转 (y, x) -> (h-1-x, w-1-y)
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    if symmetry == "center":
        return np.stack([2 * cy - pts[:, 0], 2 * cx - pts[:, 1]], axis=1)
    if symmetry == "left-right":
        return np.stack([pts[:, 0], 2 * cx - pts[:, 1]], axis=1)
    if symmetry == "diagonal":
        return np.stack([h - 1 - pts[:, 1], w - 1 - pts[:, 0]], axis=1)
    raise ValueError(f"Unsupported symmetry: {symmetry}")


def _pair_blocks_by_symmetry(
    centroids: np.ndarray,
    shape: tuple,
    symmetry: str,
) -> Tuple[List[int], List[int], List[int]]:
    """根据图形对称, 把 SRAF 连通块分成 (refs, mirrors, self_sym).

    refs:     用于 "实测取 CP 的代表块" 的 label 索引列表
    mirrors:  与 refs 一一对应, 表示 "由对应 ref 经对称生成 CP 的块" label 索引
    self_sym: 自身就在对称轴/中心上的块 label 索引 (镜像 = 自身, 直接实测)

    简单贪心配对: 把每个块的对称像点最近邻匹配到另一个块, 距离 < 容差则配对.
    """
    n = len(centroids)
    if n == 0:
        return [], [], []
    centroids = np.asarray(centroids, dtype=np.float64)
    mirror_pts = _apply_symmetry(centroids, shape, symmetry)

    # 距离矩阵: dist[i, j] = ||mirror(centroids[i]) - centroids[j]||
    diff = mirror_pts[:, None, :] - centroids[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    used = [False] * n
    refs: List[int] = []
    mirrors: List[int] = []
    self_sym: List[int] = []
    # 容差: 取图像尺寸的 1.5%, 对 200x200 约 3 像素, 足以容下离散误差
    tol = max(2.0, 0.015 * max(shape))

    # 自身对称 (mirror 像就在自己中心附近)
    for i in range(n):
        if dist[i, i] < tol:
            self_sym.append(i)
            used[i] = True

    # 配对剩余块
    for i in range(n):
        if used[i]:
            continue
        # 在未使用的块里找与 mirror(i) 最近的 j
        best_j, best_d = -1, np.inf
        for j in range(n):
            if used[j] or j == i:
                continue
            if dist[i, j] < best_d:
                best_d = dist[i, j]
                best_j = j
        if best_j != -1 and best_d < tol:
            refs.append(i)
            mirrors.append(best_j)
            used[i] = True
            used[best_j] = True
        else:
            # 找不到对称伙伴, 当作自身对称处理 (实测取 CP, 不镜像)
            self_sym.append(i)
            used[i] = True
    return refs, mirrors, self_sym


def extract_sraf_cps(
    sraf_mask_raw: np.ndarray,
    arclen_step: float,
    min_cps: int,
    symmetry: Optional[str] = None,
    min_block_area: int = 20,
    binarize_threshold: float = 1e-3,
) -> List[np.ndarray]:
    """对 SRAF 灰度图按连通块提取闭合 CP.

    Args:
        sraf_mask_raw:    LSM 中扣掉主图形后的 SRAF 区域 (灰度 / 二值均可).
        arclen_step:      沿轮廓弧长每隔多少像素取 1 个 CP.
        min_cps:          每个连通块最少 CP 数 (闭合 B-spline 至少 4, 推荐 ≥ 8).
        symmetry:         若给定, 则按对称变换镜像 CP 保证两组严格对称.
        binarize_threshold: 二值化阈值, 大于该值视为 SRAF 像素.
        min_block_area:   小于该面积的连通块当噪点丢弃.
    Returns:
        sraf_cps: list[ ndarray(K_i, 2) ]  按 [y, x] 排列, 每个块一组闭合 CP.
        nums_contours_sraf: int  所有 SRAF 块原始外轮廓像素点的总数 (调试/统计用).
    """
    bin_img = (np.asarray(sraf_mask_raw) > binarize_threshold).astype(np.uint8)
    if bin_img.sum() == 0:
        return [], 0

    # 找连通块
    labeled = label(bin_img, connectivity=2)
    props = regionprops(labeled)
    valid_props = [p for p in props if p.area >= min_block_area]
    if not valid_props:
        return [], 0

    # 取每个块的外轮廓 (cv2 轮廓的坐标顺序是 [x, y], 这里转成 [y, x])
    block_contours: dict[int, np.ndarray] = {}
    nums_contours_sraf = 0
    for p in valid_props:
        single = (labeled == p.label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        # 选最长的轮廓 (理论上每个连通块只有一个外轮廓)
        c = max(contours, key=lambda c: len(c))
        contour_yx = c[:, 0, [1, 0]].astype(np.float64)
        nums_contours_sraf += len(contour_yx)
        block_contours[p.label] = contour_yx

    labels_sorted = sorted(block_contours.keys())
    centroids = np.array([
        np.mean(block_contours[l], axis=0) for l in labels_sorted
    ])

    sraf_cps: List[np.ndarray] = [None] * len(labels_sorted)  # type: ignore

    sym = None if (symmetry is None or symmetry.lower() == "none") else symmetry
    if sym is None:
        # 无对称, 全部独立采样
        for i, l in enumerate(labels_sorted):
            sraf_cps[i] = _resample_contour_by_arclen(
                block_contours[l], arclen_step, min_cps,
            )
        return sraf_cps, nums_contours_sraf

    # 有对称: 配对后, refs 实测, mirrors 由 refs 镜像得到
    refs, mirrors, self_sym = _pair_blocks_by_symmetry(centroids, sraf_mask_raw.shape, sym)

    # 自身对称 / 找不到伙伴的块: 直接实测
    for idx in self_sym:
        l = labels_sorted[idx]
        sraf_cps[idx] = _resample_contour_by_arclen(
            block_contours[l], arclen_step, min_cps,
        )

    # 配对块: ref 实测, mirror 由 ref 镜像
    for ref_idx, mir_idx in zip(refs, mirrors):
        ref_label = labels_sorted[ref_idx]
        ref_pts = _resample_contour_by_arclen(
            block_contours[ref_label], arclen_step, min_cps,
        )
        sraf_cps[ref_idx] = ref_pts
        sraf_cps[mir_idx] = _apply_symmetry(ref_pts, sraf_mask_raw.shape, sym)

    # 兜底: 任何 None 都用实测填上 (理论上不会出现)
    for i, l in enumerate(labels_sorted):
        if sraf_cps[i] is None:
            sraf_cps[i] = _resample_contour_by_arclen(
                block_contours[l], arclen_step, min_cps,
            )

    return sraf_cps, nums_contours_sraf
