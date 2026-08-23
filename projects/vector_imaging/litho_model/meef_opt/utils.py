"""
MEEF 批量优化工具函数:
    - EPE/wEPE 计算（torch 实现，支持 batch）
    - L-curve TSVD 求解器
    - SDF 渲染与 CP 处理
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import linalg


# =============================================================================
# EPE 计算 (torch 版，支持 batch)
# =============================================================================

def compute_epe_batch(
    aerials: torch.Tensor,
    threshold: float,
    eps_px: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    批量计算 EPE。

    Parameters
    ----------
    aerials : (B, H, W) tensor — 光强图 (已过 resist sigmoid 或 aerial)
    threshold : float — resist 阈值
    eps_px : (K, 2) ndarray — EP 采样点 [y, x] 像素坐标

    Returns
    -------
    epe_sums  : (B,) tensor — 每张图的总 EPE
    epe_vecs  : (B, K) tensor — 每张图每个 EP 点的 EPE 向量
    """
    B, H, W = aerials.shape
    K = eps_px.shape[0]
    device = aerials.device
    dtype = aerials.dtype

    # 中心差分梯度 (与 numpy.gradient(aI, 0.5) 等价)
    dx_inner = (torch.roll(aerials, -1, dims=2) - torch.roll(aerials, 1, dims=2))
    dy_inner = (torch.roll(aerials, -1, dims=1) - torch.roll(aerials, 1, dims=1))

    # 边界修正
    dx = dx_inner.clone()
    dx[:, :, 0] = (aerials[:, :, 1] - aerials[:, :, 0]) / 0.5
    dx[:, :, -1] = (aerials[:, :, -1] - aerials[:, :, -2]) / 0.5

    dy = dy_inner.clone()
    dy[:, 0, :] = (aerials[:, 1, :] - aerials[:, 0, :]) / 0.5
    dy[:, -1, :] = (aerials[:, -1, :] - aerials[:, -2, :]) / 0.5

    G = torch.sqrt(dx.pow(2) + dy.pow(2)).clamp_min(1e-12)  # (B, H, W)
    epe_map = ((aerials - threshold) / G).pow(2)              # (B, H, W)

    # EP 采样
    rows = torch.tensor(eps_px[:, 0], dtype=torch.long, device=device)
    cols = torch.tensor(eps_px[:, 1], dtype=torch.long, device=device)
    epe_vecs = epe_map[:, rows, cols]                         # (B, K)
    epe_sums = epe_vecs.sum(dim=-1)                           # (B,)
    return epe_sums, epe_vecs


def compute_wepe(epe_vector: np.ndarray,
                 weight: np.ndarray,
                 num_weps: int) -> float:
    """计算 wEPE. weight 和 epe_vector 都是 (num_eps,)."""
    return float(np.sum(weight[:num_weps] * epe_vector[:num_weps]))


# =============================================================================
# SDF 渲染器封装 (numpy → torch)
# =============================================================================

class SDFMaskRenderer:
    """对 utils_model SDFRenderer 的轻量封装，输出转换为 torch tensor."""

    def __init__(self, beta: float = 0.25, device: str = "cpu"):
        self.beta = beta
        self.device = torch.device(device)
        # 包内自包含的 SDF 实现 (与 utils_model.demo_sdf_renderer 等价)
        from .sdf_renderer import SDFRenderer
        self._renderer = SDFRenderer(beta=beta)

    def render(self, polygons: list, H: int, W: int) -> torch.Tensor:
        """
        polygons: list of (N_i, 2) ndarray [y, x]
        返回 (H, W) torch tensor [0, 1]
        """
        mask_np = self._renderer.MSAA(polygons, np.zeros((H, W)), type="gray")
        return torch.from_numpy(mask_np.astype(np.float32)).to(self.device)


# =============================================================================
# Gauss-Newton 求解器 (L-curve + TSVD)
# =============================================================================

def l_curve_tsvd_solve(M: np.ndarray, e0: np.ndarray) -> np.ndarray:
    """
    L-curve + TSVD 求解 Δd:
        min ||M·Δd + e0||² + λ²||Δd||²

    Parameters
    ----------
    M : (num_eps, num_cps) MEEF 矩阵
    e0 : (num_eps,) 当前 EPE 向量

    Returns
    -------
    delta_d : (num_cps,) 推荐 CP 位移
    """
    # 截断小值
    M_proc = M.copy()
    M_proc[np.abs(M_proc) < 1e-3] = 0.0

    # L-curve 扫描
    lambdas = np.logspace(-6, 2, 100)
    U, s, Vt = linalg.svd(M_proc, full_matrices=False)

    # 能量截断 (96%)
    s_cumsum = np.cumsum(s**2)
    k = max(1, int(np.searchsorted(s_cumsum, 0.96 * s_cumsum[-1]) + 1))
    k = min(k, len(s))
    s_k = s[:k]; U_k = U[:, :k]; Vt_k = Vt[:k, :]

    best_lambda = lambdas[0]
    best_curvature = -np.inf
    residuals = []; solutions = []
    for lam in lambdas:
        denom = s_k**2 + lam**2
        x = (Vt_k.T * (s_k / denom)) @ (U_k.T @ (-e0))
        resid = np.linalg.norm(M_proc @ x + e0)
        soln = np.linalg.norm(x)
        residuals.append(resid)
        solutions.append(soln)
    residuals = np.array(residuals); solutions = np.array(solutions)

    if len(residuals) >= 3:
        log_r = np.log(residuals + 1e-30)
        log_s = np.log(solutions + 1e-30)
        dlog_r = np.gradient(log_r)
        d2log_r = np.gradient(dlog_r)
        dlog_s = np.gradient(log_s)
        d2log_s = np.gradient(dlog_s)
        num = dlog_r * d2log_s - d2log_r * dlog_s
        denom = (dlog_r**2 + dlog_s**2) ** 1.5 + 1e-30
        curvature = num / denom
        idx = np.argmax(curvature)
        best_lambda = lambdas[idx]

    # 最终求解
    denom = s_k**2 + best_lambda**2
    delta_d = (Vt_k.T * (s_k / denom)) @ (U_k.T @ (-e0))
    return np.round(delta_d.astype(np.float64), 6)


# =============================================================================
# CP 提取 (简易 demo 版)
# =============================================================================

def extract_contour_cps(
    mask: np.ndarray,
    interval: int = 6,
    smooth_window: int = 3,
) -> list[np.ndarray]:
    """
    从二值 mask 提取轮廓控制点。

    Parameters
    ----------
    mask : (H, W) ndarray, 二值 [0, 1]
    interval : CP 间隔 (每隔 interval 个轮廓点取 1 个 CP)
    smooth_window : 简单平滑窗大小

    Returns
    -------
    cps : list of (N_i, 2) ndarray [y, x]
    """
    import cv2
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cps_list = []
    for cnt in contours:
        pts = cnt.squeeze(1)  # (M, 2) [x, y] (cv2 格式)
        if pts.ndim != 2 or pts.shape[0] < 4:
            continue
        # 转换为 [y, x] 格式
        pts_yx = pts[:, ::-1].astype(np.float64)
        # 等距采样 CP
        sampled = pts_yx[::interval]
        if len(sampled) < 4:
            sampled = pts_yx[::max(1, len(pts_yx) // 4)]
        # 简单平滑
        if smooth_window > 1 and len(sampled) > smooth_window:
            from scipy.ndimage import uniform_filter1d
            smoothed = np.zeros_like(sampled)
            smoothed[:, 0] = uniform_filter1d(
                np.pad(sampled[:, 0], smooth_window, mode="wrap"),
                smooth_window,
            )[smooth_window:-smooth_window]
            smoothed[:, 1] = uniform_filter1d(
                np.pad(sampled[:, 1], smooth_window, mode="wrap"),
                smooth_window,
            )[smooth_window:-smooth_window]
            sampled = smoothed
        cps_list.append(sampled)
    return cps_list


def get_cp_directions(cps_list: list[np.ndarray]) -> list[list]:
    """
    计算每个 CP 的角平分外法向。

    Returns
    -------
    vectors : list of list of (2,) ndarray [y, x] — 单位外法向
    """
    all_vectors = []
    for cps in cps_list:
        N = len(cps)
        vecs = []
        for i in range(N):
            prev = cps[(i - 1) % N]
            curr = cps[i]
            next_ = cps[(i + 1) % N]
            # 入射边和出射边
            e1 = curr - prev
            e2 = next_ - curr
            n1 = np.linalg.norm(e1) + 1e-12
            n2 = np.linalg.norm(e2) + 1e-12
            # 角平分方向 (向内 or 向外）
            bisector = e1 / n1 + e2 / n2
            bn = np.linalg.norm(bisector) + 1e-12
            bisector = bisector / bn
            # 指向外部 (用 mask 中心判断)
            center = cps.mean(axis=0)
            to_center = center - curr
            if np.dot(bisector, to_center) > 0:
                bisector = -bisector
            vecs.append(bisector)
        all_vectors.append(vecs)
    return all_vectors


def select_ep_points(cps_list: list[np.ndarray],
                     target_mask: np.ndarray,
                     interval_line: int = 4,
                     interval_corner: int = 2,
                     corner_threshold_deg: float = 60.0,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    沿目标轮廓选取 EP 评估点。

    Returns
    -------
    eps : (K, 2) ndarray [y, x] 评估点坐标
    weight : (K,) ndarray 权重 (角点处更高)
    """
    H, W = target_mask.shape
    eps_list = []
    weight_list = []
    for cps in cps_list:
        N = len(cps)
        for i in range(N):
            prev = cps[(i - 1) % N]
            curr = cps[i]
            next_ = cps[(i + 1) % N]
            # 计算夹角判断是否为角点
            e1 = prev - curr
            e2 = next_ - curr
            n1 = np.linalg.norm(e1); n2 = np.linalg.norm(e2)
            cos_a = np.dot(e1, e2) / (n1 * n2 + 1e-12)
            cos_a = np.clip(cos_a, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_a))
            is_corner = abs(180.0 - angle_deg) > corner_threshold_deg
            # 沿从 curr 到 next_ 的线段采样
            seg_len = n2
            if is_corner:
                n_samples = max(2, int(seg_len / interval_corner))
                w = 2.0  # 角点高权重
            else:
                n_samples = max(2, int(seg_len / interval_line))
                w = 1.0
            ts = np.linspace(0, 1, n_samples, endpoint=False)
            for t in ts:
                pt = curr + t * e2
                py, px = int(round(pt[0])), int(round(pt[1]))
                py = np.clip(py, 0, H - 1)
                px = np.clip(px, 0, W - 1)
                eps_list.append([py, px])
                weight_list.append(w)
    return np.array(eps_list, dtype=np.int64), np.array(weight_list, dtype=np.float64)
