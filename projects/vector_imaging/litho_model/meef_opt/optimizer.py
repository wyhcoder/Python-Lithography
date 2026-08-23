"""
VectorMEEFOptimizer: 基于矢量 SOCS 成像 + 批量中心差分的 MEEF 优化器。

核心优化:
    - 所有 ±delta 扰动 mask 一次性堆成 [2*num_cps, N, N] batch
    - 一次 forward_batch (raw aerial) 完成全部仿真
    - 相比逐 mask 仿真的线性加速比 ≈ batch_size / GPU_SM_count
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from .utils import (
    SDFMaskRenderer,
    compute_epe_batch,
    compute_wepe,
    l_curve_tsvd_solve,
    extract_contour_cps,
    get_cp_directions,
    select_ep_points,
)


# =============================================================================
# 批量 aerial 计算 (raw, 不过 sigmoid)
# =============================================================================

def compute_aerial_batch(
    masks: torch.Tensor,
    socs_kernels: torch.Tensor,
    device: torch.device,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    """
    批量计算 raw aerial intensity (SOCS 路径，不过 sigmoid)。

    与 VectorSOCSImaging._aerial_socs / forward_batch 的物理逻辑完全一致，
    但不经过 sigmoid，返回 raw aerial 用于 EPE 计算。

    Parameters
    ----------
    masks : (B, N, N) real tensor
    socs_kernels : (K, C, N, N) complex tensor (from VectorSOCS.get_kernels())
    device, complex_dtype : 设备与数据类型

    Returns
    -------
    aerial : (B, N, N) real tensor
    """
    from tool.fft_tool import fft2c, ifft2c

    masks_c = masks.to(device=device).to(complex_dtype)

    M_batch = fft2c(masks_c)                                          # [B, N, N]
    MK = M_batch[:, None, None, :, :] * socs_kernels[None, :, :, :, :]  # [B, K, C, N, N]
    E = ifft2c(MK)                                                     # [B, K, C, N, N]
    I = (E.real * E.real + E.imag * E.imag).sum(dim=(1, 2))            # [B, N, N]

    C = socs_kernels.shape[1]
    if C == 6:   # 非偏振 SOCS: C=6 时 I = (Ix + Iy)，需除以 2
        I = I * 0.5
    return I


class VectorMEEFOptimizer:
    """
    矢量 MEEF 批量优化器。

    使用流程:
        opt = VectorMEEFOptimizer(forward_imaging, target_mask, config_dict)
        opt.setup_contour(cps_list, eps_px, ep_weights)
        opt.run(iterations=30)
    """

    def __init__(
        self,
        socs_imaging,           # VectorSOCSImaging instance
        target_mask: np.ndarray, # (H, W) target pattern
        config: dict,
    ):
        """
        Parameters
        ----------
        socs_imaging : VectorSOCSImaging
            已构建的矢量 SOCS 成像模型 (必须已包含 VectorSOCS)
        target_mask : (H, W) ndarray
            目标二值掩膜 [0, 1]
        config : dict with keys:
            - delta: float, CFD 扰动量 (默认 0.5 像素)
            - iterations: int, 优化迭代次数
            - sdf_beta: float, SDF 软化宽度
            - threshold: float, resist 阈值 (默认 0.25)
            - curve_type: str, "OA" (默认折线) 或 "BS" (B-spline)
            - output_dir: str, 输出目录
            - batch_size: int, 每批仿真的 mask 数 (0=全部一起)
        """
        self.socs_imaging = socs_imaging
        self.target_mask = target_mask.astype(np.float64)
        self.H, self.W = target_mask.shape

        # 配置
        self.delta = float(config.get("delta", 0.5))
        self.iterations = int(config.get("iterations", 30))
        self.sdf_beta = float(config.get("sdf_beta", 0.25))
        self.threshold = float(config.get("threshold", 0.25))
        self.curve_type = config.get("curve_type", "OA")
        self.output_dir = Path(config.get("output_dir", "./output/meef_opt"))
        self.batch_size = int(config.get("batch_size", 0))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SOCS 内核 (预计算一次)
        self._socs_kernels = socs_imaging.socs.get_kernels()
        self._device = socs_imaging.device
        self._cdtype = socs_imaging.complex_dtype

        # 渲染器
        self.renderer = SDFMaskRenderer(beta=self.sdf_beta, device=str(self._device))

        # 优化状态 (在 setup_contour 中初始化)
        self.cps_list: list = []
        self.eps_px: np.ndarray = None
        self.ep_weights: np.ndarray = None
        self._num_cps: int = 0
        self._num_eps: int = 0

    # -----------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------

    def setup_contour(
        self,
        cps_list: list[np.ndarray],
        eps_px: np.ndarray,
        ep_weights: np.ndarray | None = None,
    ):
        """
        设置轮廓控制点和 EP 评估点。

        Parameters
        ----------
        cps_list : list of (N_i, 2) ndarray [y, x]
        eps_px : (K, 2) ndarray [y, x] EP 点坐标
        ep_weights : (K,) ndarray or None
        """
        self.cps_list = [c.astype(np.float64).copy() for c in cps_list]
        self.eps_px = eps_px.astype(np.int64)
        self.ep_weights = (
            ep_weights.astype(np.float64)
            if ep_weights is not None
            else np.ones(len(eps_px), dtype=np.float64)
        )
        self._num_cps = sum(len(c) for c in self.cps_list)
        self._num_eps = len(eps_px)
        print(f"[Setup] {len(cps_list)} 条曲线, {self._num_cps} CPs, "
              f"{self._num_eps} EPs")

    def setup_auto(self, interval: int = 6):
        """从 target_mask 自动提取 CP + EP。"""
        cps_list = extract_contour_cps(self.target_mask, interval=interval)
        eps_px, ep_weights = select_ep_points(cps_list, self.target_mask)
        self.setup_contour(cps_list, eps_px, ep_weights)

    # -----------------------------------------------------------------
    # Mask 渲染
    # -----------------------------------------------------------------

    def _render_mask(self, cps_list: list[np.ndarray]) -> np.ndarray:
        """将 CP 渲染为 (H, W) mask。"""
        if self.curve_type == "OA":
            polygons = cps_list
        else:
            # B-spline: 使用 scipy 拟合
            from scipy.interpolate import splprep, splev
            polygons = []
            for cps in cps_list:
                if len(cps) < 4:
                    polygons.append(cps)
                    continue
                cps_yx = cps[:, ::-1]  # [y, x] → [x, y] for splprep
                try:
                    tck, _ = splprep([cps_yx[:, 0], cps_yx[:, 1]],
                                     s=0.3, per=True)
                    u = np.linspace(0, 1, max(200, len(cps) * 4))
                    x, y = splev(u, tck)
                    smooth = np.stack([y, x], axis=-1).astype(np.float64)
                    polygons.append(smooth)
                except Exception:
                    polygons.append(cps)
        return self.renderer.render(polygons, self.H, self.W).cpu().numpy()

    # -----------------------------------------------------------------
    # MEEF 矩阵构建 (批量 CFD)
    # -----------------------------------------------------------------

    def _build_meef_matrix_batch(
        self,
        cps_list: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        批量中心差分构建 MEEF 矩阵。

        核心优化: 全部 ±delta mask 一次性批量仿真。

        Parameters
        ----------
        cps_list : 当前 CP 列表 (None 时退回到 self.cps_list)

        Returns
        -------
        M : (num_eps, num_cps) MEEF 矩阵
        e0 : (num_eps,) 当前 EPE 向量
        """
        if cps_list is None:
            cps_list = self.cps_list
        vec_list = get_cp_directions(cps_list)

        # ---- Step 1: 多线程渲染所有扰动 mask ----
        print(f"[MEEF] 渲染 {2 * self._num_cps} 张扰动 mask ...")
        t0 = time.time()

        # 构建任务: (col_idx, side, contour_idx, point_idx, vec)
        tasks = []
        col = 0
        for ci, cps in enumerate(cps_list):
            for pi in range(len(cps)):
                for side in ("plus", "minus"):
                    tasks.append((col, side, ci, pi, vec_list[ci][pi]))
                col += 1

        # 多线程渲染
        all_masks = [None] * (2 * self._num_cps)

        def _render_one(task):
            col_idx, side, ci, pi, vec = task
            sign = +1.0 if side == "plus" else -1.0
            # 深拷贝 CP 并扰动
            perturbed = [c.copy() for c in cps_list]
            perturbed[ci] = perturbed[ci].copy()
            perturbed[ci][pi] = perturbed[ci][pi] + sign * self.delta * vec
            mask = self._render_mask(perturbed)
            return col_idx, side, mask

        with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
            futures = [pool.submit(_render_one, t) for t in tasks]
            for f in as_completed(futures):
                col_idx, side, mask = f.result()
                slot = col_idx * 2 + (0 if side == "plus" else 1)
                all_masks[slot] = mask

        t_render = time.time() - t0
        print(f"  渲染完成: {t_render:.1f}s")

        # ---- Step 2: 分批批量仿真 ----
        print(f"[MEEF] 批量仿真 (batch_size={self.batch_size or 'all'}) ...")
        t0 = time.time()

        B_total = len(all_masks)
        if self.batch_size <= 0:
            batch_sizes = [B_total]
        else:
            batch_sizes = [self.batch_size] * (B_total // self.batch_size)
            if B_total % self.batch_size:
                batch_sizes.append(B_total % self.batch_size)

        epe_plus_list = [None] * self._num_cps
        epe_minus_list = [None] * self._num_cps

        offset = 0
        for bs in batch_sizes:
            batch_np = np.stack(all_masks[offset:offset + bs], axis=0)
            batch_t = torch.from_numpy(batch_np.astype(np.float32)).to(self._device)

            # 批量计算 aerial
            with torch.no_grad():
                aerial_batch = compute_aerial_batch(
                    batch_t, self._socs_kernels, self._device, self._cdtype,
                )  # [bs, H, W]
                _, epe_vecs = compute_epe_batch(
                    aerial_batch, self.threshold, self.eps_px,
                )  # epe_vecs: [bs, K]

            # 拆分回 ±delta
            for k in range(bs):
                global_idx = offset + k
                col = global_idx // 2
                is_plus = (global_idx % 2 == 0)
                epe_np = epe_vecs[k].cpu().numpy().astype(np.float64)
                if is_plus:
                    epe_plus_list[col] = epe_np
                else:
                    epe_minus_list[col] = epe_np
            offset += bs

        t_sim = time.time() - t0
        print(f"  仿真完成: {t_sim:.1f}s")

        # ---- Step 3: 中心差分 + 组装 M ----
        epe_plus = np.stack(epe_plus_list, axis=0)    # (num_cps, num_eps)
        epe_minus = np.stack(epe_minus_list, axis=0)  # (num_cps, num_eps)
        # 重新算当前 EPE (用于 GN 求解的右端项 e0)
        current_mask = self._render_mask(cps_list)
        current_t = torch.from_numpy(current_mask.astype(np.float32)).to(self._device)
        with torch.no_grad():
            aerial0 = compute_aerial_batch(
                current_t.unsqueeze(0), self._socs_kernels, self._device, self._cdtype,
            )
            _, epe0_vecs = compute_epe_batch(aerial0, self.threshold, self.eps_px)
            e0 = epe0_vecs[0].cpu().numpy().astype(np.float64)

        grad = (epe_plus - epe_minus) / (2.0 * self.delta)  # (num_cps, num_eps)
        M = np.round(grad, 6).T  # (num_eps, num_cps)

        print(f"  M shape = {M.shape}, "
              f"|M| range = [{np.abs(M).min():.4e}, {np.abs(M).max():.4e}]")
        return M, e0

    # -----------------------------------------------------------------
    # 优化主循环
    # -----------------------------------------------------------------

    def run(self):
        """执行 Gauss-Newton 优化。"""
        if self._num_cps == 0:
            raise RuntimeError("请先调用 setup_contour 或 setup_auto")

        # 初始评估
        print(f"\n{'='*60}")
        print(f"Vector MEEF Batch Optimizer")
        print(f"{'='*60}")
        print(f"  CPs: {self._num_cps}, EPs: {self._num_eps}")
        print(f"  delta: {self.delta}, threshold: {self.threshold}")
        print(f"  iterations: {self.iterations}, curve: {self.curve_type}")

        current_cps = [c.copy() for c in self.cps_list]
        vec_list = get_cp_directions(current_cps)

        current_mask = self._render_mask(current_cps)
        mask_t = torch.from_numpy(current_mask.astype(np.float32)).to(self._device)
        with torch.no_grad():
            aerial0 = compute_aerial_batch(
                mask_t.unsqueeze(0), self._socs_kernels, self._device, self._cdtype,
            )
            _, epe0_vecs = compute_epe_batch(aerial0, self.threshold, self.eps_px)
        cur_epe_vec = epe0_vecs[0].cpu().numpy().astype(np.float64)
        cur_epe = float(cur_epe_vec.sum())
        cur_wepe = compute_wepe(cur_epe_vec, self.ep_weights, self._num_eps)

        print(f"  Initial: EPE={cur_epe/self._num_eps:.6f}, "
              f"wEPE={cur_wepe/self._num_eps:.6f}")

        # 最优追踪
        best_wepe = cur_wepe
        best_epe = cur_epe
        best_cps = [c.copy() for c in current_cps]

        epe_errors = [cur_epe / self._num_eps]

        for it in range(1, self.iterations + 1):
            t0 = time.time()
            print(f"\n--- Iteration {it}/{self.iterations} ---")

            # 每轮根据 current_cps 重算法向量 (CP 移动后角平分方向变化)
            vec_list = get_cp_directions(current_cps)

            # 构建 MEEF 矩阵 (批量 CFD), 基于当前 CP
            M, e0_vec = self._build_meef_matrix_batch(cps_list=current_cps)

            # MEEF 加权
            M_weighted = M.copy()
            w_flat = self.ep_weights.flatten()
            M_weighted *= w_flat[:, np.newaxis]

            # TSVD 求解
            delta_d = l_curve_tsvd_solve(M_weighted, e0_vec)
            print(f"  |Δd| max = {np.abs(delta_d).max():.4f}, "
                  f"mean = {np.abs(delta_d).mean():.4f}")

            # 更新 CP
            new_cps = []
            offset = 0
            for cps, vecs in zip(current_cps, vec_list):
                n_local = len(cps)
                dd = delta_d[offset:offset + n_local]
                new_c = cps.copy()
                for i in range(n_local):
                    new_c[i] = cps[i] + dd[i] * vecs[i]
                new_cps.append(new_c)
                offset += n_local

            # 评估
            current_mask = self._render_mask(new_cps)
            mask_t = torch.from_numpy(current_mask.astype(np.float32)).to(self._device)
            with torch.no_grad():
                aerial_cur = compute_aerial_batch(
                    mask_t.unsqueeze(0), self._socs_kernels,
                    self._device, self._cdtype,
                )
                _, epe_cur_vecs = compute_epe_batch(
                    aerial_cur, self.threshold, self.eps_px,
                )
            cur_epe_vec = epe_cur_vecs[0].cpu().numpy().astype(np.float64)
            cur_epe = float(cur_epe_vec.sum())
            cur_wepe = compute_wepe(cur_epe_vec, self.ep_weights, self._num_eps)

            elapsed = time.time() - t0
            print(f"  EPE={cur_epe/self._num_eps:.6f}, "
                  f"wEPE={cur_wepe/self._num_eps:.6f}, "
                  f"time={elapsed:.1f}s")

            epe_errors.append(cur_epe / self._num_eps)

            if cur_wepe < best_wepe:
                best_wepe = cur_wepe
                best_epe = cur_epe
                best_cps = [c.copy() for c in new_cps]
                print(f"  ★ new best wEPE!")

            current_cps = new_cps

        # 保存结果
        print(f"\n{'='*60}")
        print(f"Optimization done.")
        print(f"  Best EPE:  {best_epe/self._num_eps:.6f}")
        print(f"  Best wEPE: {best_wepe/self._num_eps:.6f}")

        best_mask = self._render_mask(best_cps)
        np.save(str(self.output_dir / "best_mask.npy"), best_mask)
        np.save(str(self.output_dir / "target_mask.npy"), self.target_mask)

        # ---- 计算最优 mask 对应的 wafer image (经过 sigmoid resist) ----
        best_mask_t = torch.from_numpy(
            best_mask.astype(np.float32)
        ).to(self._device)
        with torch.no_grad():
            best_wafer = self.socs_imaging.forward(best_mask_t)
        best_wafer_np = best_wafer.cpu().numpy()
        np.save(str(self.output_dir / "best_wafer.npy"), best_wafer_np)

        # ---- 可视化: target / 优化后 mask / 优化后 wafer image ----
        self._save_visualization(best_mask, best_wafer_np)
        print(f"  Results saved to {self.output_dir}")

        return best_cps, best_mask

    # -----------------------------------------------------------------
    # 可视化
    # -----------------------------------------------------------------

    def _save_visualization(
        self,
        best_mask: np.ndarray,
        best_wafer: np.ndarray,
    ) -> None:
        """保存 target / 优化后 mask / 优化后 wafer 三联图 + 轮廓叠加图."""
        import matplotlib.pyplot as plt

        out_path = self.output_dir / "result.png"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1) Target
        ax = axes[0]
        ax.imshow(self.target_mask, cmap="gray", vmin=0, vmax=1, origin="upper")
        ax.set_title("Target Pattern")
        ax.axis("off")

        # 2) 优化后 mask
        ax = axes[1]
        ax.imshow(best_mask, cmap="gray", vmin=0, vmax=1, origin="upper")
        ax.set_title("Optimized Mask")
        ax.axis("off")

        # 3) 优化后 wafer image (sigmoid 后)
        ax = axes[2]
        im = ax.imshow(best_wafer, cmap="inferno", vmin=0, vmax=1, origin="upper")
        ax.set_title("Wafer Image (after resist)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        plt.savefig(str(out_path), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  [vis] 三联图: {out_path}")

        # 额外: target 轮廓 (红) 叠加在 wafer 上, 方便看 EPE 偏差
        try:
            import cv2
            # 二值化 target 取轮廓
            tgt_u8 = (self.target_mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                tgt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            overlay_path = self.output_dir / "wafer_with_target_contour.png"
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.imshow(best_wafer, cmap="gray", vmin=0, vmax=1, origin="upper")
            for cnt in contours:
                pts = cnt.squeeze(1)  # (M, 2) [x, y]
                if pts.ndim != 2:
                    continue
                # matplotlib 的 plot 用 (x, y) 顺序
                ax2.plot(pts[:, 0], pts[:, 1], color="red", linewidth=1.2)
            ax2.set_title("Wafer image + Target contour (red)")
            ax2.axis("off")
            plt.tight_layout()
            plt.savefig(str(overlay_path), dpi=140, bbox_inches="tight")
            plt.close(fig2)
            print(f"  [vis] 轮廓叠加图: {overlay_path}")
        except ImportError:
            # 没有 cv2 时跳过轮廓叠加图
            pass
