"""
MEEF_GradientDescentOptimizer：MSAA + 中心差分 + 梯度下降。

核心思路:
    1. 用 MSAA 渲染 (AntiAliasRenderer) + images_simulation 做前向仿真
    2. 对每个 CP 的 X / Y 方向做 ±delta 中心差分扰动 (与建 MEEF 矩阵完全一样)
    3. 从扰动结果算出 wEPE 对每个 CP (x, y) 的标量梯度:
         grad_x[i] = Σ_j w_j · ∂EPE_j/∂x_i = Σ_j w_j · Mx[j, i]
         grad_y[i] = Σ_j w_j · ∂EPE_j/∂y_i = Σ_j w_j · My[j, i]
       (等价于直接对标量 wEPE 做中心差分)
    4. 梯度下降: cps -= lr · grad

不使用 SDF 渲染、不使用 PyTorch autograd、不使用 TSVD / L-curve。
复用 demo_meef.py 的 _build_meef_matrix_xy_parallel_threads (MSAA + 并行线程 + 中心差分)。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from litho_model.lithography_simulator import LithographySimulator
from litho_model.demo_compute_image import images_simulation_v2
from op_model.demo_epe_wepe import caculate_epe, caculate_wepe
from op_model.demo_meef import (
    _build_meef_matrix_xy_parallel_threads,
    _build_meef_matrix_parallel_threads,
    _save_cp_history,
)
from utils_model.demo_MSAA import AntiAliasRenderer
from utils_model.demo_parametric import ParametricDemo
from utils_model.demo_meef_utils import (
    save_iteration_results, save_excel, drawcostcurve, show_sample_mcps,
    highlight_contour_red, _save_images,
    get_new_cps_xy, get_new_cps, get_cp_vectors,
    save_matrix_nm, save_bspline_curve_mask,
)


# =============================================================================
# Numpy 版 Adam 优化器 (无需 PyTorch)
# =============================================================================

class SimpleAdam:
    """一维 numpy 数组的 Adam 优化器。"""

    def __init__(self, size: int, lr: float = 0.1,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(size)
        self.v = np.zeros(size)
        self.t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        """返回本次应施加的位移 = -lr * adam_corrected_grad。"""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m[:] = 0
        self.v[:] = 0
        self.t = 0


# =============================================================================
# Numpy 版 L-BFGS 优化器 (无线搜索, 用固定步长, 避免多次前向仿真)
# =============================================================================

class SimpleLBFGS:
    """
    有限内存 BFGS 优化器 (无需 PyTorch)。

    与标准 L-BFGS 的区别: 不做 Wolfe 线搜索 (那需要多次 f/∇f, 太贵),
    改用固定 lr 直接施加 L-BFGS 方向 d = -H⁻¹·g。

    公式 (Nocedal 两次循环递归, memory m):
        q = grad
        for i in range(k-1, k-m-1, -1):
            α_i = ρ_i · s_iᵀ · q
            q -= α_i · y_i
        r = γ · q                    # γ = s_last·y_last / y_last·y_last (初始 H₀)
        for i in range(k-m, k):
            β_i = ρ_i · y_iᵀ · r
            r += (α_i - β_i) · s_i
        return -lr * r               # r = H⁻¹·grad (近似)

    其中 s_k = x_{k+1} - x_k, y_k = ∇f_{k+1} - ∇f_k, ρ_k = 1/(y_kᵀ·s_k).
    """

    def __init__(self, size: int, lr: float = 1.0, history_size: int = 10):
        self.lr = lr
        self.m = history_size
        self.size = size
        self.s_list = []          # 存最近 m 个 (Δx)
        self.y_list = []          # 存最近 m 个 (Δg)
        self.rho_list = []        # ρ = 1 / (yᵀ·s)
        self._prev_x = None
        self._prev_grad = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        输入当前变量 x 和梯度 grad, 返回本次应施加的位移 delta。
        （调用者做 x_new = x + delta）
        """
        # 更新 (s, y) 历史
        if self._prev_x is not None:
            s = x - self._prev_x
            y = grad - self._prev_grad
            ys = float(y @ s)
            # 曲率条件: yᵀs > 0 才更新 (否则跳过, 保持正定)
            if ys > 1e-10:
                self.s_list.append(s)
                self.y_list.append(y)
                self.rho_list.append(1.0 / ys)
                if len(self.s_list) > self.m:
                    self.s_list.pop(0)
                    self.y_list.pop(0)
                    self.rho_list.pop(0)

        self._prev_x = x.copy()
        self._prev_grad = grad.copy()

        # 两次循环递归
        q = grad.copy()
        n_hist = len(self.s_list)
        if n_hist == 0:
            # 冷启动: 直接梯度下降
            return -self.lr * grad

        alpha = np.zeros(n_hist)
        for i in range(n_hist - 1, -1, -1):
            alpha[i] = self.rho_list[i] * (self.s_list[i] @ q)
            q -= alpha[i] * self.y_list[i]

        # 初始 H₀: γ·I, γ = (s·y) / (y·y) 取最近一对 (Nocedal & Wright 推荐)
        s_last = self.s_list[-1]
        y_last = self.y_list[-1]
        gamma = float(s_last @ y_last) / max(float(y_last @ y_last), 1e-12)
        r = gamma * q

        for i in range(n_hist):
            beta_i = self.rho_list[i] * (self.y_list[i] @ r)
            r += (alpha[i] - beta_i) * self.s_list[i]

        return -self.lr * r

    def reset(self):
        self.s_list.clear()
        self.y_list.clear()
        self.rho_list.clear()
        self._prev_x = None
        self._prev_grad = None


# =============================================================================
# 主类
# =============================================================================

class MEEF_GradientDescentOptimizer:
    """
    MSAA + 中心差分 + 梯度下降版 MEEF 优化器。

    Parameters
    ----------
    simulator        : LithographySimulator
    iterations       : int   外层迭代次数
    lr               : float 学习率 (Adam 默认 0.1, GD 默认 0.01)
    optimizer_type   : str   'adam' (默认) 或 'gd' (普通梯度下降)
    grad_clip        : float 梯度范数裁剪上限 (0 = 不裁剪)
    direction        : str   'xy' (X/Y 解耦, 4N 次前向) 或 'bisector' (角平分线, 2N 次)
    """

    def __init__(self,
                 simulator: LithographySimulator,
                 iterations: int,
                 lr: float = 0.1,
                 optimizer_type: str = "adam",
                 grad_clip: float = 10.0,
                 direction: str = "xy",
                 grad_tol: float = 0.0,
                 patience: int = 3,
                 lbfgs_history: int = 10,
                 max_step: float = 1.0,
                 stall_window: int = 0,
                 stall_ratio: float = 0.98):
        """
        新增终止准则参数:
        grad_tol : float
            梯度范数阈值。当 |grad|_norm < grad_tol 连续 patience 次时提前终止。
            设为 0 (默认) = 不启用, 仍按 iterations 跑满。
        patience : int
            连续满足阈值的次数, 避免单次波动误判。默认 3。
        max_step : float
            单步 CP 位移硬上限(像素)。当 |Δcps|_max > max_step 时按比例整体缩放,
            防止 Adam/L-BFGS 出现 v→0 或曲率异常时的位移爆炸。
            设为 0 或负数 = 关闭。默认 1.0。
        stall_window : int
            方案 A: 梯度停滞检测的滑动窗口大小 (迭代数)。
            当"最近 window 步的 |grad|_norm 最小值" 未低于
            "更早历史的最小值 × stall_ratio" 时视为停滞, 提前终止。
            适合梯度被噪声底困住的场景 (不需要预先知道噪声底大小)。
            设为 0 (默认) = 关闭该判据。建议 8~15。
        stall_ratio : float
            停滞判据的容忍系数, 0 < stall_ratio < 1。
            当 min_recent >= min_past · stall_ratio 视为停滞。
            默认 0.98 (最近窗口未比历史最优改善超过 2%)。
            越接近 1 越严格 (更容易触发早停), 越小越宽松。
        """
        self.simulator = simulator
        self.iterations = iterations
        self.lr = float(lr)
        self.optimizer_type = optimizer_type
        self.grad_clip = float(grad_clip)
        self.direction = direction
        self.grad_tol = float(grad_tol)
        self.patience = int(patience)
        self.lbfgs_history = int(lbfgs_history)
        self.max_step = float(max_step)
        self.stall_window = int(stall_window)
        self.stall_ratio = float(stall_ratio)

        self.target_mask = simulator.mask.data.copy()
        self.render = AntiAliasRenderer(msaa_level=16)
        self.parametric = ParametricDemo(simulator.meef.curve_type,
                                         self.target_mask)

        # wEPE 标量的定义权重: wEPE = Σ_j wepe_caculated_j · EPE_j
        # 求的就是这个标量 wEPE 对每个 CP (x, y) 的偏导


        self.wepe_weight = np.asarray(
            simulator.meef.wepe_caculated, dtype=float
        ).flatten()

    # -----------------------------------------------------------------
    # 中心差分求梯度: 复用 MEEF 矩阵构建 (MSAA + 并行线程)
    # -----------------------------------------------------------------

    def _compute_gradient(self, current_cps):
        """
        对每个 CP 的 (x, y) 方向做中心差分, 得到标量 wEPE 的梯度。

        wEPE = Σ_j wepe_caculated_j · EPE_j  是标量

        梯度:
            grad_x[i] = ∂wEPE/∂x_i = Σ_j wepe_caculated_j · ∂EPE_j/∂x_i
            grad_y[i] = ∂wEPE/∂y_i = Σ_j wepe_caculated_j · ∂EPE_j/∂y_i

        复用 _build_meef_matrix_xy_parallel_threads 做中心差分:
            Mx[j, i] = ∂EPE_j/∂x_i = (EPE_j(x_i+δ) - EPE_j(x_i-δ)) / (2δ)
            My[j, i] = ∂EPE_j/∂y_i

        所以:
            grad_x[i] = Σ_j wepe_caculated_j · Mx[j, i]
            grad_y[i] = Σ_j wepe_caculated_j · My[j, i]

        这等价于直接对标量 wEPE 做中心差分:
            grad_x[i] = (wEPE(x_i+δ) - wEPE(x_i-δ)) / (2δ)

        Returns
        -------
        grad_x : (num_cps,)
        grad_y : (num_cps,)
        """
        Mx, My = _build_meef_matrix_xy_parallel_threads(
            simulator=self.simulator,
            parametric=self.parametric,
            current_cps=current_cps,
            target_mask=self.target_mask,
            sraf_mask=self.simulator.meef.sraf_mask,
            eps=self.simulator.meef.eps,
            delta=self.simulator.meef.delta,
            render=self.render,
            curve_type=self.simulator.meef.curve_type,
        )

        # 标量 wEPE 对每个 CP 的梯度 = 加权雅可比沿 EP 点求和
        grad_x = (Mx * self.wepe_weight[:, np.newaxis]).sum(axis=0)
        grad_y = (My * self.wepe_weight[:, np.newaxis]).sum(axis=0)

        return grad_x, grad_y

    def _compute_gradient_bisector(self, current_cps, current_vectors):
        """
        沿角平分线方向做中心差分, 得到标量 wEPE 的梯度。

        wEPE = Σ_j wepe_calculated_j · EPE_j  是标量

        梯度 (每个 CP 一个标量, 沿法向位移 d):
            grad_d[i] = ∂wEPE/∂d_i = Σ_j wepe_calculated_j · M[j, i]

        复用 _build_meef_matrix_parallel_threads (bisector 版, 2N 次前向):
            M[j, i] = ∂EPE_j/∂d_i = (EPE_j(d_i+δ) - EPE_j(d_i-δ)) / (2δ)

        Returns
        -------
        grad_d : (num_cps,)  沿法向的标量梯度
        """
        M = _build_meef_matrix_parallel_threads(
            simulator=self.simulator,
            parametric=self.parametric,
            current_vectors=current_vectors,
            current_cps=current_cps,
            target_mask=self.target_mask,
            sraf_mask=self.simulator.meef.sraf_mask,
            eps=self.simulator.meef.eps,
            delta=self.simulator.meef.delta,
            render=self.render,
            curve_type=self.simulator.meef.curve_type,
            pixel_size=self.simulator.mask.pixel_size,
        )

        # 标量梯度 = 加权雅可比沿 EP 点求和
        grad_d = (M * self.wepe_weight[:, np.newaxis]).sum(axis=0)
        return grad_d

    # -----------------------------------------------------------------
    # 主入口
    # -----------------------------------------------------------------

    def run(self):
        sim = self.simulator
        save_dir = Path(sim.meef.filepath)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("文件保存路径为：", sim.meef.filepath)
        print(f"[gradient_descent] optimizer={self.optimizer_type}, "
              f"lr={self.lr}, grad_clip={self.grad_clip}, "
              f"direction={self.direction}")
        print(f"  渲染: MSAA-16x (AntiAliasRenderer)")
        print(f"  梯度: 中心差分 (delta={sim.meef.delta})")

        # ------------------------------------------------------------------
        # 1. LSM 初始仿真
        # ------------------------------------------------------------------
        current_mask = sim.meef.lsm_mask
        ai_lsm, wafer_lsm, _ = images_simulation_v2(
            mask_spatial=current_mask,
            resist_params=sim.params.resist,
            opt_cache=sim.opt_cache,
        )
        _save_images(ai_lsm, wafer_lsm, current_mask,
                     self.target_mask, sim.meef.filepath, "ls_mask")

        ls_epe, ls_epe_vector = caculate_epe(
            ai_lsm, sim.params.resist.threshold,
            sim.meef.eps, sim.meef.num_eps,
            sim.meef.weight_meef, add_weight=False,
        )
        ls_pe = float(np.sum((self.target_mask - wafer_lsm) ** 2))
        ls_wepe = float(caculate_wepe(sim.meef.wepe_caculated,
                                       ls_epe_vector, sim.meef.num_weps))
        np.savetxt(f"{self.simulator.meef.filepath}/target_mask.txt",self.target_mask,fmt='%d',delimiter=' ')
        print("M矩阵维数为(需要行大于列)：", sim.meef.num_eps, sim.meef.num_cps)
        print('#######################################################################')
        print(f"LSM后的平均wEPE误差：{ls_wepe / sim.meef.num_weps:.6f}, "
              f"EPE误差：{ls_epe / sim.meef.num_eps:.6f}, PE误差：{ls_pe:.6f}")

        # ------------------------------------------------------------------
        # 2. 初始化 CP + 优化器
        # ------------------------------------------------------------------
        current_cps = [np.array(c).copy() for c in sim.meef.cps]
        num_cps = sum(len(c) for c in current_cps)

        if self.optimizer_type == "adam":
            if self.direction == "bisector":
                adam_d = SimpleAdam(num_cps, lr=self.lr)
            else:
                adam_x = SimpleAdam(num_cps, lr=self.lr)
                adam_y = SimpleAdam(num_cps, lr=self.lr)
        elif self.optimizer_type == "lbfgs":
            if self.direction == "bisector":
                lbfgs_d = SimpleLBFGS(num_cps, lr=self.lr,
                                      history_size=self.lbfgs_history)
                # 累计位移 (作为 L-BFGS 内部的 x, 相对初始 CP)
                x_d = np.zeros(num_cps)
            else:
                lbfgs_x = SimpleLBFGS(num_cps, lr=self.lr,
                                      history_size=self.lbfgs_history)
                lbfgs_y = SimpleLBFGS(num_cps, lr=self.lr,
                                      history_size=self.lbfgs_history)
                x_x = np.zeros(num_cps)
                x_y = np.zeros(num_cps)

        # 初始评估
        current_mask = self.parametric.render_curve(current_cps) + sim.meef.sraf_mask
        ai_init, _wafer_init, _ = images_simulation_v2(
            mask_spatial=current_mask,
            resist_params=sim.params.resist, opt_cache=sim.opt_cache,
        )
        _initial_epe, _ = caculate_epe(
            ai_init, sim.params.resist.threshold,
            sim.meef.eps, sim.meef.num_eps,
            sim.meef.weight_meef, add_weight=False,
        )

        # 保存初始 CP 历史
        _save_cp_history(current_cps, 0, sim.meef.filepath)

        # 误差记录
        epe_errors = [ls_epe / sim.meef.num_eps]
        pe_errors = [ls_pe]
        wepe_errors = [ls_wepe / sim.meef.num_weps]
        iterations_log = [0]
        iteration_time = [0.0]

        # 最优记录
        best_wepe = np.inf
        best_wepe_epe = None
        best_wepe_mask = None
        best_wepe_wafer = None
        best_wepe_it = None
        best_wepe_cps = None

        best_epe = np.inf
        best_epe_wepe = None
        best_epe_mask = None
        best_epe_wafer = None
        best_epe_it = None
        best_epe_cps = None

        show_sample_mcps(self.target_mask, sim.meef.initial_mask,
                         sim.meef.cps, sim.meef.eps, sim.meef.filepath,
                         sim.meef.weight_meef)

        duration_total = 0.0

        # ------------------------------------------------------------------
        # 3. 主循环 (支持梯度终止准则)
        # ------------------------------------------------------------------
        below_tol_count = 0   # 连续满足 |grad|_norm < grad_tol 的次数
        grad_norm_history = []  # 每次迭代的 |grad|_norm, 用于停滞检测
        pbar = tqdm(range(1, self.iterations + 1),
                    desc="MEEF-GradDescent", ncols=100)
        for it in pbar:
            t0 = time.time()
            print(f"===== 第 {it} 次迭代开始 "
                  f"(optimizer={self.optimizer_type}, lr={self.lr}) =====")

            # ---- 中心差分求梯度 ----
            t_grad = time.time()
            if self.direction == "bisector":
                current_vectors = [get_cp_vectors(cps) for cps in current_cps]
                grad_d = self._compute_gradient_bisector(current_cps, current_vectors)
                t_grad = time.time() - t_grad

                # 梯度裁剪
                if self.grad_clip > 0:
                    norm = np.abs(grad_d)
                    scale = min(1.0, self.grad_clip / max(norm.max(), 1e-12))
                    grad_d = grad_d * scale

                grad_norm = float(np.sqrt((grad_d ** 2).sum())) # 梯度模长也就是二范数
                print(f"  [grad] |grad_d|_max={np.abs(grad_d).max():.4f}, "
                      f"|grad_d|_norm={grad_norm:.4f}, "
                      f"耗时={t_grad:.2f}s")

                # 梯度下降更新
                if self.optimizer_type == "adam":
                    delta_d = adam_d.step(grad_d)
                elif self.optimizer_type == "lbfgs":
                    delta_d = lbfgs_d.step(x_d, grad_d)
                else:
                    delta_d = -self.lr * grad_d

                # ---- 单步位移硬裁剪 (防 Adam v崩溃 / L-BFGS 曲率异常导致爆炸) ----
                if self.max_step > 0:
                    step_max = float(np.abs(delta_d).max())
                    if step_max > self.max_step:
                        scale = self.max_step / step_max
                        delta_d = delta_d * scale
                        print(f"  [clip] |Δd|_max={step_max:.3f} > max_step="
                              f"{self.max_step}, 缩放 {scale:.4f}")

                # L-BFGS 需要更新内部 x (基于裁剪后的实际位移, 保持 (s,y) 一致)
                if self.optimizer_type == "lbfgs":
                    x_d = x_d + delta_d

                delta_d = np.round(delta_d, 6)

                # 更新 CP (沿法向)
                new_vectors = [get_cp_vectors(cps) for cps in current_cps]
                new_cps = get_new_cps(current_cps, delta_d, new_vectors)

            else:
                # ---- xy 方向 ----
                grad_x, grad_y = self._compute_gradient(current_cps)
                t_grad = time.time() - t_grad

                # 梯度裁剪
                if self.grad_clip > 0:
                    norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
                    scale = min(1.0, self.grad_clip / max(norm.max(), 1e-12))
                    grad_x = grad_x * scale
                    grad_y = grad_y * scale

                grad_norm = float(np.sqrt((grad_x ** 2 + grad_y ** 2).sum()))
                print(f"  [grad] |grad_x|_max={np.abs(grad_x).max():.4f}, "
                      f"|grad_y|_max={np.abs(grad_y).max():.4f}, "
                      f"|grad|_norm={grad_norm:.4f}, "
                      f"耗时={t_grad:.2f}s")

                # 梯度下降更新
                if self.optimizer_type == "adam":
                    delta_x = adam_x.step(grad_x)
                    delta_y = adam_y.step(grad_y)
                elif self.optimizer_type == "lbfgs":
                    delta_x = lbfgs_x.step(x_x, grad_x)
                    delta_y = lbfgs_y.step(x_y, grad_y)
                else:
                    delta_x = -self.lr * grad_x
                    delta_y = -self.lr * grad_y

                # ---- 单步位移硬裁剪 (按每个 CP 的欧氏位移量) ----
                if self.max_step > 0:
                    disp = np.sqrt(delta_x ** 2 + delta_y ** 2)  # (N,)
                    step_max = float(disp.max())
                    if step_max > self.max_step:
                        scale = self.max_step / step_max
                        delta_x = delta_x * scale
                        delta_y = delta_y * scale
                        print(f"  [clip] |Δcps|_max={step_max:.3f} > max_step="
                              f"{self.max_step}, 缩放 {scale:.4f}")

                # L-BFGS 用裁剪后的实际位移更新内部累计 x
                if self.optimizer_type == "lbfgs":
                    x_x = x_x + delta_x
                    x_y = x_y + delta_y

                delta_x = np.round(delta_x, 6)
                delta_y = np.round(delta_y, 6)

                # 更新 CP
                new_cps = get_new_cps_xy(current_cps, delta_x, delta_y)

            # ---- numpy 权威评估 ----
            current_mask = (self.parametric.render_curve(new_cps)
                            + sim.meef.sraf_mask)
            ai_cur, wafer_cur, _ = images_simulation_v2(
                mask_spatial=current_mask,
                resist_params=sim.params.resist,
                opt_cache=sim.opt_cache,
            )
            cur_epe, cur_epe_vec = caculate_epe(
                ai_cur, sim.params.resist.threshold,
                sim.meef.eps, sim.meef.num_eps,
                sim.meef.weight_meef, add_weight=False,
            )
            cur_wepe = float(caculate_wepe(sim.meef.wepe_caculated,
                                            cur_epe_vec, sim.meef.num_weps))
            cur_pe = float(np.round(
                np.sum((self.target_mask - wafer_cur) ** 2), 6))

            duration_total += (time.time() - t0)
            iteration_time.append(int(duration_total))

            if self.direction == "bisector":
                print(f"迭代 {it} 次后：平均wEPE误差 = {cur_wepe / sim.meef.num_weps:.6f}, "
                      f"EPE误差 = {cur_epe / sim.meef.num_eps:.6f}, "
                      f"PE误差 = {cur_pe:.6f}, "
                      f"|Δd|_max = {np.abs(delta_d).max():.4f}")
                meef_placeholder = delta_d.reshape(-1, 1)
            else:
                print(f"迭代 {it} 次后：平均wEPE误差 = {cur_wepe / sim.meef.num_weps:.6f}, "
                      f"EPE误差 = {cur_epe / sim.meef.num_eps:.6f}, "
                      f"PE误差 = {cur_pe:.6f}, "
                      f"|Δx|_max = {np.abs(delta_x).max():.4f}, "
                      f"|Δy|_max = {np.abs(delta_y).max():.4f}")
                delta_d = np.round(np.sqrt(delta_x ** 2 + delta_y ** 2), 6)
                meef_placeholder = np.column_stack([delta_x, delta_y])
            save_iteration_results(sim.meef.filepath, current_mask,
                                   ai_cur, wafer_cur, delta_d,
                                   meef_placeholder, 'cividis',
                                   pixel_size=sim.mask.pixel_size)

            # 保存 CP 历史
            _save_cp_history(new_cps, it, sim.meef.filepath)

            epe_errors.append(cur_epe / sim.meef.num_eps)
            pe_errors.append(cur_pe)
            iterations_log.append(it)
            wepe_errors.append(cur_wepe / sim.meef.num_weps)

            # ---- 更新最优 ----
            if cur_wepe < best_wepe:
                best_wepe = cur_wepe
                best_wepe_epe = cur_epe
                best_wepe_mask = current_mask.copy()
                best_wepe_wafer = wafer_cur.copy()
                best_wepe_it = it
                best_wepe_cps = [c.copy() for c in new_cps]

            if cur_epe < best_epe:
                best_epe = cur_epe
                best_epe_wepe = cur_wepe
                best_epe_mask = current_mask.copy()
                best_epe_wafer = wafer_cur.copy()
                best_epe_it = it
                best_epe_cps = [c.copy() for c in new_cps]

            current_cps = new_cps

            # ---- 记录梯度范数历史 (用于两种收敛检测) ----
            grad_norm_history.append(grad_norm)

            # ---- 判据 1: 梯度范数阈值 ----
            if self.grad_tol > 0:
                if grad_norm < self.grad_tol:
                    below_tol_count += 1
                    print(f"  [convergence] |grad|_norm={grad_norm:.6f} < "
                          f"tol={self.grad_tol} ({below_tol_count}/{self.patience})")
                    if below_tol_count >= self.patience:
                        print(f"===== 提前终止: 连续 {self.patience} 次 "
                              f"|grad|_norm < {self.grad_tol}, 第 {it} 次迭代 =====")
                        pbar.close()
                        break
                else:
                    below_tol_count = 0

            # ---- 判据 2 (方案 A): 梯度停滞检测 (滑动窗口最小值) ----
            #   min(最近 window 步) >= min(更早历史) * stall_ratio  =>  停滞
            #   语义: 最近一个窗口的最优梯度都没能改善历史最优的 stall_ratio 倍
            if self.stall_window > 0 and len(grad_norm_history) >= 2 * self.stall_window:
                min_recent = min(grad_norm_history[-self.stall_window:])
                min_past = min(grad_norm_history[:-self.stall_window])
                if min_recent >= min_past * self.stall_ratio:
                    print(f"  [stall] 最近 {self.stall_window} 步 min|grad|={min_recent:.6f}, "
                          f"历史 min|grad|={min_past:.6f}, 比值={min_recent/max(min_past,1e-12):.4f} "
                          f">= {self.stall_ratio}")
                    print(f"===== 提前终止: 梯度停滞 (方案A), 第 {it} 次迭代 =====")
                    pbar.close()
                    break

        # ------------------------------------------------------------------
        # 4. 收尾保存
        # ------------------------------------------------------------------
        actual_iterations = it
        print(f"[summary] 实际迭代次数: {actual_iterations} / {self.iterations}")
        save_excel(sim.meef.up_filename, sim.meef.second_filename,
                   sim.meef.curve_type, sim.meef.filename,
                   iterations_log, pe_errors, epe_errors, wepe_errors,
                   iteration_time)
        print("===== MEEF (gradient descent) 优化完成 =====")
        drawcostcurve(epe_errors, iterations_log,
                      f"{sim.meef.filepath}/error/EPE_error.png",
                      "EPE_error", "EPE")
        drawcostcurve(wepe_errors, iterations_log,
                      f"{sim.meef.filepath}/error/wEPE_error.png",
                      "wEPE_error", "wEPE")
        drawcostcurve(pe_errors, iterations_log,
                      f"{sim.meef.filepath}/error/PE_error.png",
                      "PE_error", "PE")

        out_dir = Path(sim.meef.filepath)
        # wEPE 最优
        with open(out_dir / "best_wepe_result.txt", "w", encoding="utf-8") as f:
            best_wepe_epe_safe = best_wepe_epe if best_wepe_epe is not None else 0.0
            f.write(
                f"迭代过程中WEPE最优为第 {best_wepe_it} 次迭代\n"
                f"EPE误差为 {best_wepe_epe_safe / sim.meef.num_eps}\n"
                f"WEPE误差为 {best_wepe / sim.meef.num_weps}\n"
            )
        if best_wepe_cps is not None:
            with open(out_dir / "WEPE最优的控制点坐标.txt", "w") as f:
                for row in best_wepe_cps:
                    f.write(" ".join(map(str, row)) + "\n")
        ps = self.simulator.mask.pixel_size
        sraf_cps = getattr(sim.meef, "sraf_cps", None)

        if best_wepe_wafer is not None:
            # wafer: 带 nm 坐标轴 + colorbar (统一风格)
            save_matrix_nm(
                best_wepe_wafer,
                str(out_dir / "优化后WEPE最优wafer.png"),
                pixel_size=ps, cmap="cividis", dpi=600,
            )
            # 额外保留一张带红色目标轮廓的版本 (可选参考)
            highlight_contour_red(
                self.target_mask, best_wepe_wafer,
                save_path=str(out_dir / "优化后WEPE最优wafer_contour.png"),
            )
        if best_wepe_mask is not None:
            # mask (像素化): 带 nm 坐标轴 + colorbar
            save_matrix_nm(
                best_wepe_mask,
                str(out_dir / "优化后WEPE最优mask.png"),
                pixel_size=ps, cmap="cividis", dpi=600,
            )
        # 新增: 保存 wEPE 最优 CP 拟合成的 B-spline 曲线掩模
        if best_wepe_cps is not None:
            save_bspline_curve_mask(
                cps_list=best_wepe_cps,
                sraf_cps_list=sraf_cps,
                target_mask=self.target_mask,
                save_path=str(out_dir / "优化后WEPE最优bspline.png"),
                curve_type=sim.meef.curve_type,
                pixel_size=ps,
                title="wEPE optimal B-spline mask",
            )

        # EPE 最优
        with open(out_dir / "best_epe_result.txt", "w", encoding="utf-8") as f:
            best_epe_wepe_safe = best_epe_wepe if best_epe_wepe is not None else 0.0
            f.write(
                f"迭代过程中EPE最优为第 {best_epe_it} 次迭代\n"
                f"EPE误差为 {best_epe / sim.meef.num_eps}\n"
                f"WEPE误差为 {best_epe_wepe_safe / sim.meef.num_weps}\n"
                f"LSM优化后的WEPE误差为 {ls_wepe / sim.meef.num_weps}\n"
                f"EPE误差为 {ls_epe / sim.meef.num_eps}\n"
            )
        if best_epe_cps is not None:
            with open(out_dir / "EPE最优的控制点坐标.txt", "w") as f:
                for row in best_epe_cps:
                    f.write(" ".join(map(str, row)) + "\n")
        if best_epe_wafer is not None:
            save_matrix_nm(
                best_epe_wafer,
                str(out_dir / "优化后EPE最优wafer.png"),
                pixel_size=ps, cmap="cividis", dpi=600,
            )
            highlight_contour_red(
                self.target_mask, best_epe_wafer,
                save_path=str(out_dir / "优化后EPE最优wafer_contour.png"),
            )
        if best_epe_mask is not None:
            save_matrix_nm(
                best_epe_mask,
                str(out_dir / "优化后EPE最优mask.png"),
                pixel_size=ps, cmap="cividis", dpi=600,
            )
        if best_epe_cps is not None:
            save_bspline_curve_mask(
                cps_list=best_epe_cps,
                sraf_cps_list=sraf_cps,
                target_mask=self.target_mask,
                save_path=str(out_dir / "优化后EPE最优bspline.png"),
                curve_type=sim.meef.curve_type,
                pixel_size=ps,
                title="EPE optimal B-spline mask",
            )
