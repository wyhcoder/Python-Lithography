from .demo_level_set_methods.demo_evolve_kappa import evolve_kappa_gradient
from .demo_level_set_methods.demo_evolve_normal_WENO_godunov import evolve_normal_WENO_godunov
from .demo_level_set_methods.demo_reinit_SD_FMM import reinit_SD_FMM
from .demo_level_set_methods.demo_get_dt_normal_kappa import get_dt_normal_kappa

from litho_model.lithography_simulator import LithographySimulator
from litho_model.demo_compute_image import images_simulation_v2

from .demo_pe_gradient import compute_mask_pe_gradient


import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple


# class LevelSetOptimizer:
#     def __init__(self, simulator: LithographySimulator, dx=0.5, dy=0.5, cfl=0.5, b=0.0000001, ltr=0.0, iteration = 50, reinit_freq = 10):
#         self.simulator = simulator
#         self.dx = dx
#         self.dy = dy
#         self.cfl = cfl
#         self.b = b
#         self.ltr = ltr
#         self.dx2 = dx * dx
#         self.dy2 = dy * dy
#         self.iteration = iteration
#         self.reinit_freq = reinit_freq
        
#     def _monitor_sdf_quality(self, phi: np.ndarray, iteration: int):
#         """
#         调试函数， 监视SDF
#         """
#         phi_y, phi_x = np.gradient(phi, self.dy, self.dx)
#         gradient_norm = np.sqrt(phi_x**2 + phi_y**2)
#         mean_norm = np.mean(gradient_norm)
#         max_deviation = np.max(np.abs(gradient_norm - 1.0))
#         std_norm = np.std(gradient_norm)
#         print(f"\n--- SDF Quality at Iteration {iteration} ---")
#         print(f"  Mean gradient norm: {mean_norm:.4f} (ideal: 1.0)")
#         print(f"  Max deviation from 1: {max_deviation:.4f}")
#         print(f"  Standard deviation of norm: {std_norm:.4f}")
#         print("------------------------------------")
        
#     def _calculate_evolution_term(self, phi: np.ndarray, Gm: np.ndarray) -> Tuple[np.ndarray, float]:
#         """
#         辅助函数，用于计算单步的演化项 Normal 和时间步 dt。
#         """
#         # 调用我们最好的法向演化函数
#         delta_normal, H1_abs, H2_abs = evolve_normal_WENO_godunov(phi, self.dx, self.dy, Gm)
#         delta_kappa = evolve_kappa_gradient(phi, self.dx, self.dy, self.b)
#         dt = get_dt_normal_kappa(self.cfl, self.dx, self.dy, H1_abs, H2_abs, self.b)
#         Normal = delta_kappa - delta_normal
#         return Normal, dt
    
#     def run(self) -> np.ndarray:
#         # 读取含SRAFs版图
#         initmask = self.simulator.get_current_mask_spatial()
#         phi_n = reinit_SD_FMM((initmask - 0.5), self.dx, self.dy)
#         print("Initial SDF created.")
        
#         self.simulator.prepare_for_optimization()
#         min_error = float('inf')
        
#         for i in range(self.iterations):
#             # Gm
#             current_mask = self.simulator.get_current_mask_spatial()
#             _, wafer_image, intermediates = images_simulation_v2(
#                 mask_spatial=current_mask,
#                 resist_params=self.simulator.params.resist,
#                 opt_cache=self.simulator.opt_cache
#             )
#             Gm, current_error = compute_mask_pe_gradient(
#                 wafer_image=wafer_image, 
#                 target_mask=self.simulator.mask.data, 
#                 iteration_intermediates=intermediates, 
#                 opt_cache=self.simulator.opt_cache, 
#                 resist_params=self.simulator.params.resist
#             )
            
#             Normal_n, dt = self._calculate_evolution_term(phi_n, Gm)
#             phi_1 = phi_n + dt * Normal_n
#             Normal_1, _ = self._calculate_evolution_term(phi_1, Gm)
#             phi_n_plus_1 = 0.5 * phi_n + 0.5 * (phi_1 + dt * Normal_1)
#             phiM = phi_n_plus_1
            
#             if i % self.reinit_freq == 0:
#                 print(f"--- Performing periodic reinitialization (FMM) at iteration {i} ---")
#                 phiM = reinit_SD_FMM(phiM, self.dx, self.dy)
#             phi_n = phiM
            
#             self._monitor_sdf_quality(phiM, i)
#             temp = np.zeros_like(phiM)
#             temp[phiM >= self.ltr] = 1
#             # update mask
#             self.simulator.update_mask(temp)
            
#             print(f"Iteration {i+1}/{self.iteration} | Current L2 Error: {current_error:.4f}")

#             if current_error < min_error:
#                 min_error = current_error
#                 best_mask = self.simulator.update_mask(temp).copy()
            
#         return best_mask
    

from utils_model.plot_matrix import plot_matrix
class LevelSetOptimizer:
    def __init__(self, simulator: LithographySimulator, dx=0.5, dy=0.5, cfl=0.5, b=0.0000001, ltr=0.0):
        self.simulator = simulator
        self.dx = dx
        self.dy = dy
        self.cfl = cfl
        self.b = b
        self.ltr = ltr
        self.error_history = [] # 4. 添加误差记录
    
    def _monitor_sdf_quality(self, phi: np.ndarray, iteration: int):
        """
        调试函数， 监视SDF
        """
        phi_y, phi_x = np.gradient(phi, self.dy, self.dx)
        gradient_norm = np.sqrt(phi_x**2 + phi_y**2)
        mean_norm = np.mean(gradient_norm)
        max_deviation = np.max(np.abs(gradient_norm - 1.0))
        std_norm = np.std(gradient_norm)
        print(f"\n--- SDF Quality at Iteration {iteration} ---")
        print(f"  Mean gradient norm: {mean_norm:.4f} (ideal: 1.0)")
        print(f"  Max deviation from 1: {max_deviation:.4f}")
        print(f"  Standard deviation of norm: {std_norm:.4f}")
        print("------------------------------------")
        
    def _calculate_evolution_term(self, phi: np.ndarray, Gm: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        辅助函数，用于计算单步的演化项 Normal 和时间步 dt。
        """
        # 调用我们最好的法向演化函数
        # H1：|Vn*phi_x|，H2: |Vn*phi_x|
        delta_normal, H1_abs, H2_abs = evolve_normal_WENO_godunov(phi, self.dx, self.dy, Gm)
        print(f"H1_abs: {np.max(H1_abs)}, H2_abs: {np.max(H2_abs)}")
        delta_kappa = evolve_kappa_gradient(phi, self.dx, self.dy, self.b) #计算曲率正则项
        # CFL 条件就是为了确保信息传播的速度慢于网格的分辨率速度，让计算能“脚踏实地”地进行。
        dt = get_dt_normal_kappa(self.cfl, self.dx, self.dy, H1_abs, H2_abs, self.b)
        Normal = delta_kappa - delta_normal
        return Normal, dt
    
    def run(self, iterations: int = 50, reinit_freq: int = 10):
        target_mask = self.simulator.mask.data.copy()   # 固定目标版图

        initmask = self.simulator.get_current_mask_spatial()
        phi_n = reinit_SD_FMM((initmask - 0.5), self.dx, self.dy)

        plot_matrix(phi_n, "Initial SDF")
        print("Initial SDF created.")

        self.simulator.prepare_for_optimization()

        min_error = float('inf')
        best_mask = initmask.copy()
        best_wafer = None

        for i in range(1, iterations + 1):
            # ========== 1. 用当前 mask 求梯度 ==========
            current_mask = self.simulator.get_current_mask_spatial()

            _, wafer_image_old, intermediates_old = images_simulation_v2(
                mask_spatial=current_mask,
                resist_params=self.simulator.params.resist,
                opt_cache=self.simulator.opt_cache
            )

            Gm, old_error = compute_mask_pe_gradient(
                wafer_image=wafer_image_old,
                target_mask=target_mask,
                iteration_intermediates=intermediates_old,
                opt_cache=self.simulator.opt_cache,
                resist_params=self.simulator.params.resist
            )
            # plt.imshow(Gm)
            # plt.show()


            # ========== 2. 用梯度推进 level-set ==========
            Normal_n, dt = self._calculate_evolution_term(phi_n, Gm)
            phi_1 = phi_n + dt * Normal_n
            Normal_1, _ = self._calculate_evolution_term(phi_1, Gm)
            phi_n_plus_1 = 0.5 * phi_n + 0.5 * (phi_1 + dt * Normal_1)
            print(f"dt: {dt}, Normal_n_abs: {np.max(Normal_n)}, Normal_1: {np.max(Normal_1)}, diff: {np.max(phi_n_plus_1 - phi_n)}")
            phiM = phi_n_plus_1

            if i % reinit_freq == 0:
                print(f"--- Performing periodic reinitialization (FMM) at iteration {i} ---")
                phiM = reinit_SD_FMM(phiM, self.dx, self.dy)

            phi_n = phiM
            self._monitor_sdf_quality(phiM, i)

            # ========== 3. threshold 得到新 mask ==========
            temp = np.zeros_like(phiM)
            temp[phiM >= self.ltr] = 1

            # ========== 4. 重新计算 temp 对应的误差 ==========
            _, wafer_image_new, intermediates_new = images_simulation_v2(
                mask_spatial=temp,
                resist_params=self.simulator.params.resist,
                opt_cache=self.simulator.opt_cache
            )

            new_error = np.sum((target_mask - wafer_image_new)**2)

            # 这里记录的是 temp 的误差
            self.error_history.append(new_error)

            # 更新 simulator 中的 mask
            self.simulator.update_mask(temp)
            plt.imshow(temp)
            plt.show()
            print(f"Iteration {i}/{iterations} | Temp PE Error: {new_error:.4f}")


            if new_error < min_error:
                min_error = new_error
                best_mask = temp.copy()
                best_wafer = wafer_image_new.copy()

        self._plot_error_history()
        print(f"最小误差为 {min_error:.6f}")
        return best_mask, min_error
    
    def _plot_error_history(self):
        plt.figure()
        plt.plot(range(1, len(self.error_history) + 1), self.error_history)
        plt.xlabel("Iteration")
        plt.ylabel("PE Error")
        plt.title("Optimization Error vs. Iteration")
        plt.grid(True)
        plt.show()
                
