import numpy as np
import matplotlib.pyplot as plt
from .demo_pe_gradient import compute_mask_pe_gradient
from litho_model.lithography_simulator import LithographySimulator
from litho_model.demo_compute_image import images_simulation_v1, images_simulation_v2


class CTM_Optimizer:
    """
    使用 CTM 方法的优化器。
    """
    def __init__(self, simulator: LithographySimulator, iterations: int, learning_rate: float):
        """
        初始化优化器。

        Args:
            simulator: 配置好的 LithographySimulator 实例。
            iterations: 优化的迭代次数。
            learning_rate: 梯度下降的学习率。
        """
        self.simulator = simulator
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.target_mask = simulator.mask.data.copy() # 优化目标是原始的二值掩模
        self.error_history = []
    
    def _initialize_theta(self, tbias: float = 0.3) -> np.ndarray:
        """根据目标pattern初始化theta"""
        theta = self.target_mask.copy()
        theta[theta > 0] = tbias * np.pi
        theta[theta == 0] = (1 - tbias) * np.pi
        return theta
    
    def _plot_error_history(self):
        """误差随迭代次数的变化"""
        plt.figure()
        plt.plot(range(1, self.iterations + 1), self.error_history)
        plt.xlabel("Iteration")
        plt.ylabel("L2 Error")
        plt.title("Optimization Error vs. Iteration")
        plt.grid(True)
        plt.show()
        
    def run(self) -> np.ndarray:
        """
        执行完整的 CTM 优化流程。

        Returns:
            np.ndarray: 经过优化后的最终掩模版图。
        """
        # 1. 初始化 theta 和初始灰度掩模
        theta = self._initialize_theta()
        initial_gray_mask = 0.5 * (1 + np.cos(theta))
        self.simulator.update_mask(initial_gray_mask)
        
        # 2. 准备预计算缓存和追踪变量
        self.simulator.prepare_for_optimization()
        best_mask = initial_gray_mask
        min_error = float('inf')
        
         # 3. 开始优化循环
        for i in range(self.iterations):
            current_mask = self.simulator.get_current_mask_spatial()
            _, wafer_image, intermediates = images_simulation_v2(
                mask_spatial=current_mask,
                resist_params=self.simulator.params.resist,
                opt_cache=self.simulator.opt_cache
            )
            Gm, current_error = compute_mask_pe_gradient(
                wafer_image=wafer_image, 
                target_mask=self.target_mask, 
                iteration_intermediates=intermediates, 
                opt_cache=self.simulator.opt_cache, 
                resist_params=self.simulator.params.resist
            )
            if i == 0:
                plt.imshow(Gm)
                plt.colorbar()
                plt.title("Original gardient image")
                plt.show()
            self.error_history.append(current_error)
            
            Gt = -0.5 * np.sin(theta) * Gm
            # Gt = Gt.reshape((sp ** 2, 1))
            print(f"Iteration {i+1}/{self.iterations} | Current L2 Error: {current_error:.4f}")

            if current_error < min_error:
                min_error = current_error
                best_mask = current_mask.copy()

            theta -= self.learning_rate * Gt # 减去梯度
            new_op_mask = 0.5 * (1 + np.cos(theta))
            self.simulator.update_mask(new_op_mask)
            
        print("\n--- Optimization Finished ---")
        print(f"Minimum L2 Error achieved: {min_error:.4f}")
        
        self._plot_error_history()
        return best_mask