import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  
from litho_model.simulation_parameters import SimulationParameters
from litho_model.lithography_simulator import LithographySimulator
from litho_model.demo_compute_image import images_simulation_v2
from utils_model.plot_matrix import plot_matrix
from op_model.demo_pe_gradient import compute_mask_pe_gradient
from op_model.demo_pe_ctm import CTM_Optimizer
import matplotlib.pyplot as plt
from utils_model.project_paths import SCALAR_CONFIG_PATH

def main():
    """
    主执行函数
    """
    
    print("--- STAGE 1: INITIALIZING THE LITHOGRAPHY SYSTEM ---")
    
    # 步骤 1: 从配置文件加载参数
    params = SimulationParameters.from_yaml(str(SCALAR_CONFIG_PATH))
    
    # 步骤 2: 创建并初始化仿真器实例
    litho_simulator = LithographySimulator(params)

    # ---display Data ---
    
    litho_simulator.prepare_for_optimization()
    plot_matrix(litho_simulator.source.source_map,'Source')
    plot_matrix(litho_simulator.mask.data,'Target Patterns')
    plot_matrix(litho_simulator.optics.pupil_function, 'Puiple')
    # --- end ---
    
    
    # ---display images---
    initial_mask = litho_simulator.get_current_mask_spatial()
    
    aerial_image, wafer_image, intermediates = images_simulation_v2(
        mask_spatial=initial_mask,
        resist_params=litho_simulator.params.resist,
        opt_cache=litho_simulator.opt_cache
    )
    
    plot_matrix(aerial_image)
    plot_matrix(wafer_image)
    
    # compute pe
    Gm, pe_erro = compute_mask_pe_gradient(wafer_image, initial_mask, 
                                intermediates, 
                                litho_simulator.opt_cache, 
                                litho_simulator.params.resist)
    
    print(f"\nL2 Error: {pe_erro}")
    
    
    from op_model.demo_pv_band import PVBandComputer
    # PE Loss 通常是 L2 范数，数值可能很大（取决于像素数）；PV Loss 是 Sum，数值量级可能不同。如果有需求或许可以调整一下？都变成 L2 范数
    
    pv_computer = PVBandComputer(
                    simulator=litho_simulator,
                    dose_margin=0.05,      # ±5% 剂量容差
                    dof_range_nm = 200.0,    # ±50nm 离焦范围
                    dof_steps=3,           # 采样 -100, 0, +100
                    mode='full',
                    method="SOCS"            # Dose + Defocus 全模式 dose_only
                    )
    
    pv_loss, pv_map = pv_computer.compute(initial_mask)

    pv_map_visual = np.flipud(pv_map) 
    import matplotlib.lines as mlines
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(pv_map_visual, cmap='hot', alpha=0.8, origin='upper')
    
    plt.contour(litho_simulator.mask.data, levels=[0.5], colors='cyan', linewidths=2, origin='upper')
    
    plt.colorbar(im, label="Probabilistic Failure / Variability Width")
    
    target_line = mlines.Line2D([], [], color='cyan', label='Target Pattern')
    
    plt.legend(handles=[target_line], loc='upper right')
    

    plt.title(f"PV Band Heatmap (Loss={pv_loss:.4f})")
    
    plt.show()
    
    
    from op_model.demo_pv_gradient import PVLossWithGradient
    # @方法 1: 独立计算 PV Loss 和梯度
    pv_module = PVLossWithGradient(litho_simulator, mode='full')
    pv_loss, gradient, pv_map = pv_module.compute_with_gradient(initial_mask)
    # 1. 打印标量 Loss 值
    print(f"PV Band Loss (Sum): {pv_loss:.6f}")

    # 2. 打印数组统计信息 (检查梯度是否正常，有没有全0或爆炸)
    print(f"PV Map 统计: Max={np.max(pv_map):.4f}, Mean={np.mean(pv_map):.4f}")
    print(f"Gradient 统计: Max={np.max(np.abs(gradient)):.4e}, Min={np.min(gradient):.4e}")

    # 3. 可视化结果 (强烈推荐)
    plt.figure(figsize=(12, 5))

    # 显示 PV Band 热力图
    plt.subplot(1, 2, 1)
    plt.imshow(pv_map, cmap='hot', origin='upper')
    plt.title(f"PV Band Map\nLoss={pv_loss:.4f}")
    plt.colorbar(label='Variability')

    # 显示 梯度图
    plt.subplot(1, 2, 2)
    limit = np.max(np.abs(gradient))
    plt.imshow(gradient, cmap='seismic', origin='upper', vmin=-limit, vmax=limit)
    plt.title("PV Loss Gradient (dL/dM)")
    plt.colorbar(label='Gradient Magnitude')

    plt.tight_layout()
    plt.show()
    
    # @方法 2: 组合 PE + PV Loss (共享计算)
    result = pv_module.compute_combined_loss_and_gradient(initial_mask, initial_mask, pv_weight=1)
        
    # 1. 打印所有 Loss 值
    print("-" * 30)
    print(f"Total Loss : {result['loss_total']:.6f}")
    print(f"  - PE Loss: {result['loss_pe']:.6f}")
    print(f"  - PV Loss: {result['loss_pv']:.6f}")
    print("-" * 30)

    # 2. 打印梯度的统计信息
    grad_total = result['grad_total']
    print(f"总梯度统计 (Grad Total):")
    print(f"  Shape: {grad_total.shape}")
    print(f"  Range: [{np.min(grad_total):.4e}, {np.max(grad_total):.4e}]")
    print(f"  L2 Norm: {np.linalg.norm(grad_total):.4f}")

    # 3. 可视化对比 (PE 梯度 vs PV 梯度)
    plt.figure(figsize=(15, 5))

    # PE 梯度
    plt.subplot(1, 3, 1)
    lim_pe = np.max(np.abs(result['grad_pe']))
    plt.imshow(result['grad_pe'], cmap='seismic', origin='upper', vmin=-lim_pe, vmax=lim_pe)
    plt.title("PE Gradient (Fidelity)")
    plt.colorbar()

    # PV 梯度
    plt.subplot(1, 3, 2)
    lim_pv = np.max(np.abs(result['grad_pv']))
    plt.imshow(result['grad_pv'], cmap='seismic', origin='upper', vmin=-lim_pv, vmax=lim_pv)
    plt.title("PV Gradient (Stability)")
    plt.colorbar()

    # 总梯度
    plt.subplot(1, 3, 3)
    lim_tot = np.max(np.abs(result['grad_total']))
    plt.imshow(result['grad_total'], cmap='seismic', origin='upper', vmin=-lim_tot, vmax=lim_tot)
    plt.title(f"Total Gradient\n(PE + {1.0}*PV)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    
    # 红色：正梯度（增加像素值会增加 Loss，所以优化方向是减，即变黑）。

    # 蓝色：负梯度（增加像素值会减小 Loss，所以优化方向是加，即变白）。

    # 白色：零梯度（无需改变）。

    
if __name__ == "__main__":
    main()
