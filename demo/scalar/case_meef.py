import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  
from matplotlib import pyplot as plt
from litho_model.simulation_parameters import SimulationParameters
from litho_model.lithography_simulator import LithographySimulator
from op_model.demo_meef import MEEF_Optimizer
from utils_model.demo_sdf_renderer import SDFRenderer
from utils_model.project_paths import SCALAR_CONFIG_PATH

# ============================================================================
# Renderer 切换开关 (方式 B: 不改业务代码, 在 main 里热替换 renderer)
#
#   USE_SDF_RENDERER = True  : 启用 SDF + sigmoid 渲染 (光滑梯度, MEEF 矩阵更密)
#   USE_SDF_RENDERER = False : 退回原本的 MSAA-16x 渲染 (保持原行为)
#
# SDF_BETA: sigmoid 软化宽度 (像素), 仅在 USE_SDF_RENDERER=True 时生效.
#   - 1.0  : 推荐起步值, 边界过渡 ~2 像素, 梯度平滑
#   - 0.5  : 边界更锐利 (~1 像素), 接近真实光刻边
#   - 2.0  : 较糊, 适合早期粗优化
# ============================================================================
USE_SDF_RENDERER = False
SDF_BETA = 0.25

# ---- TSVD 步长收敛终止准则 ----
# 当每次 TSVD 求得的 ‖Δd‖_max 连续 STEP_PATIENCE 次小于 STEP_TOL 时提前终止
# STEP_TOL = 0 表示关闭, 跑满 iterations 次
STEP_TOL = 0.02       # 单位: 像素; 建议 0.01~0.05
STEP_PATIENCE = 3


def main():
    
    #步骤1：从配置文件加载参数
    params = SimulationParameters.from_yaml(str(SCALAR_CONFIG_PATH))
    
    #步骤2：创建并初始化仿真器实例
    litho_simulator = LithographySimulator(params)
    
    # ---display Data ---
    
    litho_simulator.prepare_for_optimization(method="SOCS") #"Abbe"SOCS
    
    # 可视化输出
    # print(litho_simulator.meef.wepe_caculated)
    # plot_matrix(litho_simulator.source.source_map,'Source')
    # plot_matrix(litho_simulator.mask.data,'Target Patterns')
    # plot_matrix(litho_simulator.optics.pupil_function, 'Puiple')
    # plot_matrix(litho_simulator.meef.initial_mask, 'LSM')
    # plot_matrix(litho_simulator.meef.sraf_mask, 'SRAFs')
    
    # 创建优化器实例
    if litho_simulator.mask.pixel_size == 6:
        iterations = 50
    else:
        iterations = 50
    meef_optimizer = MEEF_Optimizer(
        simulator=litho_simulator,
        iterations=iterations,
        step_tol=STEP_TOL,
        patience=STEP_PATIENCE,
    )

    # ------------------------------------------------------------------
    # 热替换 renderer: MEEF worker 用 self.render, 主流程评估用
    # self.parametric.render, 必须保持同一个实例, 否则 MEEF 矩阵 (梯度)
    # 与评估 cost (wEPE/EPE/PE) 走两套不同的 renderer, 优化方向会失真.
    # ------------------------------------------------------------------
    if USE_SDF_RENDERER:
        sdf = SDFRenderer(beta=SDF_BETA)
        # SDFRenderer 与 AntiAliasRenderer 是鸭子类型兼容 (都暴露 .MSAA(...)),
        # 但不是同一个继承体系. 用 setattr 动态赋值绕开静态类型检查器,
        # 运行时与直接赋值完全等价.
        setattr(meef_optimizer, "render", sdf)
        setattr(meef_optimizer.parametric, "render", sdf)
        print(f"[Renderer] Using SDFRenderer(beta={SDF_BETA})")
    else:
        print("[Renderer] Using AntiAliasRenderer(msaa_level=16) (default)")

    meef_optimizer.run()
    
if __name__ == "__main__":
    main()
