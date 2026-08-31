"""
MSAA + 中心差分 + 梯度下降版 MEEF 优化入口脚本。

用法:
    python -m demo.scalar.case_meef_autograd

核心思路:
    1. MSAA 渲染 (与原版 demo_meef.py 一致, 不用 SDF)
    2. 中心差分求 wEPE 对 CP 的梯度
    3. 梯度下降 (Adam 或 GD) 更新 CP 位置

两种方向模式 (DIRECTION):
    "xy"       : X/Y 解耦, 每个 CP 有 (Δx, Δy) 两个梯度, 4N 次前向仿真
    "bisector" : 沿角平分线方向, 每个 CP 只有标量 Δd, 2N 次前向仿真 (更快)

可调超参:
    LR            : 学习率 (Adam 建议 0.05~0.5, GD 建议 0.005~0.02, LBFGS 建议 0.5~1.0)
    OPTIMIZER_TYPE: 'adam' / 'gd' / 'lbfgs'
    GRAD_CLIP     : 梯度范数裁剪上限 (0 = 不裁剪)
    DIRECTION     : 'xy' 或 'bisector'
    GRAD_TOL      : 梯度范数收敛阈值 (0 = 关闭, 跑满 iterations)
    PATIENCE      : 连续多少次满足阈值才终止, 避免单次波动误判
    LBFGS_HISTORY : L-BFGS 保留的历史 (s, y) 对数量, 建议 5~15
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from litho_model.simulation_parameters import SimulationParameters
from litho_model.lithography_simulator import LithographySimulator
from op_model.demo_meef_autograd.optimizer import MEEF_GradientDescentOptimizer
from utils_model.project_paths import SCALAR_CONFIG_PATH


# ============================== CONFIG ============================
LR = 0.5               # 学习率; Adam: 0.05~0.5, GD: 0.005~0.02, LBFGS: 0.5~1.0
OPTIMIZER_TYPE = "lbfgs"   # 'adam' / 'gd' / 'lbfgs'
GRAD_CLIP = 10.0        # 梯度范数裁剪上限 (0 = 不裁剪)
DIRECTION = "xy"  # 'xy' (X/Y 解耦) 或 'bisector' (角平分线)

# --- 梯度收敛终止准则 ---
GRAD_TOL = 0.05         # |grad|_norm < GRAD_TOL 时视为收敛, 0 = 关闭
PATIENCE = 3            # 连续 PATIENCE 次满足才终止

# --- L-BFGS 专用超参 (OPTIMIZER_TYPE="lbfgs" 时生效) ---
LBFGS_HISTORY = 10      # 保留的 (s, y) 对数量, 建议 5~15

# --- 单步位移硬裁剪 (防 Adam v崩溃 / L-BFGS 曲率异常导致爆炸) ---
MAX_STEP = 1.0          # 单步 CP 位移上限 (像素); 0 = 关闭

# --- 方案 A: 梯度停滞检测 (滑动窗口最小值) ---
# 当"最近 STALL_WINDOW 步的 min |grad|" >= "更早历史的 min |grad|" * STALL_RATIO
# 视为停滞, 提前终止. 适合"不知道噪声底具体值"的场景.
STALL_WINDOW = 10       # 滑动窗口大小; 0 = 关闭
STALL_RATIO = 0.98      # 停滞判据的容忍系数 (0<x<1), 越大越易触发
# ==================================================================


def main():
    # 步骤 1: 加载配置
    params = SimulationParameters.from_yaml(str(SCALAR_CONFIG_PATH))

    # 步骤 2: 创建仿真器
    litho_simulator = LithographySimulator(params)
    litho_simulator.prepare_for_optimization(method="SOCS")

    # 步骤 3: 决定迭代次数
    if litho_simulator.mask.pixel_size == 6:
        iterations = 50
    else:
        iterations = 60

    # 步骤 4: 创建梯度下降优化器并运行
    optimizer = MEEF_GradientDescentOptimizer(
        simulator=litho_simulator,
        iterations=iterations,
        lr=LR,
        optimizer_type=OPTIMIZER_TYPE,
        grad_clip=GRAD_CLIP,
        direction=DIRECTION,
        grad_tol=GRAD_TOL,
        patience=PATIENCE,
        lbfgs_history=LBFGS_HISTORY,
        max_step=MAX_STEP,
        stall_window=STALL_WINDOW,
        stall_ratio=STALL_RATIO,
    )
    optimizer.run()


if __name__ == "__main__":
    main()
