"""新 MEEF 管线入口 (与 case_meef.py 完全独立).

流程:
    1) 从 yaml 读 system / mask / optics / source / resist + meef_pipeline 段
    2) 用 LithographySimulator 装配光学/光源, 但旁路掉老的 set_meef.MEEF
       (用 stub 占位, 避免它读历史文件)
    3) 用 MEEFPipelineSetup 重新装配 simulator.meef:
         - LSM 按 yaml 路径加载
         - 主图形 / SRAF 都走 "CP -> 闭合 B-spline -> MSAA"
         - 优化变量仅主图形 CP, SRAF CP 全程冻结
    4) prepare_for_optimization(SOCS) -> 优化器 run()
       优化器可选:
         OPTIMIZER = "tsvd"       : MEEF_Optimizer (原版, 角平分线/xy + TSVD)
         OPTIMIZER = "gradient"   : MEEF_GradientDescentOptimizer (中心差分 + 梯度下降)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from litho_model.simulation_parameters import SimulationParameters
from op_model.meef_pipeline import MEEFPipelineSetup, tee_console_to_file
from op_model.meef_pipeline.setup import MEEFPipelineConfig
from utils_model.project_paths import SCALAR_CONFIG_PATH


# ============================== OPTIMIZER CONFIG ============================
# 优化器选择: "tsvd" (原版 MEEF矩阵+TSVD) 或 "gradient" (中心差分+梯度下降)
OPTIMIZER = "tsvd"

# gradient 模式专用超参 (OPTIMIZER="gradient" 时生效)
GRAD_LR = 0.5              # 学习率; Adam: 0.05~0.5, GD: 0.005~0.02, LBFGS: 0.5~1.0
GRAD_OPTIMIZER_TYPE = "lbfgs"  # 'adam' / 'gd' / 'lbfgs'
GRAD_LBFGS_HISTORY = 10     # L-BFGS 历史对数量 (仅 lbfgs 生效)
GRAD_MAX_STEP = 3.0         # 单步 CP 位移硬上限 (像素); 0 = 关闭
GRAD_STALL_WINDOW = 10      # 方案A: 梯度停滞检测滑动窗口 (迭代数); 0 = 关闭
GRAD_STALL_RATIO = 0.95     # 停滞容忍系数, 0.9~0.99 (越大越易触发)
GRAD_CLIP = 10.0            # 梯度裁剪上限 (0 = 不裁剪)
GRAD_DIRECTION = "xy"       # 'xy' 或 'bisector'
GRAD_TOL = 1.0             # 梯度范数收敛阈值 (0 = 关闭, 跑满 iterations)
GRAD_PATIENCE = 3           # 连续多少次满足阈值才终止

# tsvd 模式的终止准则 (OPTIMIZER="tsvd" 时生效)
TSVD_STEP_TOL = 0.05    # ‖Δd‖_max 阈值 (像素); 0 = 关闭
TSVD_PATIENCE = 3           # 连续满足次数
# ===========================================================================


# ---------------------------------------------------------------------------
# 旁路老 MEEF: 在 LithographySimulator.__init__ 阶段, 用一个最小化 stub
# 占位, 避免 set_meef.MEEF 触发它对历史文件 (sraf_width_opt 等) 的依赖.
# ---------------------------------------------------------------------------
class _MEEFStub:
    """占位 MEEF: 只持有 target_mask, 让 LithographySimulator 顺利构造完成.

    随后我们会把 simulator.meef 整体替换成 MEEFPipelineSetup.
    """
    def __init__(self, target_mask, **_kwargs):
        self.target_mask = target_mask


def build_simulator_with_pipeline(yaml_path: str):
    """读 yaml, 装配 simulator + 用 MEEFPipelineSetup 替换 simulator.meef."""
    params = SimulationParameters.from_yaml(yaml_path)
    if params.meef_pipeline is None:
        raise RuntimeError(
            "yaml 中缺少 meef_pipeline 段, 请参考 litho_model/config.yaml 的示例补充."
        )

    # 用 stub patch 掉老 MEEF, 让 LithographySimulator 的构造不去碰历史文件
    with patch("litho_model.lithography_simulator.MEEF", _MEEFStub):
        from litho_model.lithography_simulator import LithographySimulator
        simulator = LithographySimulator(params)

    # 用新管线装配 simulator.meef
    cfg = MEEFPipelineConfig(
        pattern_name=params.meef_pipeline.pattern_name,
        ls_mask_path=params.meef_pipeline.ls_mask_path,
        file_name=params.meef_pipeline.file_name,
        move_strategy = params.meef_pipeline.move_strategy,
        main_cp_interval=params.meef_pipeline.main_cp_interval,
        main_symmetry=params.meef_pipeline.main_symmetry,
        sraf_cp_arclen=params.meef_pipeline.sraf_cp_arclen,
        sraf_min_cps=params.meef_pipeline.sraf_min_cps,
        sraf_smoothing=params.meef_pipeline.sraf_smoothing,
        sraf_curve_pts=params.meef_pipeline.sraf_curve_pts,
        sraf_min_area=params.meef_pipeline.sraf_min_area,
        msaa_level=params.meef_pipeline.msaa_level,
        curve_type=params.meef_pipeline.curve_type,
        delta=params.meef_pipeline.delta,
        dilate_radius=params.meef_pipeline.dilate_radius,
        interval_line=params.meef_pipeline.interval_line,
        interval_corner=params.meef_pipeline.interval_corner,
        mid_weight=params.meef_pipeline.mid_weight,
        other_weight=params.meef_pipeline.other_weight,

    )
    simulator.meef = MEEFPipelineSetup(target_mask=simulator.mask.data, cfg=cfg)
    return simulator, params


def main():
    yaml_path = str(SCALAR_CONFIG_PATH)
    # 1. 先装配 (这一步耗时短, 失败也无需写日志)
    simulator, params = build_simulator_with_pipeline(yaml_path)

    # 2. 拿到输出目录后立即开 tee, 把后续所有 print/异常 traceback
    #    都同步落盘到 {filepath}/run.log; 屏幕仍正常显示, 不影响交互.
    log_path = Path(simulator.meef.filepath) / "run.log"
    with tee_console_to_file(log_path, header="MEEF Pipeline Run Log"):
        print(f"[log] 控制台输出同步保存至: {log_path}")
        print(f"[Pipeline] yaml file       = {yaml_path}")
        print(f"[Pipeline] pattern_name    = {params.meef_pipeline.pattern_name}")
        print(f"[Pipeline] ls_mask_path    = {params.meef_pipeline.ls_mask_path}")
        print(f"[Pipeline] curve_type      = {params.meef_pipeline.curve_type}, "
              f"delta={params.meef_pipeline.delta}, "
              f"iterations={params.meef_pipeline.iterations}")
        print(f"[Pipeline] main_symmetry   = {params.meef_pipeline.main_symmetry}, "
              f"main_cp_interval(k)={params.meef_pipeline.main_cp_interval}")
        print(f"[Pipeline] sraf_cp_arclen  = {params.meef_pipeline.sraf_cp_arclen}, "
              f"sraf_min_cps={params.meef_pipeline.sraf_min_cps}, "
              f"sraf_min_area={params.meef_pipeline.sraf_min_area}")
        print(f"[Pipeline] LSM mask shape  = {simulator.meef.lsm_mask.shape}")
        print(f"[Pipeline] main CP groups  = {len(simulator.meef.cps)}, "
              f"total = {simulator.meef.num_cps}")
        print(f"[Pipeline] SRAF blocks     = {len(simulator.meef.sraf_cps)}, "
              f"total CP = {simulator.meef.num_sraf_cps}, "
              f"raw contour pts = {simulator.meef.sraf_contour_points}")
        print(f"[Pipeline] EP points       = {simulator.meef.num_eps}, "
              f"weighted = {simulator.meef.num_weps}")
        print(f"[Pipeline] output dir      = {simulator.meef.filepath}")

        # 3. 预计算光学/光源相关常量
        simulator.prepare_for_optimization(method="SOCS")

        # 4. 选择优化器
        iterations = params.meef_pipeline.iterations

        if OPTIMIZER == "gradient":
            # ---- 梯度下降版 (中心差分求 wEPE 梯度 + Adam/GD) ----
            from op_model.demo_meef_autograd.optimizer import MEEF_GradientDescentOptimizer
            print(f"[Pipeline] optimizer = gradient_descent "
                  f"({GRAD_OPTIMIZER_TYPE}, lr={GRAD_LR}, clip={GRAD_CLIP})")
            optimizer = MEEF_GradientDescentOptimizer(
                simulator=simulator,
                iterations=iterations,
                lr=GRAD_LR,
                optimizer_type=GRAD_OPTIMIZER_TYPE,
                grad_clip=GRAD_CLIP,
                direction=params.meef_pipeline.move_strategy,
                grad_tol=GRAD_TOL,
                patience=GRAD_PATIENCE,
                lbfgs_history=GRAD_LBFGS_HISTORY,
                max_step=GRAD_MAX_STEP,
                stall_window=GRAD_STALL_WINDOW,
                stall_ratio=GRAD_STALL_RATIO,
            )
        else:
            # ---- 原版 TSVD (角平分线/xy + MEEF矩阵 + TSVD) ----
            from op_model.demo_meef import MEEF_Optimizer
            print(f"[Pipeline] optimizer = tsvd (MEEF_Optimizer)")
            optimizer = MEEF_Optimizer(
                simulator=simulator,
                iterations=iterations,
                step_tol=TSVD_STEP_TOL,
                patience=TSVD_PATIENCE,
            )

        optimizer.run()


if __name__ == "__main__":
    main()
