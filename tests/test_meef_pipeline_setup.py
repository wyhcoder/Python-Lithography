"""烟雾测试: 验证 MEEFPipelineSetup 能装配出与 set_meef.MEEF 鸭子兼容的对象,
并且 LSM 加载 / CP 提取 / SRAF 渲染 / initial_mask 拼接 都能跑通.

只测装配, 不跑光刻仿真和 MEEF 矩阵, 因此非常快.
运行:
    python -m tests.test_meef_pipeline_setup
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litho_model.simulation_parameters import SimulationParameters
from litho_model.load_mask_bmp import load_mask_image
from op_model.meef_pipeline import MEEFPipelineSetup
from op_model.meef_pipeline.setup import MEEFPipelineConfig


def main():
    yaml_path = ROOT / "litho_model" / "config.yaml"
    params = SimulationParameters.from_yaml(str(yaml_path))
    assert params.meef_pipeline is not None, "yaml 中需要 meef_pipeline 段"
    mp = params.meef_pipeline

    # 直接读 target_mask, 不构造完整 LithographySimulator (避免触发老 MEEF)
    target_mask = load_mask_image(params.mask.image_name)
    print(f"[setup] target_mask shape = {target_mask.shape}")

    cfg = MEEFPipelineConfig(
        pattern_name=mp.pattern_name,
        ls_mask_path=mp.ls_mask_path,
        file_name=mp.file_name + "_smoketest",
        move_strategy=mp.move_strategy,
        main_cp_interval=mp.main_cp_interval,
        main_symmetry=mp.main_symmetry,
        sraf_cp_arclen=mp.sraf_cp_arclen,
        sraf_min_cps=mp.sraf_min_cps,
        sraf_smoothing=mp.sraf_smoothing,
        sraf_curve_pts=mp.sraf_curve_pts,
        msaa_level=mp.msaa_level,
        curve_type=mp.curve_type,
        delta=mp.delta,
        dilate_radius=mp.dilate_radius,
        interval_line=mp.interval_line,
        interval_corner=mp.interval_corner,
        mid_weight=mp.mid_weight,
        other_weight=mp.other_weight,
    )

    setup = MEEFPipelineSetup(target_mask=target_mask, cfg=cfg)

    print(f"[setup] LSM mask shape  = {setup.lsm_mask.shape}")
    print(f"[setup] main CP groups  = {len(setup.cps)}, total = {setup.num_cps}")
    print(f"[setup] SRAF blocks     = {len(setup.sraf_cps)}, total CP = {setup.num_sraf_cps}")
    print(f"[setup] EP points       = {setup.num_eps}, weighted = {setup.num_weps}")
    print(f"[setup] initial_mask     min={setup.initial_mask.min():.3f}, max={setup.initial_mask.max():.3f}")
    print(f"[setup] sraf_mask        min={setup.sraf_mask.min():.3f}, max={setup.sraf_mask.max():.3f}")
    print(f"[setup] output dir      = {setup.filepath}")

    # ----- 可视化: target / LSM / main / sraf_mask / initial_mask + CP 散点 -----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=110)
    axes[0, 0].imshow(target_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("target_mask")
    axes[0, 1].imshow(setup.lsm_mask, cmap="gray")
    axes[0, 1].set_title("LSM mask (input)")
    axes[0, 2].imshow(setup.main_mask, cmap="gray")
    axes[0, 2].set_title("main_mask (split from LSM)")

    axes[1, 0].imshow(setup.sraf_mask, cmap="gray")
    axes[1, 0].set_title(f"sraf_mask (CP -> Bspline -> MSAA), {len(setup.sraf_cps)} blocks")
    # 把 SRAF CP 画到 sraf_mask 上
    for block in setup.sraf_cps:
        if len(block) > 0:
            axes[1, 0].scatter(block[:, 1], block[:, 0], s=5, c="red")

    axes[1, 1].imshow(setup.initial_mask, cmap="gray")
    axes[1, 1].set_title("initial_mask = main(CP) + sraf_mask")
    # 主图形 CP
    for block in setup.cps:
        block_arr = np.asarray(block)
        if len(block_arr) > 0:
            axes[1, 1].scatter(block_arr[:, 1], block_arr[:, 0], s=10, c="cyan")

    axes[1, 2].imshow(target_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title(f"EP points (n={setup.num_eps}, weighted={setup.num_weps})")
    if len(setup.eps) > 0:
        axes[1, 2].scatter(setup.eps[:, 1], setup.eps[:, 0], s=4, c="red")

    for ax in axes.flat:
        ax.set_axis_off()
    plt.tight_layout()

    out_png = Path(setup.filepath) / "smoketest_overview.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[setup] 概览图已保存: {out_png}")
    print("[setup] OK")


if __name__ == "__main__":
    main()
