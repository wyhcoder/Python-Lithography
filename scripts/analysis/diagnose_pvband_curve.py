"""
PVBand & wEPE vs SRAF 等宽扫描诊断脚本 (一次性使用, 跑完即弃).

目的:
    第一次诊断已确认: PVBand 在 [1.0, 4.5] 内单调递增, baseline 反而最低,
    说明 PVBand 不奖励 SRAF.
    本次扩展: 同步扫描 wEPE_main, 验证 wEPE 是否对 SRAF 宽度敏感.

判定:
    - 若 wEPE 在某 w* 有内部最低点, 且该最低点显著低于 baseline:
      → SRAF 通过 EPE 通道有真实物理价值, 应改用联合 cost = α·wEPE + β·PVBand
    - 若 wEPE 也几乎平坦或同样单调递减:
      → 主图形已被 Stage1 优化到极限, 当前 SRAF 配置(数量/位置)对 EPE 无意义,
        需重新审视 SRAF 骨架/位置, 而不是改 cost.

输出:
    outputs/diagnostics/pvband_w_curve.png  PVBand-w 曲线
    outputs/diagnostics/cost_w_curves.png   双 y 轴: PVBand + wEPE 同图对照
    控制台数值表格.
"""
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from litho_model.simulation_parameters import SimulationParameters
from litho_model.lithography_simulator import LithographySimulator
from litho_model.demo_compute_image import images_simulation_v2
from op_model.demo_pv_band import PVBandComputer
from op_model.demo_sraf import SRAF_Optimizer
from op_model.demo_epe_wepe import caculate_epe, caculate_wepe
from utils_model.demo_sraf_utils import (
    distance_msaa_multi_curves,
    distance_msaa_multi_skeleton,
    read_cps_txt,
)


def diagnose(
    config_path: str = "litho_model/config.yaml",
    w_lo: float = 0.5,
    w_hi: float = 5.0,
    n_points: int = 31,
    out_png_pv: str = "outputs/diagnostics/pvband_w_curve.png",
    out_png_dual: str = "outputs/diagnostics/cost_w_curves.png",
):
    print(f"[1/4] 加载配置 + 构建 simulator from {config_path} ...")
    params = SimulationParameters.from_yaml(config_path)
    simulator = LithographySimulator(params)
    simulator.prepare_for_optimization(method="SOCS")

    print("[2/4] 构造 SRAF_Optimizer (复用其 parametric 渲染器 + pv_computer) ...")
    optimizer = SRAF_Optimizer(simulator=simulator)
    parametric = optimizer.parametric
    pv_computer: PVBandComputer = optimizer.pv_computer
    target_mask = optimizer.target_mask

    if not simulator.sraf.pvband_cost:
        raise RuntimeError(
            "config.yaml 中 sraf.pvband_cost = False, 该诊断仅适用 PVBand cost 模式."
        )
    raw_path = simulator.sraf.meef_opt_cps_path
    if not raw_path:
        raise FileNotFoundError(
            "config.yaml: sraf.meef_opt_cps_path 未配置, 无法获取主图形 cps."
        )
    cps_txt_path = Path(raw_path)
    if not cps_txt_path.is_file():
        raise FileNotFoundError(f"主图形 cps 文件不存在: {cps_txt_path}")
    base_cps = read_cps_txt(str(cps_txt_path))
    print(f"      主图形 cps loaded: {len(base_cps)} 组")

    # ---- baseline: 无 SRAF ----
    print("[3/4] 计算 baseline (无 SRAF) PVBand + wEPE ...")
    base_mask = parametric.render_curve(base_cps)
    pv_baseline, _ = pv_computer.compute(base_mask)

    ai_base, _, _ = images_simulation_v2(
        mask_spatial=base_mask,
        resist_params=simulator.params.resist,
        opt_cache=simulator.opt_cache,
    )
    epe_base, epe_vec_base = caculate_epe(
        ai_base, simulator.params.resist.threshold,
        simulator.sraf.eps, simulator.sraf.num_eps,
        simulator.sraf.weight_meef, add_weight=False,
    )
    wepe_base = caculate_wepe(
        simulator.sraf.wepe_caculated,
        epe_vec_base, simulator.sraf.num_weps,
    )
    wepe_base_avg = wepe_base / simulator.sraf.num_weps
    epe_base_avg = epe_base / simulator.sraf.num_eps
    print(f"      baseline PVBand = {pv_baseline:.4f}, "
          f"avg wEPE = {wepe_base_avg:.6f}, "
          f"avg EPE = {epe_base_avg:.6f}")

    # ---- 扫描 ----
    num_srafs = simulator.sraf.num_srafs
    use_simplify = simulator.sraf.sraf_simplify
    mask_shape = simulator.mask.data.shape
    print(f"[4/4] 等宽扫描 (PVBand + wEPE) "
          f"num_srafs={num_srafs}, simplify={use_simplify}, "
          f"w∈[{w_lo},{w_hi}], n={n_points} ...")

    ws = np.linspace(w_lo, w_hi, n_points)
    pvs = np.empty_like(ws)
    wepes = np.empty_like(ws)
    epes = np.empty_like(ws)
    t0 = time.time()
    for i, w in enumerate(ws):
        width_vec = np.full(num_srafs, float(w))
        if use_simplify:
            srafs = distance_msaa_multi_curves(
                simulator.sraf.fitted_curves_by_order,
                width_vec, mask_shape,
            )
        else:
            srafs = distance_msaa_multi_skeleton(
                simulator.sraf.srafs_skelet_layer,
                width_vec,
            )
        mask = parametric.render_curve(base_cps) + srafs

        pv, _ = pv_computer.compute(mask)
        ai, _, _ = images_simulation_v2(
            mask_spatial=mask,
            resist_params=simulator.params.resist,
            opt_cache=simulator.opt_cache,
        )
        epe_total, epe_vec = caculate_epe(
            ai, simulator.params.resist.threshold,
            simulator.sraf.eps, simulator.sraf.num_eps,
            simulator.sraf.weight_meef, add_weight=False,
        )
        wepe_total = caculate_wepe(
            simulator.sraf.wepe_caculated,
            epe_vec, simulator.sraf.num_weps,
        )

        pvs[i] = pv
        wepes[i] = wepe_total / simulator.sraf.num_weps
        epes[i] = epe_total / simulator.sraf.num_eps
        print(f"  [{i+1:>2}/{n_points}] w={w:.3f}  "
              f"PVBand={pv:.2f}  wEPE={wepes[i]:.6f}  EPE={epes[i]:.6f}")

    print(f"      扫描完成, 用时 {time.time() - t0:.1f}s")

    # ---- 输出 1: 单 PVBand 曲线 (兼容原图) ----
    out_pv = (PROJECT_ROOT / out_png_pv).resolve()
    out_pv.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
    ax.plot(ws, pvs, "o-", color="#3070c0",
            label="PVBand vs uniform SRAF width")
    ax.axhline(pv_baseline, color="#c04040", ls="--",
               label=f"baseline (no SRAF) = {pv_baseline:.2f}")
    ax.axvspan(1.0, 4.5, color="#cccccc", alpha=0.25,
               label="optimizer bounds [1.0, 4.5]")
    i_min = int(np.argmin(pvs))
    ax.scatter([ws[i_min]], [pvs[i_min]], s=120, c="#e08020", zorder=5,
               label=f"min @ w={ws[i_min]:.3f}, PVBand={pvs[i_min]:.2f}")
    ax.set_xlabel("uniform SRAF width  w  (all stages = w)")
    ax.set_ylabel("PVBand cost")
    ax.set_title("PVBand cost vs SRAF uniform width (diagnosis)")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_pv)
    plt.close(fig)
    print(f"\n[输出 1] PVBand-w 曲线: {out_pv}")

    # ---- 输出 2: 双 y 轴 PVBand + wEPE ----
    out_dual = (PROJECT_ROOT / out_png_dual).resolve()
    fig2, ax_pv = plt.subplots(figsize=(9, 5.5), dpi=110)
    color_pv = "#3070c0"
    color_we = "#c04060"

    # PVBand: 左轴
    ax_pv.plot(ws, pvs, "o-", color=color_pv, label="PVBand")
    ax_pv.axhline(pv_baseline, color=color_pv, ls=":", alpha=0.7,
                  label=f"PVBand baseline = {pv_baseline:.1f}")
    ax_pv.set_xlabel("uniform SRAF width  w  (all stages = w)")
    ax_pv.set_ylabel("PVBand cost", color=color_pv)
    ax_pv.tick_params(axis="y", labelcolor=color_pv)
    ax_pv.grid(True, ls=":", alpha=0.5)
    ax_pv.axvspan(1.0, 4.5, color="#cccccc", alpha=0.2)

    # wEPE: 右轴
    ax_we = ax_pv.twinx()
    ax_we.plot(ws, wepes, "s-", color=color_we, label="avg wEPE")
    ax_we.axhline(wepe_base_avg, color=color_we, ls=":", alpha=0.7,
                  label=f"wEPE baseline = {wepe_base_avg:.4f}")
    ax_we.set_ylabel("avg wEPE (per weighted EP)", color=color_we)
    ax_we.tick_params(axis="y", labelcolor=color_we)

    # 各自最低点
    iw_min = int(np.argmin(wepes))
    ax_we.scatter([ws[iw_min]], [wepes[iw_min]], s=120, c="#80c040",
                  zorder=5, edgecolors="black", linewidths=1.0,
                  label=f"wEPE min @ w={ws[iw_min]:.3f}")

    # 合并图例 (双轴)
    lines1, labels1 = ax_pv.get_legend_handles_labels()
    lines2, labels2 = ax_we.get_legend_handles_labels()
    ax_pv.legend(lines1 + lines2, labels1 + labels2,
                 loc="upper left", fontsize=8.5, framealpha=0.9)

    fig2.suptitle("PVBand & wEPE vs SRAF uniform width  "
                  "(does SRAF help on EPE channel?)",
                  fontsize=11)
    fig2.tight_layout()
    fig2.savefig(out_dual)
    plt.close(fig2)
    print(f"[输出 2] 双曲线对照图: {out_dual}")

    # ---- 结论 ----
    print("\n=== 诊断结论 ===")

    # PVBand 部分
    if i_min == 0:
        pv_msg = "PVBand 在 w_lo 取最低 (单调递减)"
    elif i_min == n_points - 1:
        pv_msg = "PVBand 在 w_hi 取最低 (单调递增)"
    else:
        pv_msg = f"PVBand 在 w*={ws[i_min]:.3f} 有内部最低点"
    pv_drop = (pv_baseline - pvs[i_min]) / max(pv_baseline, 1e-9) * 100
    print(f"  [PVBand] {pv_msg}, 相对 baseline 改善 {pv_drop:+.1f}%")

    # wEPE 部分
    if iw_min == 0:
        we_msg = "wEPE 在 w_lo 取最低 (单调递减)"
    elif iw_min == n_points - 1:
        we_msg = "wEPE 在 w_hi 取最低 (单调递增)"
    else:
        we_msg = f"wEPE 在 w*={ws[iw_min]:.3f} 有内部最低点 ★"
    we_drop = ((wepe_base_avg - wepes[iw_min])
               / max(wepe_base_avg, 1e-9) * 100)
    print(f"  [wEPE]   {we_msg}, 相对 baseline 改善 {we_drop:+.1f}%")

    print()
    if iw_min not in (0, n_points - 1) and we_drop > 5:
        print("  → SRAF 在 wEPE 通道有显著价值, 强烈推荐改用联合 cost = α·wEPE + β·PVBand")
        # 给个 α/β 数量级建议
        # 让 α·wEPE_base ≈ β·PVBand_base
        ratio = pv_baseline / max(wepe_base_avg, 1e-9)
        print(f"  推荐权重 (使两项 baseline 同量级): α/β ≈ {ratio:.0f}, "
              f"例如 α={ratio:.0f}, β=1")
    elif we_drop <= 5:
        print("  → SRAF 在 wEPE 上的改善 <5%, "
              "考虑 (a) Stage1 主图形未优到位; (b) SRAF 骨架/位置选错; "
              "(c) dose/defocus 采样太弱.")
    else:
        print("  → 结果不明显, 建议手工查看曲线.")


if __name__ == "__main__":
    diagnose()
