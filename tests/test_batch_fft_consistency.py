"""
一致性验证脚本：对比
  - 老路径 v2 (逐核 fftconvolve1)
  - 新路径 v3-fast (batch FFT, 用 H_k_fft_stack)
  - fallback 路径 v3-loop (临时清空 H_k_fft_stack 后的 v3，应与 v2 完全一致)

跑完会打印三条信息：
  [aerial] / [wafer]   v2 vs v3-fast 的最大绝对误差和相对误差
  [aerial] / [wafer]   v2 vs v3-loop 的最大绝对误差 (应严格为 0 或 ~1e-15)
  [meef-1col] 单列梯度 在 batch FFT / 逐核两路径下的差异

用法:
    python -m tests.test_batch_fft_consistency

不修改任何主流程代码。
"""

import os
import sys
import time

import numpy as np

# 让脚本在工程根目录下也能 import
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from litho_model.simulation_parameters import SimulationParameters  # noqa: E402
from litho_model.lithography_simulator import LithographySimulator  # noqa: E402
from litho_model.demo_compute_image import (  # noqa: E402
    images_simulation_v2,
    images_simulation_v3,
)


def _fmt_diff(a: np.ndarray, b: np.ndarray) -> str:
    """格式化两个数组的最大绝对/相对误差。"""
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    denom = float(np.max(np.abs(a))) + 1e-30
    max_rel = max_abs / denom
    return f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"


def main():
    # 1. 用工程默认配置初始化仿真器
    cfg_path = os.path.join(_ROOT, "litho_model", "config.yaml")
    print(f"--- Loading config: {cfg_path} ---")
    params = SimulationParameters.from_yaml(cfg_path)
    sim = LithographySimulator(params)
    sim.prepare_for_optimization(method="SOCS")

    assert "H_k_fft_stack" in sim.opt_cache, (
        "opt_cache 没有 H_k_fft_stack，说明 _build_H_k_fft_stack 没有生效；"
        "请确认 lithography_simulator.py 的改动已保存。"
    )

    # 2. 使用当前真实输入 mask。MEEF/SRAF 是可选优化组件，不能作为本测试前置条件。
    mask = sim.get_current_mask_spatial().astype(np.float64)
    print(f"--- mask shape={mask.shape}, dtype={mask.dtype}, "
          f"non-zero ratio={np.count_nonzero(mask) / mask.size:.4f} ---")

    # ============================================================
    # 验证 1: v2 (老) vs v3-fast (新 batch FFT)
    # ============================================================
    print("\n[Test 1] v2 (legacy) vs v3-fast (batch FFT) ...")
    t0 = time.time()
    ai_v2, wf_v2, _ = images_simulation_v2(mask, sim.params.resist, sim.opt_cache)
    t_v2 = time.time() - t0

    t0 = time.time()
    ai_v3f, wf_v3f, _ = images_simulation_v3(mask, sim.params.resist, sim.opt_cache)
    t_v3f = time.time() - t0

    print(f"  time:       v2={t_v2*1000:7.1f} ms   v3-fast={t_v3f*1000:7.1f} ms   "
          f"speedup={t_v2/max(t_v3f,1e-9):5.2f}x")
    print(f"  [aerial]    {_fmt_diff(ai_v2, ai_v3f)}")
    print(f"  [wafer ]    {_fmt_diff(wf_v2, wf_v3f)}")

    # ============================================================
    # 验证 2: v2 vs v3-loop (临时屏蔽 H_k_fft_stack, 走 v3 fallback)
    # 应严格一致 (因为代码路径完全相同)
    # ============================================================
    print("\n[Test 2] v2 (legacy) vs v3-loop (fallback) ...")
    H_stack_backup = sim.opt_cache.pop("H_k_fft_stack")  # 临时移除
    try:
        t0 = time.time()
        ai_v3l, wf_v3l, _ = images_simulation_v3(mask, sim.params.resist, sim.opt_cache)
        t_v3l = time.time() - t0
    finally:
        sim.opt_cache["H_k_fft_stack"] = H_stack_backup  # 还原

    print(f"  time:       v3-loop={t_v3l*1000:7.1f} ms")
    print(f"  [aerial]    {_fmt_diff(ai_v2, ai_v3l)}    (期望 ~0)")
    print(f"  [wafer ]    {_fmt_diff(wf_v2, wf_v3l)}    (期望 ~0)")

    # ============================================================
    # 验证 3: 模拟一次 MEEF "单列梯度" 的中心差分计算
    # 用一个微扰过的 mask 模拟 +delta / -delta 两次仿真,
    # 看 v2 与 v3-fast 计算出的 (ai_plus - ai_minus) / (2*delta) 是否一致
    # ============================================================
    print("\n[Test 3] MEEF 单列梯度模拟 (中心差分) ...")
    delta = 1e-3
    rng = np.random.default_rng(0)
    bump = rng.standard_normal(mask.shape) * 0.0
    # 在 mask 中心附近做一个像素级的微扰 (模拟控制点移动)
    cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    bump[cy, cx] = 1.0
    bump[cy + 1, cx] = 1.0

    mask_plus = mask + delta * bump
    mask_minus = mask - delta * bump

    # v2 路径
    ai_p_v2, _, _ = images_simulation_v2(mask_plus, sim.params.resist, sim.opt_cache)
    ai_m_v2, _, _ = images_simulation_v2(mask_minus, sim.params.resist, sim.opt_cache)
    grad_v2 = (ai_p_v2 - ai_m_v2) / (2.0 * delta)

    # v3-fast 路径
    ai_p_v3, _, _ = images_simulation_v3(mask_plus, sim.params.resist, sim.opt_cache)
    ai_m_v3, _, _ = images_simulation_v3(mask_minus, sim.params.resist, sim.opt_cache)
    grad_v3 = (ai_p_v3 - ai_m_v3) / (2.0 * delta)

    print(f"  [grad ]     {_fmt_diff(grad_v2, grad_v3)}")
    # round(.,6) 后是否一致 —— 这是 MEEF 矩阵实际写入的精度
    grad_v2_r = np.round(grad_v2, 6)
    grad_v3_r = np.round(grad_v3, 6)
    eq = np.array_equal(grad_v2_r, grad_v3_r)
    print(f"  round(.,6) 后两路径是否完全相等: {eq}")

    # ============================================================
    # 总结
    # ============================================================
    print("\n--- Summary ---")
    print("  通过标准:")
    print("    Test 1 [aerial] max_abs < 1e-9  -> 数学等价")
    print("    Test 2 [aerial] max_abs ~ 0     -> fallback 路径完全一致")
    print("    Test 3 round(.,6) 后完全相等    -> MEEF 矩阵将完全一致")


if __name__ == "__main__":
    main()
