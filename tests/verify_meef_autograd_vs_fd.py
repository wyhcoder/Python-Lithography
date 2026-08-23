"""
验证 MEEF autograd 实现的正确性: 与中心差分 (CFD) 数值结果对比.

设计:
    - 同一个前向函数: optimizer._epe_from_displacement(d, cps, vec)
      (SDF + torch, 关于 d 可微)
    - 对比两种 M[i, j] = ∂EPE_i / ∂d_j 的算法:
        (1) autograd: optimizer._build_meef_matrix_autograd  (forward-mode jacfwd)
        (2) CFD:     M_fd[i, j] = (epe_i(d_j=+ε) - epe_i(d_j=-ε)) / (2ε)
    - 同一前向, 唯一变量是「求导方式」, 控制变量纯粹.

判定:
    - 两者数值差应在 ε² 量级误差内 (~1e-4 @ ε=1e-3)
    - 关键统计: max|M_ag - M_fd|, 各列相对误差, 几何/方向一致性

输出:
    控制台数值表格 + outputs/diagnostics/meef_autograd_vs_fd.png 散点图
"""
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from litho_model.simulation_parameters import SimulationParameters
from litho_model.lithography_simulator import LithographySimulator
from op_model.demo_meef_autograd.optimizer import MEEF_AutogradOptimizer


def central_finite_diff_M(optimizer: MEEF_AutogradOptimizer,
                          cps_t: torch.Tensor,
                          vec_t: torch.Tensor,
                          eps: float = 1e-3,
                          n_cols: int = 0) -> np.ndarray:
    """对同一个 _epe_from_displacement 做中心差分, 返回 M_fd: (num_eps, n_cols).

    M_fd[i, j] = (epe_i(d_j=+eps) - epe_i(d_j=-eps)) / (2*eps)

    Args:
        n_cols: 0 表示算所有 N 列, >0 表示只算前 n_cols 列 (调试用, 节省时间).
    """
    N = cps_t.shape[0]
    cols = N if n_cols <= 0 else min(n_cols, N)
    # 先跑一次拿到 num_eps
    d_zero = torch.zeros(N, dtype=torch.float64, device=optimizer.device)
    with torch.no_grad():
        e0 = optimizer._epe_from_displacement(d_zero, cps_t, vec_t)
    num_eps = e0.shape[0]

    M_fd = np.empty((num_eps, cols), dtype=np.float64)
    print(f"[CFD] 计算 {cols} 列 (共 {N}) × 每列 2 次前向, "
          f"总 {2*cols} 次仿真...")
    t0 = time.time()
    for j in range(cols):
        d_plus = torch.zeros(N, dtype=torch.float64, device=optimizer.device)
        d_plus[j] = eps
        d_minus = torch.zeros(N, dtype=torch.float64, device=optimizer.device)
        d_minus[j] = -eps
        with torch.no_grad():
            e_plus = optimizer._epe_from_displacement(d_plus, cps_t, vec_t)
            e_minus = optimizer._epe_from_displacement(d_minus, cps_t, vec_t)
        M_fd[:, j] = ((e_plus - e_minus) / (2 * eps)).cpu().numpy()
        if cols >= 5 and (j + 1) % max(1, cols // 5) == 0:
            print(f"      [{j+1}/{cols}] 已用时 {time.time()-t0:.1f}s")
    print(f"[CFD] 完成, 总用时 {time.time()-t0:.1f}s")
    return M_fd


def main(
    config_path: str = "litho_model/config.yaml",
    eps: float = 1e-3,                        # CFD 步长
    n_cols_cfd: int = 10,                     # 仅对照前 N 列 (节省时间, 0=全列)
    n_cols_to_check: int = 8,                 # 详细打印前几列的对比
    out_png: str = "outputs/diagnostics/meef_autograd_vs_fd.png",
):
    print("[1/4] 加载 simulator + optimizer ...")
    params = SimulationParameters.from_yaml(config_path)
    simulator = LithographySimulator(params)
    simulator.prepare_for_optimization(method="SOCS")
    optimizer = MEEF_AutogradOptimizer(
        simulator=simulator,
        iterations=1,           # 不会真跑, 我们手动调内部函数
        beta=0.25,
        device="cpu",           # 验证用 CPU 即可, 与 cuda 数值无差异
        solver="gauss_newton",
    )

    # 拿到初始 cps + vec (与 optimizer.run() 内的逻辑一致)
    print("[2/4] 取初始 cps 与角平分外法向 vec ...")
    cps_np_list = simulator.meef.cps                  # list of (N_i, 2) ndarray
    cps_flat = np.concatenate([np.asarray(c, dtype=np.float64)
                                for c in cps_np_list], axis=0)
    cps_t = torch.from_numpy(cps_flat).to(
        device=optimizer.device, dtype=torch.float64,
    )

    # vec: 用 utils 里的 get_cp_vectors 生成 (与 run() 完全一致)
    from utils_model.demo_meef_utils import get_cp_vectors
    current_vectors = [get_cp_vectors(c) for c in cps_np_list]
    vec_list = []
    for vs in current_vectors:
        for v in vs:
            vec_list.append(np.asarray(v.vector, dtype=np.float64))
    vec_flat = np.stack(vec_list, axis=0)
    vec_t = torch.from_numpy(vec_flat).to(
        device=optimizer.device, dtype=torch.float64,
    )
    N_total = cps_flat.shape[0]
    print(f"      cps: {N_total} 个控制点 (来自 {len(cps_np_list)} 条曲线)")

    # ------ Autograd ------
    print("[3/4] autograd 计算 M (forward-mode jacfwd) ...")
    t0 = time.time()
    M_ag, e0_ag = optimizer._build_meef_matrix_autograd(cps_t, vec_t)
    t_ag = time.time() - t0
    num_eps = M_ag.shape[0]
    print(f"      M_ag shape={M_ag.shape}, e0 shape={e0_ag.shape}, "
          f"用时 {t_ag:.1f}s")

    # ------ CFD ------
    print(f"[4/4] 中心差分计算 M_fd (eps={eps}) ...")
    M_fd = central_finite_diff_M(optimizer, cps_t, vec_t, eps=eps)

    # ------ 比对 ------
    diff = M_ag - M_fd
    abs_diff = np.abs(diff)
    rel_denom = np.maximum(np.abs(M_fd), np.abs(M_ag))
    rel_diff = np.where(rel_denom > 1e-8, abs_diff / rel_denom, 0.0)

    print("\n" + "=" * 60)
    print("        AUTOGRAD vs CENTRAL FINITE DIFFERENCE")
    print("=" * 60)
    print(f"M shape:                     {M_ag.shape}")
    print(f"|M_ag|  范围:                [{np.abs(M_ag).min():.4e}, "
          f"{np.abs(M_ag).max():.4e}], mean={np.abs(M_ag).mean():.4e}")
    print(f"|M_fd|  范围:                [{np.abs(M_fd).min():.4e}, "
          f"{np.abs(M_fd).max():.4e}], mean={np.abs(M_fd).mean():.4e}")
    print(f"|M_ag - M_fd|  最大:        {abs_diff.max():.4e}")
    print(f"|M_ag - M_fd|  平均:        {abs_diff.mean():.4e}")
    print(f"相对误差 (>1e-8 项)  最大:  {rel_diff.max():.4e}")
    print(f"相对误差 (>1e-8 项)  平均:  {rel_diff.mean():.4e}")
    # cosine similarity (向量化整张矩阵)
    a = M_ag.ravel(); b = M_fd.ravel()
    cos_sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"全局 cosine similarity:      {cos_sim:.10f}  (1.0 = 完全一致)")
    # 列方向相对误差 (∥M_ag[:,j] - M_fd[:,j]∥ / ∥M_fd[:,j]∥)
    col_rel = np.linalg.norm(diff, axis=0) / (np.linalg.norm(M_fd, axis=0) + 1e-30)
    print(f"逐列 ‖ΔM_j‖ / ‖M_fd_j‖:     "
          f"max={col_rel.max():.4e}, mean={col_rel.mean():.4e}")

    print("\n--- 前几列详细比对 (前 5 个 EPE 行 × 前 N 列) ---")
    K = min(n_cols_to_check, N_total)
    print(f"列 j ↓, EPE 行 i →  (前 5 行)")
    header = "       " + "    ".join([f" i={i:2d}" for i in range(min(5, num_eps))])
    print(header)
    for j in range(K):
        ag_row = "  ".join([f"{M_ag[i, j]:+.3e}" for i in range(min(5, num_eps))])
        fd_row = "  ".join([f"{M_fd[i, j]:+.3e}" for i in range(min(5, num_eps))])
        print(f" j={j:2d}|AG: {ag_row}")
        print(f"     |FD: {fd_row}")
        print(f"     |ΔΔ: " + "  ".join(
            [f"{(M_ag[i, j]-M_fd[i, j]):+.1e}" for i in range(min(5, num_eps))]))

    # ------ 散点图 ------
    out_png_abs = (PROJECT_ROOT / out_png).resolve()
    out_png_abs.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=110)

    # 左: M_fd vs M_ag 散点
    ax = axes[0]
    ax.scatter(M_fd.ravel(), M_ag.ravel(), s=3, alpha=0.4,
               color="#3070c0", label=f"all entries ({M_ag.size})")
    lim = max(np.abs(M_fd).max(), np.abs(M_ag).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.0, label="y = x (perfect)")
    ax.set_xlabel("CFD  M_fd[i, j]")
    ax.set_ylabel("Autograd  M_ag[i, j]")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(f"Element-wise scatter\n"
                 f"cos_sim={cos_sim:.6f}, "
                 f"max|Δ|={abs_diff.max():.2e}",
                 fontsize=10)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    # 右: 误差直方图
    ax = axes[1]
    ax.hist(abs_diff.ravel(), bins=80, color="#c05050", alpha=0.85)
    ax.set_xlabel("|M_ag[i,j] - M_fd[i,j]|")
    ax.set_ylabel("count (log)")
    ax.set_yscale("log")
    ax.set_title(f"Abs error histogram\n"
                 f"max={abs_diff.max():.2e}, mean={abs_diff.mean():.2e}",
                 fontsize=10)
    ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle(f"MEEF matrix: autograd (forward-mode jacfwd) "
                 f"vs central finite difference (eps={eps})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png_abs)
    plt.close(fig)
    print(f"\n[图] {out_png_abs}")

    # 结论
    print("\n=== 结论 ===")
    if cos_sim > 0.9999 and rel_diff.max() < 1e-2:
        print("✅ Autograd 实现正确: cosine 相似度 > 0.9999, "
              "最大相对误差 < 1%, 与中心差分高度一致.")
    elif cos_sim > 0.99:
        print("⚠️  Autograd 大致正确但有偏差: cosine > 0.99 但相对误差较大. "
              "可能是 SDF 在某些 cps 附近梯度本身敏感, 检查异常列.")
    else:
        print("❌ 两者差异显著, autograd 实现有 bug 或前向函数不一致.")


if __name__ == "__main__":
    main()
