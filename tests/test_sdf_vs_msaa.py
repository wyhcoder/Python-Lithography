"""
对比测试：SDF + Sigmoid Renderer vs 现有 MSAA Renderer

目的
----
在切换 MEEF 渲染前，先量化两种渲染在以下三方面的差异：

1) **整体掩模差异**：同一组多边形分别用两个 renderer 渲染，看 max_abs / mean_abs 差异；
2) **单次渲染耗时**：判断 SDF 在你机器上是否够快（典型 30-200 ms）；
3) **亚像素扰动响应**：把控制点沿法向移动 delta=0.05/0.1/0.2 像素，看
   - MSAA 的 mask diff 有多少非零像素（量化噪声 1/16）
   - SDF  的 mask diff 是否光滑、是否随 delta 线性缩放
   这一步直接决定 MEEF 单列梯度的精度。

可视化产物 (默认输出到 outputs/diagnostics/sdf_vs_msaa_figs/)：
- ``01_full_masks.png``     : 三种 renderer 的整体 mask + 两两差异图
- ``02_boundary_zoom.png``  : 边界局部放大，看灰阶过渡形态
- ``03_perturb_diff.png``   : 不同 delta 下 mask(δ) - mask(-δ) 的热图

用法
----
    python -m tests.test_sdf_vs_msaa

不依赖整个 LithographySimulator，只 import 两个 renderer。
"""

import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import matplotlib
# 用非交互后端，避免在 SSH/无显示环境下卡住；本地有显示也仍然能看图（保存到文件）
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils_model.demo_MSAA import AntiAliasRenderer  # noqa: E402
from utils_model.demo_sdf_renderer import SDFRenderer  # noqa: E402


# -----------------------------------------------------------------------------
# 构造一个测试多边形（不依赖工程数据，可独立运行）
# -----------------------------------------------------------------------------
def make_test_polygon(cy: float, cx: float, r: float = 30.0, n: int = 32) -> np.ndarray:
    """生成一个 n 边正多边形，圆心 (cy, cx) 半径 r，返回 (N, 2) 的 [y, x]。"""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ys = cy + r * np.sin(theta)
    xs = cx + r * np.cos(theta)
    return np.stack([ys, xs], axis=-1)


def perturb_one_vertex(poly: np.ndarray, idx: int, delta: float,
                       direction: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """把 poly 的第 idx 个顶点沿 direction（默认径向外法线）移动 delta 像素。"""
    poly2 = poly.copy()
    if direction is None:
        # 用相邻三点的角平分（跟你 MEEF 里 get_cp_vectors 一致），简化：径向
        cy = poly[:, 0].mean()
        cx = poly[:, 1].mean()
        v = poly[idx] - np.array([cy, cx])
        v = v / (np.linalg.norm(v) + 1e-8)
    else:
        v = np.asarray(direction, dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-8)
    poly2[idx] = poly[idx] + delta * v
    return poly2


def fmt(arr: np.ndarray) -> str:
    return (f"min={arr.min():+.4f}  max={arr.max():+.4f}  "
            f"mean={arr.mean():+.4f}  nnz={int((arr != 0).sum())}/{arr.size}")


# =============================================================================
# 可视化函数（统一输出到 outputs/diagnostics/sdf_vs_msaa_figs/）
# =============================================================================

def _ensure_outdir() -> str:
    out = os.path.join(_ROOT, "outputs", "diagnostics", "sdf_vs_msaa_figs")
    os.makedirs(out, exist_ok=True)
    return out


def plot_full_masks(m_msaa: np.ndarray,
                    m_sdf05: np.ndarray,
                    m_sdf10: np.ndarray,
                    poly: np.ndarray,
                    save_path: str) -> None:
    """
    画三张整体 mask + 两两差异图，共 6 子图 (2x3 布局)。

    上排：MSAA / SDF β=0.5 / SDF β=1.0
    下排：MSAA-SDF05 / MSAA-SDF10 / SDF05-SDF10
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ---------- 上排：三种 renderer 的 mask ----------
    titles_top = ["MSAA-16x", "SDF (β=0.5)", "SDF (β=1.0)"]
    masks_top = [m_msaa, m_sdf05, m_sdf10]
    for ax, title, m in zip(axes[0], titles_top, masks_top):
        im = ax.imshow(m, cmap="gray", vmin=0, vmax=1)
        # 把多边形顶点画上去做参考
        ax.plot(poly[:, 1], poly[:, 0], 'r-', linewidth=0.6, alpha=0.7)
        ax.plot(poly[:, 1], poly[:, 0], 'r.', markersize=2)
        ax.set_title(f"{title}\n#nonzero = {int((m > 1e-6).sum())}")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---------- 下排：两两差异 ----------
    diffs = [
        ("MSAA - SDF(β=0.5)", m_msaa - m_sdf05),
        ("MSAA - SDF(β=1.0)", m_msaa - m_sdf10),
        ("SDF(β=0.5) - SDF(β=1.0)", m_sdf05 - m_sdf10),
    ]
    for ax, (title, d) in zip(axes[1], diffs):
        vmax = max(abs(d.min()), abs(d.max()), 1e-3)
        im = ax.imshow(d, cmap="seismic", vmin=-vmax, vmax=+vmax)
        ax.set_title(f"{title}\nmax|diff|={vmax:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Full-mask comparison: MSAA vs SDF", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_boundary_zoom(m_msaa: np.ndarray,
                       m_sdf05: np.ndarray,
                       m_sdf10: np.ndarray,
                       poly: np.ndarray,
                       save_path: str,
                       half: int = 24) -> None:
    """
    在多边形某一段边界附近做局部放大，直观比较灰阶过渡形态。
    half: 放大窗口的半边长（像素）
    """
    # 选第 0 个顶点附近做放大
    cy, cx = int(round(poly[0, 0])), int(round(poly[0, 1]))
    H, W = m_msaa.shape
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))

    # 上排：放大后的 mask 灰度
    masks = [
        ("MSAA-16x", m_msaa[y0:y1, x0:x1]),
        ("SDF (β=0.5)", m_sdf05[y0:y1, x0:x1]),
        ("SDF (β=1.0)", m_sdf10[y0:y1, x0:x1]),
    ]
    for ax, (title, sub) in zip(axes[0], masks):
        im = ax.imshow(sub, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{title}\nunique values = {len(np.unique(sub))}")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 下排：沿同一水平扫线的灰度截面（穿过顶点）
    row_idx = (y1 - y0) // 2
    xs = np.arange(x0, x1)
    for ax, (title, sub) in zip(axes[1], masks):
        ax.plot(xs, sub[row_idx, :], '-o', markersize=3,
                label=title, linewidth=1.2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("pixel x")
        ax.set_ylabel("coverage")
        ax.set_title(f"{title}: 横向灰阶截面")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Boundary zoom around vertex #0  (y={cy}, x={cx})",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_perturb_diff(poly: np.ndarray,
                      canvas: np.ndarray,
                      renderers: list,
                      idx: int,
                      deltas: list,
                      save_path: str,
                      half: int = 24) -> None:
    """
    画 mask(δ) - mask(-δ) 在不同 δ 下的差异热图，每个 renderer 一行，每个 δ 一列。

    这张图直接呈现"渲染对亚像素扰动有多敏感"——MSAA 行通常是稀疏几个亮点，
    SDF 行则是连续的 +/- 条带，正是 MEEF 单列梯度想要的形状。
    """
    n_rows = len(renderers)
    n_cols = len(deltas)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 3.0 * n_rows),
                             squeeze=False)

    # 选放大窗口（同 plot_boundary_zoom 的设定）
    cy, cx = int(round(poly[idx, 0])), int(round(poly[idx, 1]))
    H, W = canvas.shape
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)

    for i, (name, renderer) in enumerate(renderers):
        for j, d in enumerate(deltas):
            poly_pos = perturb_one_vertex(poly, idx, +d)
            poly_neg = perturb_one_vertex(poly, idx, -d)
            m_pos = renderer.MSAA([poly_pos], canvas, type='gray')
            m_neg = renderer.MSAA([poly_neg], canvas, type='gray')
            diff = m_pos - m_neg

            sub = diff[y0:y1, x0:x1]
            vmax = max(abs(sub.min()), abs(sub.max()), 1e-3)

            ax = axes[i, j]
            im = ax.imshow(sub, cmap="seismic",
                           vmin=-vmax, vmax=+vmax, interpolation="nearest")
            nnz = int((np.abs(sub) > 1e-6).sum())
            ax.set_title(f"{name}\nδ={d}  nnz={nnz}  max|d|={vmax:.3f}",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Perturbation response: mask(+δ) - mask(-δ), vertex #{idx}",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    H, W = 256, 256
    canvas = np.zeros((H, W), dtype=np.float32)

    # 一个简单的测试多边形（圆形，32 个顶点）
    poly = make_test_polygon(cy=128, cx=128, r=40.0, n=32)
    polys = [poly]

    # 渲染器
    msaa = AntiAliasRenderer(msaa_level=16)
    sdf_b05 = SDFRenderer(beta=0.5)
    sdf_b10 = SDFRenderer(beta=1.0)

    # ----------------------------------------------------------------
    # Test 1: 整体掩模差异 + 耗时
    # ----------------------------------------------------------------
    print("=" * 72)
    print("[Test 1] 整体掩模差异 + 单次渲染耗时")
    print("=" * 72)

    t0 = time.time()
    m_msaa = msaa.MSAA(polys, canvas, type='gray')
    t_msaa = (time.time() - t0) * 1000

    t0 = time.time()
    m_sdf05 = sdf_b05.MSAA(polys, canvas, type='gray')
    t_sdf05 = (time.time() - t0) * 1000

    t0 = time.time()
    m_sdf10 = sdf_b10.MSAA(polys, canvas, type='gray')
    t_sdf10 = (time.time() - t0) * 1000

    print(f"  渲染耗时 (ms):  MSAA-16x={t_msaa:6.1f}    "
          f"SDF β=0.5={t_sdf05:6.1f}    SDF β=1.0={t_sdf10:6.1f}")
    print(f"  掩模总像素:    {H*W}")
    print(f"  MSAA  非零像素: {int((m_msaa > 1e-6).sum())}")
    print(f"  SDF05 非零像素: {int((m_sdf05 > 1e-6).sum())}")
    print(f"  SDF10 非零像素: {int((m_sdf10 > 1e-6).sum())}    "
          f"(β 越大边界过渡越宽，非零像素更多)")

    diff_05 = m_sdf05 - m_msaa
    diff_10 = m_sdf10 - m_msaa
    print(f"\n  MSAA vs SDF β=0.5  diff: {fmt(diff_05)}")
    print(f"  MSAA vs SDF β=1.0  diff: {fmt(diff_10)}")
    print("  备注: 两者本就有不同的边界软化策略, 大约 0.05~0.2 的差异属正常.")

    # ----------------------------------------------------------------
    # Test 2: 亚像素扰动响应 (这是 MEEF 真正关心的)
    # ----------------------------------------------------------------
    print()
    print("=" * 72)
    print("[Test 2] 单顶点亚像素扰动 → 渲染差异 (MEEF 单列梯度的核心)")
    print("=" * 72)

    deltas = [0.05, 0.1, 0.2, 0.5, 1.0]
    idx = 0  # 扰动第 0 个顶点 (径向外推)

    print(f"\n  方案: 把多边形顶点 #{idx} 沿径向外法线移动 ±delta, "
          f"看 mask(δ) - mask(-δ) 的范数与稀疏度.")
    print(f"  理想情况下: |mask_diff| 应该正比于 δ, 且非零像素数稳定 (代表线性化区).")
    print()
    print(f"  {'delta':>6}  {'renderer':>14}  {'L2 norm':>10}  "
          f"{'L1 norm':>10}  {'#nonzero':>9}  {'L1/δ':>10}")
    print("  " + "-" * 70)

    for renderer_name, renderer in [("MSAA-16x", msaa),
                                     ("SDF β=0.5", sdf_b05),
                                     ("SDF β=1.0", sdf_b10)]:
        for d in deltas:
            poly_pos = perturb_one_vertex(poly, idx, +d)
            poly_neg = perturb_one_vertex(poly, idx, -d)

            m_pos = renderer.MSAA([poly_pos], canvas, type='gray')
            m_neg = renderer.MSAA([poly_neg], canvas, type='gray')
            diff = m_pos - m_neg

            l2 = float(np.linalg.norm(diff))
            l1 = float(np.abs(diff).sum())
            nnz = int((np.abs(diff) > 1e-6).sum())
            ratio = l1 / max(d, 1e-9)

            print(f"  {d:>6.2f}  {renderer_name:>14}  "
                  f"{l2:>10.4f}  {l1:>10.4f}  {nnz:>9d}  {ratio:>10.4f}")
        print()

    # ----------------------------------------------------------------
    # 总结提示
    # ----------------------------------------------------------------
    print("=" * 72)
    print("[Summary] 怎么看上面这张表")
    print("=" * 72)
    print("""
  对每个 renderer:
   - L1/δ 这一列在不同 δ 下应当**接近常数**(说明是线性响应 = 真实梯度).
   - MSAA 在 δ < 0.2 时 L1/δ 通常波动很大 → 量化噪声主导;
     在 δ ≈ 0 时甚至可能出现 L1=0 (mask 完全没变).
   - SDF (β=0.5 或 1.0) 在所有 δ 下 L1/δ 应当稳定且接近.
   - SDF 的 #nonzero 一般比 MSAA 大几倍, 说明梯度信号更密集
     (MEEF 矩阵列将变得更密, np.round(.,6) 就不会再把信号截没).

  如果你看到 SDF 这一组数据在 δ=0.05 时仍然给出稳定的 L1/δ,
  说明可以放心地把 MEEF 的 delta 调小一档 (比如从 0.5 nm 降到 0.1 nm),
  在不增加迭代次数的前提下提高优化精度.
""")

    # ----------------------------------------------------------------
    # Test 3: 可视化 (生成 PNG 图)
    # ----------------------------------------------------------------
    print()
    print("=" * 72)
    print("[Test 3] 生成可视化对比图")
    print("=" * 72)
    outdir = _ensure_outdir()
    print(f"  输出目录: {outdir}")

    plot_full_masks(
        m_msaa, m_sdf05, m_sdf10, poly,
        save_path=os.path.join(outdir, "01_full_masks.png"),
    )

    plot_boundary_zoom(
        m_msaa, m_sdf05, m_sdf10, poly,
        save_path=os.path.join(outdir, "02_boundary_zoom.png"),
        half=24,
    )

    plot_perturb_diff(
        poly, canvas,
        renderers=[("MSAA-16x", msaa),
                   ("SDF β=0.5", sdf_b05),
                   ("SDF β=1.0", sdf_b10)],
        idx=0,
        deltas=[0.05, 0.1, 0.2, 0.5, 1.0],
        save_path=os.path.join(outdir, "03_perturb_diff.png"),
        half=24,
    )

    print()
    print(f"  全部图已写入: {outdir}/")
    print("  推荐查看顺序: 01_full_masks.png  →  02_boundary_zoom.png  →  03_perturb_diff.png")


if __name__ == "__main__":
    main()
