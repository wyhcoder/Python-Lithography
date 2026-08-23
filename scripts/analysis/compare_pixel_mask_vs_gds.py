"""对比"像素化版图 (CP→Bspline→MSAA)" vs "GDS 文件 rasterize" 的成像差异.

差异来源:
    ① B-spline 离散点数: 管线内部 200 点  vs  GDS 内主图形 400 点 / SRAF 200 点
    ② 抗锯齿方式:        管线 MSAA 16x (灰度 0~1) vs GDS rasterize (这里也用同一 MSAA, 公平对比)
    ③ GDS 1 nm 精度截断: 坐标被 round 到 nm 网格, 像素位置最多 0.5nm = 0.083 像素偏移

本脚本输出:
    - mask 像素 max abs diff / mean abs diff
    - aerial image max abs diff / 相对误差
    - wafer (resist 二值化后) IoU
    - EPE / wEPE 数值对比
    - 一张 4 子图的总览 png

只读结果, 不重跑优化.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import gdstk
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from litho_model.simulation_parameters import SimulationParameters
from litho_model.load_mask_bmp import load_mask_image
from litho_model.demo_compute_image import images_simulation_v2
from utils_model.demo_MSAA import AntiAliasRenderer
from op_model.demo_epe_wepe import caculate_epe


# ---------------- 配置 ----------------
RESULT_DIR = ROOT / "outputs/opc" / "MEEF_pipeline" / "BS" / \
    "matrix_0011_k=5_sym=none_srafArc=5.0_BS_pipeline_test"
GDS_PATH = RESULT_DIR / "optimized_mask.gds"
PIXEL_MASK_TXT_DIR = RESULT_DIR / "gray_mask_txt"   # 优化器每轮 dump 的灰度 mask
PIXEL_SIZE_NM = 6.0
H = W = 256


def rasterize_gds_with_msaa(gds_path: Path, H: int, W: int,
                            pixel_size_nm: float,
                            msaa_level: int = 16) -> np.ndarray:
    """读 GDS 多边形, 用与管线一致的 MSAA 16x rasterize 成灰度 mask.

    像素 -> GDS 时做了 y 翻转, 这里反过来; 多个 polygon 灰度取 max (与管线一致).
    """
    lib = gdstk.read_gds(str(gds_path))
    cell = lib.top_level()[0]
    pixel_size_um = pixel_size_nm * 1e-3

    polygons_yx = []
    for poly in cell.polygons:
        pts_um = poly.points  # (N, 2)  [x_um, y_um]
        x_pix = pts_um[:, 0] / pixel_size_um
        y_pix = H - pts_um[:, 1] / pixel_size_um
        polygons_yx.append(np.stack([y_pix, x_pix], axis=1))

    template = np.zeros((H, W), dtype=np.float32)
    renderer = AntiAliasRenderer(msaa_level=msaa_level)
    return renderer.MSAA(polygons_yx, template, type="gray")


def find_pixel_mask() -> np.ndarray:
    """找像素化版图的 ground truth: 优先用 优化后WEPE最优mask.png 对应的 txt;
    否则取 gray_mask_txt 里最后一帧."""
    cand1 = RESULT_DIR / "WEPE最优mask.txt"
    if cand1.exists():
        return np.loadtxt(cand1)
    if PIXEL_MASK_TXT_DIR.exists():
        txts = sorted(PIXEL_MASK_TXT_DIR.glob("*.txt"))
        if txts:
            print(f"[info] use last gray_mask: {txts[-1].name}")
            return np.loadtxt(txts[-1])
    raise FileNotFoundError("找不到像素化版图 mask 的 txt, 请确认优化结果目录")


def main():
    yaml_path = ROOT / "litho_model" / "config.yaml"
    params = SimulationParameters.from_yaml(str(yaml_path))

    # ---------- 1. 装配 simulator (旁路老 MEEF) ----------
    from unittest.mock import patch

    class _MEEFStub:
        def __init__(self, target_mask, **_):
            self.target_mask = target_mask

    with patch("litho_model.lithography_simulator.MEEF", _MEEFStub):
        from litho_model.lithography_simulator import LithographySimulator
        simulator = LithographySimulator(params)
    simulator.prepare_for_optimization(method="SOCS")
    target_mask = simulator.mask.data

    # ---------- 2. 两张 mask ----------
    # 仿真器的真实网格尺寸 (pixel_size=6 时是 257x257); 必须按它对齐
    H_sim, W_sim = simulator.mask.data.shape
    mask_pixel = find_pixel_mask()
    mask_gds = rasterize_gds_with_msaa(GDS_PATH, H, W, PIXEL_SIZE_NM)

    def to_sim_grid(arr: np.ndarray) -> np.ndarray:
        """把 (H, W) 大小的 mask 对齐到仿真器网格 (H_sim, W_sim).

        设计期 H/W=256, 仿真期 257; 差一个像素时做右下补零 (与 set_meef
        在同样情况下的 align 行为对齐).
        """
        if arr.shape == (H_sim, W_sim):
            return arr
        out = np.zeros((H_sim, W_sim), dtype=arr.dtype)
        h_c = min(arr.shape[0], H_sim)
        w_c = min(arr.shape[1], W_sim)
        out[:h_c, :w_c] = arr[:h_c, :w_c]
        return out

    mask_pixel = to_sim_grid(mask_pixel)
    mask_gds = to_sim_grid(mask_gds)

    # ---------- 3. mask 差异 ----------
    diff = mask_gds - mask_pixel
    print("\n========== mask 像素差异 ==========")
    print(f"shapes        : pixel {mask_pixel.shape}, gds {mask_gds.shape}")
    print(f"min / max     : pixel [{mask_pixel.min():.3f}, {mask_pixel.max():.3f}]  "
          f"gds [{mask_gds.min():.3f}, {mask_gds.max():.3f}]")
    print(f"max abs diff  : {np.max(np.abs(diff)):.4f}")
    print(f"mean abs diff : {np.mean(np.abs(diff)):.4f}")
    print(f"L2  diff      : {np.linalg.norm(diff):.4f}")

    # ---------- 4. aerial 差异 ----------
    ai_p, wf_p, _ = images_simulation_v2(
        mask_spatial=mask_pixel,
        resist_params=params.resist,
        opt_cache=simulator.opt_cache,
    )
    ai_g, wf_g, _ = images_simulation_v2(
        mask_spatial=mask_gds,
        resist_params=params.resist,
        opt_cache=simulator.opt_cache,
    )
    ai_diff = ai_g - ai_p
    print("\n========== aerial image 差异 ==========")
    print(f"max abs diff  : {np.max(np.abs(ai_diff)):.5f}")
    print(f"mean abs diff : {np.mean(np.abs(ai_diff)):.5f}")
    print(f"相对均方差    : {np.sqrt(np.mean(ai_diff**2))/np.sqrt(np.mean(ai_p**2))*100:.4f} %")

    # ---------- 5. wafer (resist 二值) IoU ----------
    inter = np.sum((wf_p > 0.5) & (wf_g > 0.5))
    union = np.sum((wf_p > 0.5) | (wf_g > 0.5))
    iou = inter / max(union, 1)
    print("\n========== wafer (resist) 差异 ==========")
    print(f"wafer IoU     : {iou:.6f}")
    print(f"wafer 像素差异: {np.sum((wf_p > 0.5) ^ (wf_g > 0.5))} / {H*W} 像素")

    # ---------- 6. EPE ----------
    eps = np.loadtxt(RESULT_DIR / "eps.txt").astype(int)
    weight_meef_path = RESULT_DIR / "eps.txt"
    # 没存 weight, 这里用全 1 算"非加权 EPE"做粗对比就够看差距
    n_eps = len(eps)
    weight = np.ones((1, n_eps))
    epe_p, _ = caculate_epe(ai_p, params.resist.threshold, eps, n_eps, weight, False)
    epe_g, _ = caculate_epe(ai_g, params.resist.threshold, eps, n_eps, weight, False)
    print("\n========== EPE 对比 ==========")
    print(f"EPE pixel mask : {epe_p:.4f}")
    print(f"EPE gds   mask : {epe_g:.4f}")
    print(f"差异 (EPE_g - EPE_p): {epe_g - epe_p:+.4f}")

    # ---------- 7. 可视化 ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=110)
    axes[0, 0].imshow(mask_pixel, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("mask (pixel from CP render)")
    axes[0, 1].imshow(mask_gds, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("mask (GDS rasterize)")
    im2 = axes[0, 2].imshow(diff, cmap="RdBu", vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title("mask diff (gds - pixel)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(ai_p, cmap="hot")
    axes[1, 0].set_title("aerial (pixel)")
    axes[1, 1].imshow(ai_g, cmap="hot")
    axes[1, 1].set_title("aerial (gds)")
    im5 = axes[1, 2].imshow(ai_diff, cmap="RdBu",
                            vmin=-np.max(np.abs(ai_diff)),
                            vmax=np.max(np.abs(ai_diff)))
    axes[1, 2].set_title(f"aerial diff (max={np.max(np.abs(ai_diff)):.4f})")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        ax.set_axis_off()
    plt.tight_layout()
    out_png = RESULT_DIR / "compare_pixel_vs_gds.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_png}")


if __name__ == "__main__":
    main()
