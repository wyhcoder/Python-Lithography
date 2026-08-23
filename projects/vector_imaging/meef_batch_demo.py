"""
矢量 MEEF 批量优化 Demo。

用法:
    python3 meef_batch_demo.py [--config configs/litho_system.yaml]
                                [--iterations 20]
                                [--delta 0.5]
                                [--batch 0]

流程:
    1. 读 YAML → 构建 Grid/Pupil/Illumination/VectorTransfer/VectorSOCS
    2. 从 target 图像提取 CP → 构建 SDF 渲染器
    3. 每轮迭代:
       a. 多线程渲染所有 ±delta 扰动 mask
       b. 叠成 batch → 一次 forward (raw aerial) → EPE
       c. 中心差分 → MEEF 矩阵 → L-curve TSVD → Δd
       d. 更新 CP → 评估
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

# 确保可以 import litho_model
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from litho_model.config import SimulationConfig
from litho_model.forwardImage import VectorSOCSImaging
from litho_model.grid import Grid
from litho_model.pupil import Pupil
from litho_model.socs import VectorSOCS
from litho_model.source import Illumination, SourceType
from litho_model.vector_transfer import VectorTransfer
from litho_model.meef_opt import VectorMEEFOptimizer
from tool.paths import project_root


# ─────────────────────── 构建光学系统 ───────────────────────

def build_optical_system(cfg_path: str, device: str = "cpu"):
    """构建 Grid / Pupil / Illumination / VectorTransfer / VectorSOCS / VectorSOCSImaging。"""
    cfg = SimulationConfig.from_yaml(cfg_path)
    dtype = torch.float32
    dev = torch.device(device)

    print("[1] 构建光学系统 ...")
    grid = Grid(
        N=cfg.system.mask_size,
        dx=cfg.system.pixel_size_nm,
        device=dev, dtype=dtype,
    )
    pupil = Pupil(
        NA=cfg.pupil.na,
        wave_length=cfg.system.wavelength_nm,
        refractive_index=cfg.pupil.n_image,
        grid=grid,
        aberrations=cfg.pupil.aberration,
        defocus_nm=cfg.pupil.defocus_nm,
        device=dev,
    )
    illum = Illumination(
        pupil=pupil, grid=grid,
        source_type=SourceType(cfg.source.type),
        sigma_in=cfg.source.sigma_in,
        sigma_out=cfg.source.sigma_out,
        smoothing=cfg.source.smoothing,
        upsample=cfg.source.upsample,
    )
    vt = VectorTransfer(pupil=pupil, grid=grid)
    socs = VectorSOCS(
        pupil=pupil, illumination=illum, vector_transfer=vt,
        K=10, weight_threshold=1e-6,
        unpolarized=True,  # 非偏振
    )
    imaging = VectorSOCSImaging(
        socs=socs, grid=grid,
        sigmoid_tr=cfg.resist.sigmoid_params.threshold,
        sigmoid_a=cfg.resist.sigmoid_params.alpha,
    )
    print(f"    λ={cfg.system.wavelength_nm}nm, NA={cfg.pupil.na}, "
          f"N={grid.N}, dx={grid.dx}nm, K={socs.K}")
    return imaging, cfg


# ─────────────────────── 生成 demo target mask ───────────────────────

def make_demo_target(N: int) -> np.ndarray:
    """生成 contact hole 类测试图形。"""
    mask = np.zeros((N, N), dtype=np.float64)
    cy, cx = N // 2, N // 2
    # 中心方孔
    r = 50
    mask[cy - r:cy + r, cx - r:cx + r] = 1.0
    return mask


# ─────────────────────── main ───────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="矢量 MEEF 批量优化 Demo")
    p.add_argument("--config", default=str(project_root() / "configs" / "litho_system.yaml"),
                   help="YAML 配置路径")
    p.add_argument("--iterations", type=int, default=40,
                   help="优化迭代次数")
    p.add_argument("--delta", type=float, default=0.5,
                   help="CFD 扰动步长 (像素)")
    p.add_argument("--batch", type=int, default=0,
                   help="每批仿真的 mask 数 (0=全部一起)")
    p.add_argument("--sdf-beta", type=float, default=0.25,
                   help="SDF 软化宽度")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=str(project_root() / "output" / "meef_opt"),
                   help="输出目录")
    p.add_argument("--cp-interval", type=int, default=8,
                   help="CP 采样间隔 (越大 CP 越少)")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. 构建光学系统
    imaging, cfg = build_optical_system(args.config, args.device)
    N = cfg.system.mask_size

    # 2. 生成 target mask
    print("[2] 生成 target mask ...")
    target = make_demo_target(N)
    
    print(f"    target shape: {target.shape}, "
          f"nonzero: {np.count_nonzero(target)}")

    # 3. 创建优化器 + 自动提取 CP
    print("[3] 初始化优化器 ...")
    opt_config = {
        "delta": args.delta,
        "iterations": args.iterations,
        "sdf_beta": args.sdf_beta,
        "threshold": cfg.resist.sigmoid_params.threshold,
        "curve_type": "OA",
        "output_dir": args.output,
        "batch_size": args.batch,
    }
    optimizer = VectorMEEFOptimizer(imaging, target, opt_config)
    optimizer.setup_auto(interval=args.cp_interval)

    # 4. 运行优化
    print("[4] 开始优化 ...")
    best_cps, best_mask = optimizer.run()

    print(f"\nDone! 结果保存至: {args.output}")


if __name__ == "__main__":
    main()
