"""
批量光刻仿真脚本。

用法：
    python3 batch_simulate.py [--config configs/litho_system.yaml]
                               [--input  target_images/]
                               [--output output/wafer/]
                               [--batch  32]
                               [--save-png]
                               [--workers 4]

流程：
1. 读 yaml 配置，构建 Grid / Pupil / Illumination / VectorTransfer / ForwardImaging
2. 用 MaskDataset + DataLoader 遍历 target_images/
3. 每个 batch 调用 forward_batch_unpolarized，结果存到 output/wafer/
4. 进度条显示 ETA / 吞吐量
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from turtle import forward

import numpy as np
import torch
from torch.utils.data import DataLoader
from litho_model.socs import VectorSOCS,ScalarSOCS
from litho_model.config import SimulationConfig
from litho_model.forwardImage import VectorForwardImaging, ScalarForwardImaging
from litho_model.forwardImage import VectorSOCSImaging, ScalarSOCSImaging
from litho_model.grid import Grid
from litho_model.mask import MaskDataset
from litho_model.pupil import Pupil
from litho_model.source import Illumination, SourceType
from litho_model.vector_transfer import VectorTransfer
from tool.paths import output_path, project_root


# ─────────────────────── CLI ───────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量矢量光刻仿真")
    p.add_argument(
        "--config",
        default=str(project_root() / "configs" / "litho_system.yaml"),
        help="YAML 配置文件路径（默认：configs/litho_system.yaml）",
    )
    p.add_argument(
        "--input",
        default=str(project_root() / "target_images"),
        help="mask 图像目录（默认：target_images/）",
    )
    p.add_argument(
        "--output",
        default=str(project_root() / "output" / "wafer"),
        help="输出目录（默认：output/wafer/）",
    )
    p.add_argument(
        "--batch", type=int, default=8,
        help="每次处理的 mask 张数（默认：8）",
    )
    p.add_argument(
        "--save-png", action="store_true",
        help="同时保存可视化 png（默认只保存 npy）",
    )
    p.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader num_workers（默认 0，macOS 推荐 0）",
    )
    p.add_argument(
        "--device", default="cpu",
        help="torch device（默认 cpu；有 GPU 可改为 cuda）",
    )
    p.add_argument(
        "--limit", type=int, default=10,
        help="最多处理前 N 张（调试用，默认全部处理）",
    )
    p.add_argument(
        "--type",choices=['vector','scalar'],default='scalar',
        help="成像方式：vector（默认）或 scalar"
    )
    return p.parse_args()


# ─────────────────────── 构建光学系统 ───────────────────────

def build_optical_system(cfg: SimulationConfig, args:argparse.Namespace,device: torch.device, dtype: torch.dtype):
    """根据配置构造 Grid / Pupil / Illumination / VectorTransfer / ForwardImaging。"""
    print("  构造 grid ...")
    grid = Grid(
        N=cfg.system.mask_size,
        dx=cfg.system.pixel_size_nm,
        device=device,
        dtype=dtype,
    )

    print("  构造 pupil ...")
    pupil = Pupil(
        NA=cfg.pupil.na,
        wave_length=cfg.system.wavelength_nm,
        refractive_index=cfg.pupil.n_image,
        grid=grid,
        aberrations=cfg.pupil.aberration,
        defocus_nm=cfg.pupil.defocus_nm,
        device=device,
    )

    print("  构造 source ...")
    illumination = Illumination(
        pupil=pupil,
        grid=grid,
        source_type=SourceType(cfg.source.type),
        sigma_in=cfg.source.sigma_in,
        sigma_out=cfg.source.sigma_out,
        smoothing=cfg.source.smoothing,
        upsample=cfg.source.upsample,
    )

    print("  构造 vector transfer ...")
    vt = VectorTransfer(pupil=pupil, grid=grid)

    print("  构造 forward imaging ...")
    if args.type == 'vector':
        forward = VectorForwardImaging(
            grid=grid,
            pupil=pupil,
            illumination=illumination,
            vector_transfer=vt,
            weight_threshold=1e-6,
            chunk_size=64,
        )
        socs = VectorSOCS(pupil=pupil, illumination=illumination, vector_transfer=vt,
                          K=10, weight_threshold=1e-6, jones=(1.0, 0.0))
        forward = VectorSOCSImaging(socs=socs,
                                grid=grid,)
    else:
        forward = ScalarForwardImaging(
            grid=grid,
            pupil=pupil,
            illumination=illumination,
            weight_threshold=1e-6,
            chunk_size=64,
        )
        socs = ScalarSOCS(pupil=pupil, illumination=illumination, 
                          K=10, weight_threshold=1e-6)
        forward = ScalarSOCSImaging(socs=socs, grid=grid)
    
                            
    
    return forward


# ─────────────────────── 保存结果 ───────────────────────

def save_results(
    aerials: torch.Tensor,   # [B, N, N]
    names: list[str],
    out_dir: Path,
    save_png: bool,
) -> None:
    """把一个 batch 的结果保存到 out_dir。"""
    aerials_np = aerials.detach().cpu().numpy()  # [B, N, N]
    for img, name in zip(aerials_np, names):
        stem = Path(name).stem
        # 保存 npy（浮点精度，后续可直接 load 做训练数据）
        np.save(str(out_dir / f"{stem}.npy"), img)
        # 可选：保存 png（uint8 可视化）
        if save_png:
            try:
                from PIL import Image as PILImage
                vis = (img * 255).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(vis).save(str(out_dir / f"{stem}.png"))
            except Exception as e:
                print(f"  [warn] png 保存失败 {stem}: {e}")


# ─────────────────────── 主流程 ───────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32

    # 1. 读配置
    print(f"[1] 读取配置: {args.config}")
    cfg = SimulationConfig.from_yaml(args.config)
    print(
        f"    系统: λ={cfg.system.wavelength_nm}nm, dx={cfg.system.pixel_size_nm}nm,"
        f" N={cfg.system.mask_size}"
    )

    # 2. 构建光学系统（只构建一次，所有 mask 复用）
    print("[2] 构建光学系统 ...")
    forward = build_optical_system(cfg, args,device, dtype)
    print("    完成。")

    # 3. 准备数据集
    print(f"[3] 扫描 mask 目录: {args.input}")
    dataset = MaskDataset(
        mask_dir=args.input,
        mask_size=cfg.system.mask_size,
        normalize=True,
    )
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"    找到 {len(dataset)} 张图像，本次处理 {total} 张。")

    # 若有 limit，截取子集
    if args.limit is not None:
        indices = list(range(args.limit))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(args.device != "cpu"),
    )

    # 4. 准备输出目录
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[4] 输出目录: {out_dir}")

    # 5. 批量仿真
    print(f"[5] 开始批量仿真 (batch_size={args.batch}) ...")
    t0 = time.time()
    n_done = 0

    for batch_masks, batch_names in loader:
        batch_masks = batch_masks.to(device=device, dtype=dtype)  # [B, N, N]

        with torch.no_grad():
            # aerials = forward.forward_batch_unpolarized(batch_masks)  # [B, N, N]
            aerials = forward.forward_batch(batch_masks)
        save_results(aerials, list(batch_names), out_dir, args.save_png)

        n_done += len(batch_names)
        elapsed = time.time() - t0
        speed = elapsed / n_done  # s/张
        eta = (total - n_done) / speed if speed > 0 else float("inf")
        print(
            f"    进度: {n_done}/{total}"
            f"  速度: {speed:.3f} 张/s"
            f"  ETA: {eta:.3f}s",
            end="\r",
        )

    elapsed_total = time.time() - t0
    print(
        f"\n[完成] 共处理 {n_done} 张，总耗时 {elapsed_total:.3f}s"
        f"（平均 {elapsed_total / n_done:.3f} 张/s）"
    )
    print(f"       结果保存到: {out_dir}")


if __name__ == "__main__":
    main()
