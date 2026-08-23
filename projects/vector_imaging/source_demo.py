"""
litho_model/source.py 的 demo + 可视化测试。

依次生成 conventional / annular / dipole / quasar 四种 source，
并将结果绘制到一张图中保存为 source_demo.png。
"""
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

from litho_model.grid import Grid
from litho_model.pupil import Pupil
from litho_model.source import Illumination, SourceType
from tool.paths import output_path

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
rcParams["axes.unicode_minus"] = False


def build_pupil(N: int = 257, dx_nm: float = 10.0) -> tuple[Grid, Pupil]:
    grid = Grid(N=N, dx=dx_nm)
    pupil = Pupil(
        NA=1.35,
        wave_length=193.0,
        refractive_index=1.44,
        grid=grid,
        aberrations={},
        defocus_nm=0.0,
    )
    return grid, pupil


def _draw_circle(ax, r, **kwargs):
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(theta), r * np.sin(theta), **kwargs)


def make_source(mask: Grid, pupil: Pupil, kind: SourceType) -> Illumination:
    if kind == SourceType.CONVENTIONAL:
        return Illumination(
            pupil, mask, source_type=kind,
            sigma_in=0.0, sigma_out=0.6,
            smoothing=0.01, upsample=8,
        )
    if kind == SourceType.ANNULAR:
        return Illumination(
            pupil, mask, source_type=kind,
            sigma_in=0.6, sigma_out=0.9,
            smoothing=0.01, upsample=8,
        )
    if kind == SourceType.DIPOLE:
        return Illumination(
            pupil, mask, source_type=kind,
            sigma_in=0.6, sigma_out=0.9,
            smoothing=0.01, upsample=8,
            lobe_half_width_deg=30.0,
            angular_smoothing_deg=2.0,
        )
    if kind == SourceType.QUASAR:
        return Illumination(
            pupil, mask, source_type=kind,
            sigma_in=0.6, sigma_out=0.9,
            smoothing=0.01, upsample=8,
            lobe_half_width_deg=30.0,
            angular_smoothing_deg=2.0,
        )
    raise ValueError(kind)


def main():
    print("[1/3] 构造 grid 与 pupil ...")
    mask, pupil = build_pupil(N=257, dx_nm=10.0)

    print("[2/3] 生成四种 source ...")
    kinds = [
        SourceType.CONVENTIONAL,
        SourceType.ANNULAR,
        SourceType.DIPOLE,
        SourceType.QUASAR,
    ]
    illums = {k: make_source(mask, pupil, k) for k in kinds}

    # 一致性检查
    for k, ill in illums.items():
        smap = ill.get_source().cpu().numpy()
        wmap = ill.get_weights().cpu().numpy()
        print(
            f"  - {k.value:<13s} source_map shape={smap.shape}, "
            f"max={smap.max():.4f}, weight sum={wmap.sum():.6f}"
        )
        assert np.isclose(wmap.sum(), 1.0, atol=1e-6), "权重总和必须归一化为 1"

    print("[3/3] 绘图 ...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, k in zip(axes, kinds):
        ill = illums[k]
        smap = ill.get_source().cpu().numpy()
        sx, sy = ill.get_axes()
        sx_np = sx.cpu().numpy()
        sy_np = sy.cpu().numpy()
        extent = [sx_np.min(), sx_np.max(), sy_np.min(), sy_np.max()]

        im = ax.imshow(
            smap, origin="lower", extent=extent, cmap="hot",
            interpolation="nearest",
        )
        # 画出 sigma_in / sigma_out / pupil 边界
        _draw_circle(ax, 1.0, color="white", lw=1.0, ls="--", label="pupil σ=1")
        _draw_circle(ax, ill.sigma_out, color="yellow", lw=0.8, ls="--",
                     label=f"σ_out={ill.sigma_out:g}")
        if ill.sigma_in > 0:
            _draw_circle(ax, ill.sigma_in, color="cyan", lw=0.8, ls="--",
                         label=f"σ_in={ill.sigma_in:g}")
        ax.set_title(k.value)
        ax.set_xlabel(r"$\sigma_x$")
        ax.set_ylabel(r"$\sigma_y$")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Illumination source demo  (NA=1.35, λ=193nm, n=1.44)",
        fontsize=13,
    )
    plt.tight_layout()
    out_path = output_path("source_demo.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
