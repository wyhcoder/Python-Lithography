"""
可视化 test.py 中的掩膜、光源以及对应的矢量成像结果
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from test import VectorAerialModel
from tool.paths import output_path

# 中文显示
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
rcParams["axes.unicode_minus"] = False


def main():
    # ---------- 1. 构建模型 ----------
    N = 257
    dx_nm = 6.0

    model = VectorAerialModel(
        N=N,
        dx_nm=dx_nm,
        lambda0_nm=193.0,
        n_image=1.44,
        NA=1.35,
        device="cpu",
    )

    # ---------- 2. 掩膜 ----------
    mask = torch.zeros((N, N))
    mask[100:157, 120:137] = 1.0  # 一条竖直线

    # ---------- 3. 环形光源 ----------
    Ns = 29
    sigma_in = 0.6
    sigma_out = 0.9
    sx, sy, J = model.make_annular_source(
        Ns=Ns, sigma_in=sigma_in, sigma_out=sigma_out
    )

    # ---------- 4. 三种偏振的成像 ----------
    print("Computing x-polarized aerial image ...")
    Ix = model.forward_single_polarization(mask, sx, sy, J, jones=(1.0, 0.0))
    print("Computing y-polarized aerial image ...")
    Iy = model.forward_single_polarization(mask, sx, sy, J, jones=(0.0, 1.0))
    print("Computing unpolarized aerial image ...")
    Iunp = 0.5 * (Ix + Iy)

    # ---------- 5. 转 numpy ----------
    mask_np = mask.cpu().numpy()
    sx_np = sx.cpu().numpy()
    sy_np = sy.cpu().numpy()
    J_np = J.cpu().numpy()
    Ix_np = Ix.cpu().numpy()
    Iy_np = Iy.cpu().numpy()
    Iunp_np = Iunp.cpu().numpy()

    # 空间坐标范围（nm），用于 imshow 的 extent
    half = (N // 2) * dx_nm
    extent_xy = [-half, half, -half, half]

    # 光源坐标范围（归一化到 NA），sx,sy in [-sigma_out, sigma_out]
    extent_src = [-sigma_out, sigma_out, -sigma_out, sigma_out]

    # ---------- 6. 绘图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # (0,0) 掩膜
    ax = axes[0, 0]
    im0 = ax.imshow(
        mask_np.T,
        origin="lower",
        extent=extent_xy,
        cmap="gray",
        interpolation="nearest",
    )
    ax.set_title("掩膜 Mask (透过率)")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    # (0,1) 光源（环形照明）
    ax = axes[0, 1]
    im1 = ax.imshow(
        J_np.T,
        origin="lower",
        extent=extent_src,
        cmap="hot",
        interpolation="nearest",
    )
    ax.set_title(
        f"环形光源 J(σ)\nσ_in={sigma_in}, σ_out={sigma_out}, Ns={Ns}"
    )
    ax.set_xlabel(r"$\sigma_x$ (归一化到 NA)")
    ax.set_ylabel(r"$\sigma_y$ (归一化到 NA)")
    # 画出 NA=1 的边界（pupil 边界）
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "w--", lw=1, label="pupil (σ=1)")
    ax.plot(
        sigma_in * np.cos(theta),
        sigma_in * np.sin(theta),
        "c--",
        lw=0.8,
        label=f"σ_in={sigma_in}",
    )
    ax.plot(
        sigma_out * np.cos(theta),
        sigma_out * np.sin(theta),
        "y--",
        lw=0.8,
        label=f"σ_out={sigma_out}",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # (0,2) 物镜光瞳 + 截止频率
    ax = axes[0, 2]
    pupil_np = model.pupil.cpu().numpy()
    fmax = model.NA / model.lambda0_nm  # cycles/nm
    fx = torch.fft.fftshift(model.FX[:, 0]).cpu().numpy()
    fy = torch.fft.fftshift(model.FY[0, :]).cpu().numpy()
    pupil_shift = np.fft.fftshift(pupil_np)
    extent_f = [fx.min(), fx.max(), fy.min(), fy.max()]
    im2 = ax.imshow(
        pupil_shift.T,
        origin="lower",
        extent=extent_f,
        cmap="gray",
        interpolation="nearest",
    )
    ax.set_title(f"物镜光瞳 Pupil\nNA={model.NA}, f_max={fmax:.4f} cyc/nm")
    ax.set_xlabel(r"$f_x$ (cycles/nm)")
    ax.set_ylabel(r"$f_y$ (cycles/nm)")
    ax.set_aspect("equal")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # 统一三幅成像的色阶上下限，便于比较
    vmax = max(Ix_np.max(), Iy_np.max(), Iunp_np.max())
    vmin = 0.0

    # (1,0) X 偏振成像
    ax = axes[1, 0]
    im3 = ax.imshow(
        Ix_np.T,
        origin="lower",
        extent=extent_xy,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("X 偏振成像 $I_x$")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    # (1,1) Y 偏振成像
    ax = axes[1, 1]
    im4 = ax.imshow(
        Iy_np.T,
        origin="lower",
        extent=extent_xy,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("Y 偏振成像 $I_y$")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

    # (1,2) 非偏振成像（X/Y 平均）
    ax = axes[1, 2]
    im5 = ax.imshow(
        Iunp_np.T,
        origin="lower",
        extent=extent_xy,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("非偏振成像 $I_{unp}=(I_x+I_y)/2$")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    plt.colorbar(im5, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    out_path = output_path("visualize_result.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    # ---------- 7. 中心切线对比 ----------
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))

    # 沿 y 方向中心切线（穿过竖线掩膜的水平剖面）
    cy = N // 2
    x_axis = (np.arange(N) - N // 2) * dx_nm

    ax2[0].plot(x_axis, mask_np[:, cy], "k--", label="mask", lw=1)
    ax2[0].plot(x_axis, Ix_np[:, cy], label="$I_x$")
    ax2[0].plot(x_axis, Iy_np[:, cy], label="$I_y$")
    ax2[0].plot(x_axis, Iunp_np[:, cy], label="$I_{unp}$", lw=2, alpha=0.7)
    ax2[0].set_title("水平剖面 (y=0)")
    ax2[0].set_xlabel("x (nm)")
    ax2[0].set_ylabel("强度 / 透过率")
    ax2[0].legend()
    ax2[0].grid(alpha=0.3)

    # 沿 x 方向中心切线（垂直剖面）
    cx = N // 2
    y_axis = (np.arange(N) - N // 2) * dx_nm
    ax2[1].plot(y_axis, mask_np[cx, :], "k--", label="mask", lw=1)
    ax2[1].plot(y_axis, Ix_np[cx, :], label="$I_x$")
    ax2[1].plot(y_axis, Iy_np[cx, :], label="$I_y$")
    ax2[1].plot(y_axis, Iunp_np[cx, :], label="$I_{unp}$", lw=2, alpha=0.7)
    ax2[1].set_title("垂直剖面 (x=0)")
    ax2[1].set_xlabel("y (nm)")
    ax2[1].set_ylabel("强度 / 透过率")
    ax2[1].legend()
    ax2[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path2 = output_path("visualize_profiles.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {out_path2}")

    plt.show()


if __name__ == "__main__":
    main()
