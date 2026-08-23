"""
对比三种优化方法的 CP 移动轨迹: bisector_MEEF / xy_MEEF / xy_gradient。

选若干位移差异最大的 CP, 每个 CP 画一个子图, 三种方法叠加对比。
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

mpl.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei",
    "Microsoft YaHei", "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_cp_history(cp_dir):
    files = sorted(Path(cp_dir).glob("cp_iter_*.txt"))
    all_cps = []
    for fp in files:
        pts = []
        for line in open(fp):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            pts.append([float(a), float(b)])
        all_cps.append(np.asarray(pts))
    return np.asarray(all_cps)  # (T, K, 2) [y, x]


def load_target_mask(mask_path):
    if not Path(mask_path).is_file():
        return None
    return np.loadtxt(mask_path)


def main():
    base = "outputs/opc/MEEF_pipeline/BS"
    dirs = {
        "bisector_MEEF": f"{base}/工字型_k=5_sym=none_srafArc=5.0_BS_pipeline_test_bisector",
        "xy_MEEF":       f"{base}/工字型_k=5_sym=none_srafArc=5.0_BS_pipeline_test_xy",
        "xy_gradient":   f"{base}/工字型_k=5_sym=none_srafArc=5.0_BS_pipeline_test_xy_gradient",
    }

    # 加载三种方法的 CP 历史
    data = {}
    for label, d in dirs.items():
        cp_dir = f"{d}/curves/cp_history"
        data[label] = load_cp_history(cp_dir)
        print(f"{label}: {data[label].shape}")

    # 三种方法形状应一致
    T, K, _ = data["bisector_MEEF"].shape

    # 目标版图 (三种方法应该用同一个)
    target = load_target_mask(f"{dirs['bisector_MEEF']}/target_mask.txt")

    # 选三种方法轨迹差异最大的 6 个 CP
    diffs = np.zeros(K)
    labels = list(data.keys())
    for i in range(K):
        # 三条轨迹两两之间的平均距离之和
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                d = np.linalg.norm(data[labels[a]][:, i, :] - data[labels[b]][:, i, :], axis=1)
                diffs[i] += d.mean()
    top8 = np.argsort(diffs)[::-1][:8]
    print(f"选取 CP: {top8.tolist()}")

    # 三种方法的颜色和标记
    styles = {
        "bisector_MEEF": {"color": "#2f4858", "marker": "o", "start": "red", "end": "blue"},
        "xy_MEEF":       {"color": "#c0524a", "marker": "s", "start": "green", "end": "purple"},
        "xy_gradient":   {"color": "#4a7c59", "marker": "^", "start": "orange", "end": "brown"},
    }

    scale = 5.0  # 放大倍数

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for ax_idx, cp_i in enumerate(top8):
        ax = axes[ax_idx]

        # 目标版图轮廓
        if target is not None:
            ax.contour(target, levels=[0.5], colors="0.85",
                       linewidths=1.0, alpha=0.6)

        for label in labels:
            traj = data[label][:, cp_i, :]  # (T, 2) [y, x]
            x0, y0 = traj[0, 1], traj[0, 0]
            xs = x0 + (traj[:, 1] - x0) * scale
            ys = y0 + (traj[:, 0] - y0) * scale
            s = styles[label]
            ax.plot(xs, ys, "-", color=s["color"], lw=1.5, alpha=0.85,
                    label=label if ax_idx == 0 else "")
            ax.plot(xs[0], ys[0], marker=s["marker"], color=s["start"],
                    markersize=6, zorder=4, linestyle="None")
            ax.plot(xs[-1], ys[-1], marker=s["marker"], color=s["end"],
                    markersize=6, zorder=4, linestyle="None")

        ax.set_title(f"CP #{cp_i}  ({scale:.0f}x)", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", alpha=0.3)

        # 聚焦
        all_x = np.concatenate([x0 + (data[l][:, cp_i, 1] - x0) * scale for l in labels])
        all_y = np.concatenate([y0 + (data[l][:, cp_i, 0] - y0) * scale for l in labels])
        pad = 5
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.max() + pad, all_y.min() - pad)

    # 总图例
    from matplotlib.lines import Line2D
    legend_elems = []
    for label in labels:
        s = styles[label]
        legend_elems.append(
            Line2D([0], [0], color=s["color"], lw=1.5, label=label)
        )
    legend_elems.append(Line2D([0], [0], color="0.5", linewidth=1.5, label="Target contour"))
    fig.legend(handles=legend_elems, loc="lower center", ncol=len(legend_elems), fontsize=9)

    fig.suptitle(f"CP Trajectory Comparison: bisector_MEEF vs xy_MEEF vs xy_gradient ({scale:.0f}x)",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out = f"{base}/cp_trajectory_compare_3way.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


if __name__ == "__main__":
    main()
