"""
绘制 B-spline 控制点 (CP) 的移动轨迹示意图 (可叠加目标版图作背景)。

读取 curves/cp_history/cp_iter_XXX.txt (由 demo_meef.py 的 _save_cp_history 保存),
对全部 CP 做下采样或手动选取, 把每个被跟踪 CP 从 iter 0 → iter N 的逐次位置
连成折线, 末端画箭头表示最终移动方向。

风格与 curves/bspline_evolution.png 保持一致 (灰色底图 + origin='upper' + y 轴反向)。

用法:
    # 默认: 叠加 target_mask.txt 作背景, 每隔 4 个 CP 取一个
    python scripts/analysis/plot_cp_trajectory.py \
        --cp_dir outputs/opc/DPS/BS/0.8_BS_工字型/curves/cp_history

    # 手动指定要跟踪的 CP 下标 (0-based)
    python scripts/analysis/plot_cp_trajectory.py --cp_dir .../cp_history \
        --cp_indices 0 5 10 15 20

    # 只画位移最大的前 K 个 CP
    python scripts/analysis/plot_cp_trajectory.py --cp_dir .../cp_history --top_k 15

    # 不叠加背景 (纯轨迹图)
    python scripts/analysis/plot_cp_trajectory.py --cp_dir .../cp_history --no_background

    # 手动指定背景版图路径
    python scripts/analysis/plot_cp_trajectory.py --cp_dir .../cp_history \
        --target_mask 路径/target_mask.txt
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 中文字体配置 (macOS 优先 PingFang SC / Arial Unicode MS)
mpl.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei",
    "Microsoft YaHei", "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


# ----------------------- IO ----------------------- #
def load_cp_history(cp_dir: Path):
    """
    加载 cp_iter_XXX.txt 全部文件。
    文件每行两列, 与 bspline_iter_XXX.txt 相同的 [y, x] 格式。

    Returns
    -------
    cps_history : (T, K, 2) ndarray  内容为 [y, x], T=迭代数, K=CP 数
    iter_ids    : list[int]          对应的迭代编号
    """
    files = sorted(cp_dir.glob("cp_iter_*.txt"))
    if not files:
        raise FileNotFoundError(f"未在 {cp_dir} 找到 cp_iter_*.txt")

    all_cps = []
    iter_ids = []
    for fp in files:
        pts = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                # 文件里是 [y, x], 保持原顺序不改
                pts.append([float(a), float(b)])
        all_cps.append(np.asarray(pts, dtype=float))
        stem = fp.stem  # cp_iter_015
        try:
            iter_ids.append(int(stem.split("_")[-1]))
        except ValueError:
            iter_ids.append(len(iter_ids))

    K = all_cps[0].shape[0]
    for i, c in enumerate(all_cps):
        if c.shape[0] != K:
            raise ValueError(
                f"iter {iter_ids[i]} 的 CP 数={c.shape[0]} 与首帧 K={K} 不一致"
            )

    return np.asarray(all_cps), iter_ids  # (T, K, 2), 内容 [y, x]


def load_target_mask(mask_path: Path):
    """加载 target_mask.txt (整数网格 0/1 mask)。"""
    if not mask_path.is_file():
        return None
    try:
        mask = np.loadtxt(mask_path)
        return mask
    except Exception as e:
        print(f"[warn] 加载目标版图失败: {e}")
        return None


# ----------------------- Plot ----------------------- #
def plot_cp_trajectory(
    cps_history: np.ndarray,
    out_path: Path,
    target_mask=None,
    track_indices=None,
    step: int = 4,
    top_k: int = None,
    figsize=(9, 9),
    title: str = None,
    coord_yx: bool = True,
    zoom_pad: float = None,
):
    """
    cps_history : (T, K, 2)
        默认 coord_yx=True 表示内容是 [y, x], 与 bspline_iter 格式一致。
    target_mask : (H, W) or None
        目标版图, 作为灰色背景。
    """
    T, K, _ = cps_history.shape

    # ---- 决定要跟踪哪些 CP ----
    if track_indices is not None:
        idx = np.asarray(track_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < K)]
        sel_mode = "manual"
    elif top_k is not None and top_k > 0:
        disp = np.linalg.norm(cps_history[-1] - cps_history[0], axis=1)
        idx = np.argsort(disp)[::-1][:top_k]
        idx = np.sort(idx)
        sel_mode = f"top-{top_k}"
    else:
        idx = np.arange(0, K, step)
        sel_mode = f"step={step}"

    if len(idx) == 0:
        raise ValueError("没有选中任何 CP, 检查 --step / --cp_indices / --top_k")

    # ---- 拆分绘图坐标: matplotlib 用 (x, y) ----
    if coord_yx:
        # 文件是 [y, x] → 画图 x=col=cps[..., 1], y=row=cps[..., 0]
        get_xy = lambda pts: (pts[..., 1], pts[..., 0])
    else:
        get_xy = lambda pts: (pts[..., 0], pts[..., 1])

    # ---- 颜色按总位移幅度上色 ----
    total_disp = np.linalg.norm(cps_history[-1] - cps_history[0], axis=1)  # (K,)
    max_disp = total_disp[idx].max() if total_disp[idx].max() > 1e-9 else 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # ---- 背景: 目标版图轮廓线 (不是像素填充, 只画边界) ----
    if target_mask is not None:
        ax.contour(target_mask, levels=[0.5],
                   colors="0.5", linewidths=1.5, alpha=0.8)
        H, W = target_mask.shape
    else:
        ys = cps_history[..., 0] if coord_yx else cps_history[..., 1]
        xs = cps_history[..., 1] if coord_yx else cps_history[..., 0]
        H = int(np.ceil(ys.max())) + 20
        W = int(np.ceil(xs.max())) + 20

    # ---- 轨迹: 黑色曲线(放大) + 红色起止点 ----
    scale = 1.0  # 位移放大倍数, 让小位移轨迹清晰可见
    for i in idx:
        traj = cps_history[:, i, :]               # (T, 2), [y, x]
        xs_traj, ys_traj = get_xy(traj)

        # 以起点为中心放大位移: 每个点 = 起点 + scale * (点 - 起点)
        x0, y0 = xs_traj[0], ys_traj[0]
        xs_scaled = x0 + (xs_traj - x0) * scale
        ys_scaled = y0 + (ys_traj - y0) * scale

        # 黑色曲线连接所有迭代位置
        ax.plot(xs_scaled, ys_scaled,
                color="black", lw=0.8, alpha=0.7, zorder=2)

        # 红色点: 起始位置 (iter 0)
        ax.plot(xs_scaled[0], ys_scaled[0],
                "ro", markersize=3, zorder=4)

        # 蓝色点: 终止位置 (iter T-1)
        ax.plot(xs_scaled[-1], ys_scaled[-1],
                "bo", markersize=3, zorder=4)

    # ---- 美化 ----
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    # 视图范围: 默认全图; 若指定 zoom_pad, 则聚焦到 CP bbox + pad 像素
    if zoom_pad is not None and zoom_pad >= 0:
        all_xs = cps_history[..., 1] if coord_yx else cps_history[..., 0]
        all_ys = cps_history[..., 0] if coord_yx else cps_history[..., 1]
        xmin, xmax = float(all_xs.min()) - zoom_pad, float(all_xs.max()) + zoom_pad
        ymin, ymax = float(all_ys.min()) - zoom_pad, float(all_ys.max()) + zoom_pad
        ax.set_xlim(max(0, xmin), min(W, xmax))
        ax.set_ylim(min(H, ymax), max(0, ymin))  # y 反向
    else:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)          # y 向下 (与 imshow origin='upper' 一致)

    if title is None:
        title = (f"控制点移动轨迹 (放大 {scale:.0f}×)：{T} 次迭代, 共 {K} 个 CP")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.3)

    # 图例
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="black", linewidth=0.8,
               label="移动轨迹 (放大 %g×)" % scale),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=6, linestyle="None", label="起始位置 (iter 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
               markersize=6, linestyle="None", label="终止位置 (iter %d)" % (T - 1)),
        Line2D([0], [0], color="0.5", linewidth=1.5, label="目标版图轮廓"),
    ]
    ax.legend(handles=legend_elems, loc="upper right",
              fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已保存: {out_path}")
    print(f"     迭代数 T={T}, CP 总数 K={K}, 跟踪数={len(idx)}, 选取方式={sel_mode}")
    sel_disp = total_disp[idx]
    print(f"     被跟踪 CP 总位移: min={sel_disp.min():.3f}, "
          f"max={sel_disp.max():.3f}, mean={sel_disp.mean():.3f} pixel")


# ----------------------- Main ----------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cp_dir", type=str,
        default="outputs/opc/DPS/BS/0.8_BS_工字型/curves/cp_history",
        help="包含 cp_iter_XXX.txt 的目录",
    )
    ap.add_argument(
        "--target_mask", type=str, default=None,
        help="目标版图 txt 路径, 未指定时自动从 cp_dir 向上找 target_mask.txt",
    )
    ap.add_argument(
        "--no_background", action="store_true",
        help="不叠加目标版图作背景",
    )
    ap.add_argument(
        "--step", type=int, default=1,
        help="每隔 step 个 CP 取一个 (默认 4)",
    )
    ap.add_argument(
        "--cp_indices", type=int, nargs="*", default=None,
        help="手动指定要跟踪的 CP 下标 (0-based)",
    )
    ap.add_argument(
        "--top_k", type=int, default=None,
        help="只画位移最大的前 K 个 CP",
    )
    ap.add_argument(
        "--out", type=str, default=None,
        help="输出 PNG 路径 (默认存到 cp_dir 上一级 curves/cp_trajectory.png)",
    )
    ap.add_argument(
        "--title", type=str, default=None,
        help="自定义图标题",
    )
    ap.add_argument(
        "--zoom_pad", type=float, default=None,
        help="聚焦到 CP bbox + 该像素边距 (例如 10); 不指定则显示全图",
    )
    args = ap.parse_args()

    cp_dir = Path(args.cp_dir).resolve()
    if not cp_dir.is_dir():
        raise SystemExit(
            f"目录不存在: {cp_dir}\n"
            f"提示: 需要用更新后的 demo_meef.py 重新跑一次优化, "
            f"才会自动生成 cp_history/ 目录"
        )

    out_path = Path(args.out).resolve() if args.out \
        else cp_dir.parent / "cp_trajectory.png"

    # 加载 CP 历史
    cps_history, iter_ids = load_cp_history(cp_dir)
    print(f"加载 {len(iter_ids)} 个迭代文件, 迭代号={iter_ids[:5]}...{iter_ids[-3:]}, "
          f"shape={cps_history.shape}")

    # 加载目标版图 (默认路径: cp_dir/../../target_mask.txt, 即 run 根目录)
    target_mask = None
    if not args.no_background:
        if args.target_mask:
            mask_path = Path(args.target_mask).resolve()
        else:
            # cp_dir = <run>/curves/cp_history, run = cp_dir.parents[2]
            mask_path = cp_dir.parents[2] / "target_mask.txt"
        target_mask = load_target_mask(mask_path)
        if target_mask is not None:
            print(f"叠加背景: {mask_path} shape={target_mask.shape}")
        else:
            print(f"[warn] 未找到 target_mask, 将画无背景版本 (尝试路径: {mask_path})")

    plot_cp_trajectory(
        cps_history,
        out_path=out_path,
        target_mask=target_mask,
        track_indices=args.cp_indices,
        step=args.step,
        top_k=args.top_k,
        title=args.title,
        zoom_pad=args.zoom_pad,
    )


if __name__ == "__main__":
    main()
