"""
根据 SRAF 控制点坐标文件画 B-spline 拟合曲线，叠加目标版图轮廓。

用法:
    python scripts/analysis/plot_sraf_curves.py \
        --cps outputs/opc/MEEF_pipeline/BS/工字型_k=5_sym=none_srafArc=5.0_BS_pipeline_test_xy/sraf_cps.txt

    # 省略 --target_mask 时自动从 cps 同目录找 target_mask.txt
    # 省略 --out 时自动存到 cps 同目录 sraf_curves.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils_model.demo_parametric import ParametricDemo

mpl.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei",
    "Microsoft YaHei", "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_sraf_blocks(cps_path: Path):
    """解析 sraf_cps.txt, 返回 list of (N, 2) [y, x] ndarray。"""
    blocks = []
    current = []
    for line in cps_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current:
                blocks.append(np.array(current))
            current = []
        else:
            a, b = line.split()
            current.append([float(a), float(b)])  # [y, x]
    if current:
        blocks.append(np.array(current))
    return blocks


def main():
    ap = argparse.ArgumentParser(description="画 SRAF B-spline 拟合曲线 + 目标版图轮廓")
    ap.add_argument("--cps", required=True, help="sraf_cps.txt 路径 ([y x] 格式)")
    ap.add_argument("--target_mask", default=None,
                    help="目标版图 txt 路径 (默认从 cps 同目录找)")
    ap.add_argument("--out", default=None,
                    help="输出 PNG 路径 (默认存到 cps 同目录 sraf_curves.png)")
    ap.add_argument("--curve_color", default="blue", help="曲线颜色 (默认 blue)")
    ap.add_argument("--cp_color", default="red", help="控制点颜色 (默认 red)")
    ap.add_argument("--cp_size", type=int, default=3, help="控制点大小 (默认 3)")
    ap.add_argument("--num_points", type=int, default=200, help="B-spline 采样点数")
    ap.add_argument("--smoothing", type=float, default=0.3, help="B-spline 平滑参数")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    cps_path = Path(args.cps).resolve()
    blocks = load_sraf_blocks(cps_path)
    print(f"解析到 {len(blocks)} 个 SRAF block")

    # 目标版图
    if args.target_mask:
        mask_path = Path(args.target_mask).resolve()
    else:
        mask_path = cps_path.parent / "target_mask.txt"
    target_mask = np.loadtxt(mask_path) if mask_path.exists() else None

    # B-spline 拟合
    parametric = ParametricDemo(
        "BS", target_mask if target_mask is not None else np.zeros((300, 300))
    )

    # 画图
    fig, ax = plt.subplots(figsize=(10, 10))

    if target_mask is not None:
        ax.contour(target_mask, levels=[0.5], colors="gray",
                   linewidths=1.5, alpha=0.5)

    for cps in blocks:
        if len(cps) < 4:
            xs = np.append(cps[:, 1], cps[0, 1])
            ys = np.append(cps[:, 0], cps[0, 0])
            ax.plot(xs, ys, "-", color=args.curve_color, lw=1.5, alpha=0.8)
        else:
            try:
                curves = parametric.b_spile([cps], args.smoothing,
                                            num_points=args.num_points)
                cv = curves[0]  # [y, x]
                xs = np.append(cv[:, 1], cv[0, 1])
                ys = np.append(cv[:, 0], cv[0, 0])
                ax.plot(xs, ys, "-", color=args.curve_color, lw=1.5, alpha=0.8)
            except Exception:
                xs = np.append(cps[:, 1], cps[0, 1])
                ys = np.append(cps[:, 0], cps[0, 0])
                ax.plot(xs, ys, "-", color=args.curve_color, lw=1.5, alpha=0.8)
        # 控制点
        ax.plot(cps[:, 1], cps[:, 0], "o", color=args.cp_color,
                ms=args.cp_size, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    title = args.title or f"SRAF B-spline Curves ({len(blocks)} blocks)"
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.out).resolve() if args.out \
        else cps_path.parent / "sraf_curves.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已保存: {out_path}")


if __name__ == "__main__":
    main()
