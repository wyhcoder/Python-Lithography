"""
根据控制点坐标文件画 B-spline 拟合曲线，叠加目标版图轮廓。

用法:
    python scripts/analysis/plot_bspline_curve.py \
        --cps outputs/opc/DPS/BS/0.8_BS_工字型/wEPE最优的控制点坐标.txt \
        --target_mask outputs/opc/DPS/BS/0.8_BS_工字型/target_mask.txt \
        --out outputs/opc/DPS/BS/0.8_BS_工字型/bspline_wepe_optimal.png

    # 省略 --target_mask 时自动从 cps 同目录找 target_mask.txt
    # 省略 --out 时自动存到 cps 同目录下 bspline_curve.png
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils_model.demo_parametric import ParametricDemo

mpl.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei",
    "Microsoft YaHei", "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_cps(cps_path: Path) -> np.ndarray:
    """解析 [y x] [y x] ... 格式的控制点坐标文件，返回 (N, 2) [y, x]。"""
    txt = cps_path.read_text(encoding="utf-8").strip()
    pairs = re.findall(r"\[([^\]]+)\]", txt)
    if not pairs:
        raise ValueError(f"未在 {cps_path} 中找到 [y x] 格式的坐标对")
    return np.array([[float(v) for v in p.split()] for p in pairs])


def load_target_mask(mask_path: Path) -> np.ndarray:
    return np.loadtxt(mask_path)


def main():
    ap = argparse.ArgumentParser(description="画 B-spline 拟合曲线 + 目标版图轮廓")
    ap.add_argument("--cps", required=True, help="控制点坐标 txt 路径 ([y x] 格式)")
    ap.add_argument("--target_mask", default=None,
                    help="目标版图 txt 路径 (默认从 cps 同目录找 target_mask.txt)")
    ap.add_argument("--out", default=None,
                    help="输出 PNG 路径 (默认存到 cps 同目录 bspline_curve.png)")
    ap.add_argument("--title", default="Optimization Result vs Target Contour")
    ap.add_argument("--num_points", type=int, default=200,
                    help="B-spline 采样点数 (默认 200)")
    ap.add_argument("--smoothing", type=float, default=0.3,
                    help="B-spline 平滑参数 (默认 0.3)")
    args = ap.parse_args()

    cps_path = Path(args.cps).resolve()
    cps_yx = load_cps(cps_path)  # (N, 2) [y, x]
    print(f"加载 {len(cps_yx)} 个控制点, 前3个 [y,x]: {cps_yx[:3]}")

    # 目标版图
    if args.target_mask:
        mask_path = Path(args.target_mask).resolve()
    else:
        mask_path = cps_path.parent / "target_mask.txt"
    target_mask = load_target_mask(mask_path)
    print(f"目标版图: {mask_path} shape={target_mask.shape}")

    # B-spline 拟合
    parametric = ParametricDemo("BS", target_mask)
    curve = parametric.b_spile([cps_yx], args.smoothing,
                               num_points=args.num_points)

    # 画图
    fig, ax = plt.subplots(figsize=(8, 8))

    # 目标版图轮廓
    ax.contour(target_mask, levels=[0.5], colors="gray",
               linewidths=1.5, alpha=0.7)

    # B-spline 曲线 [y, x] -> plot (x, y), 闭合
    cv = curve[0]
    xs = np.append(cv[:, 1], cv[0, 1])
    ys = np.append(cv[:, 0], cv[0, 0])
    ax.plot(xs, ys, "b-", lw=2, label="B-spline curve")

    # 控制点 [y, x] -> plot (x, y)
    ax.plot(cps_yx[:, 1], cps_yx[:, 0], "r.", ms=5, label="Control Points")

    ax.set_aspect("equal")
    ax.set_xlim(0, target_mask.shape[1])
    ax.set_ylim(target_mask.shape[0], 0)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_title(args.title)
    ax.legend(loc="upper right")
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.out).resolve() if args.out \
        else cps_path.parent / "bspline_curve.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已保存: {out_path}")


if __name__ == "__main__":
    main()
