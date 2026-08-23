"""诊断: GDS 文件里多边形的顶点存储顺序是否真的"绕一圈".

做的事:
    1) 读回 optimized_mask.gds 的所有多边形
    2) 对每个多边形:
       - 检测顶点序列是否自交叉 (按存储顺序连首尾)
       - 检测绕向 (signed area > 0 = CCW, < 0 = CW); 同一类多边形混着两种绕向也是问题
    3) 把所有"自交叉"或"绕向异常"的多边形单独画出来 + 序号 + 箭头
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import gdstk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GDS_PATH = ROOT / "outputs/opc" / "MEEF_pipeline" / "BS" / \
    "matrix_0011_k=5_sym=none_srafArc=5.0_BS_pipeline_test" / "optimized_mask.gds"
OUT_DIR = GDS_PATH.parent


# ---------------- 自交叉检测 ----------------
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def seg_intersect(p1, p2, p3, p4) -> bool:
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return ((d1 > 0 > d2) or (d1 < 0 < d2)) and \
           ((d3 > 0 > d4) or (d3 < 0 < d4))


def has_self_intersect(pts: np.ndarray) -> bool:
    n = len(pts)
    if n < 4:
        return False
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for (i, e1), (j, e2) in combinations(enumerate(edges), 2):
        if abs(i - j) <= 1 or abs(i - j) == n - 1:
            continue
        if seg_intersect(e1[0], e1[1], e2[0], e2[1]):
            return True
    return False


def signed_area(pts: np.ndarray) -> float:
    """signed area; >0 = CCW, <0 = CW (在 +y up 坐标系下)."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def main():
    lib = gdstk.read_gds(str(GDS_PATH))
    cell = lib.top_level()[0]
    polys = cell.polygons
    print(f"GDS file        : {GDS_PATH}")
    print(f"#polygons       : {len(polys)}")

    layers = sorted(set(p.layer for p in polys))
    print(f"layers          : {layers}")

    # 按 layer 分别诊断
    by_layer: dict = {}
    for p in polys:
        by_layer.setdefault(p.layer, []).append(p.points)

    print()
    bad_polys = []
    for layer, pts_list in by_layer.items():
        n = len(pts_list)
        si = [has_self_intersect(p) for p in pts_list]
        areas = np.array([signed_area(p) for p in pts_list])
        n_si = sum(si)
        n_ccw = int(np.sum(areas > 0))
        n_cw = int(np.sum(areas < 0))
        print(f"layer {layer}: {n} polys | self_intersect={n_si} | "
              f"CCW={n_ccw} CW={n_cw}")
        # 画出有问题的
        for i, (p, b) in enumerate(zip(pts_list, si)):
            if b:
                bad_polys.append((layer, i, p))

    if not bad_polys:
        print("\n所有 GDS 多边形顶点都是按 '绕一圈' 顺序存的, 没有自交叉.")
        return

    print(f"\n发现 {len(bad_polys)} 个自交叉多边形, 单独画出来:")
    n_show = min(12, len(bad_polys))
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=110)
    axes = np.atleast_2d(axes)

    for k in range(n_show):
        layer, idx, pts = bad_polys[k]
        ax = axes[k // cols, k % cols]
        ax.set_aspect("equal")

        # 主图: 顶点按顺序连线
        closed = np.vstack([pts, pts[:1]])
        ax.plot(closed[:, 0], closed[:, 1], "-", color="#3498db",
                lw=1.0, alpha=0.85)
        ax.scatter(pts[:, 0], pts[:, 1], s=14, c="blue",
                   edgecolors="white", linewidths=0.4, zorder=5)
        # 起点用大红圆标
        ax.scatter(pts[0, 0], pts[0, 1], s=80, c="red",
                   marker="*", zorder=6, label=f"start (P0)")
        # 标几个中间序号 (太多会糊)
        for j in range(0, len(pts), max(1, len(pts) // 8)):
            ax.annotate(f"{j}", (pts[j, 0], pts[j, 1]),
                        fontsize=7, color="black",
                        textcoords="offset points", xytext=(3, 3))
        # 几条箭头看方向
        n = len(pts)
        for j in range(0, n, max(1, n // 6)):
            j2 = (j + 1) % n
            ax.annotate(
                "", xy=(pts[j2, 0], pts[j2, 1]),
                xytext=(pts[j, 0], pts[j, 1]),
                arrowprops=dict(arrowstyle="->", color="orange", lw=0.8),
            )

        ax.set_title(f"layer {layer}  poly#{idx}  n_verts={len(pts)}",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

    for k in range(n_show, rows * cols):
        axes[k // cols, k % cols].set_axis_off()

    plt.tight_layout()
    out = OUT_DIR / "diagnose_gds_polygon_order.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")

    # 把第一个 BAD 多边形的顶点坐标全部打印
    layer, idx, pts = bad_polys[0]
    print(f"\n========== 第一个 BAD 多边形 (layer {layer}, poly#{idx}) 的顶点 ==========")
    for j, p in enumerate(pts):
        print(f"  [{j:3d}]  ({p[0]:.4f}, {p[1]:.4f})")


if __name__ == "__main__":
    main()
