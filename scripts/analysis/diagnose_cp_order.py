"""诊断: 为什么 GDS / 拟合曲线上有"穿过中间的实线"?

把 main + sraf 的 CP 按它们在文件里的顺序画出来:
    - 每个块用一种颜色; CP 用散点 + 数字编号; CP 之间用细线按顺序连接
    - 自动检测自交叉 (Bentley-Ottmann 简版): 任意两条非相邻边是否有交点
    - 自交叉的块用粗红边框标出, 加上 [BAD] 标签
输出: PNG 图 + 终端打印每个有问题的块的 CP 坐标
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "outputs/opc" / "MEEF_pipeline" / "BS" / \
    "matrix_0011_k=5_sym=none_srafArc=5.0_BS_pipeline_test"
MAIN_PATH = RESULT_DIR / "WEPE最优的控制点坐标.txt"
SRAF_PATH = RESULT_DIR / "sraf_cps.txt"
BG_LSM = ROOT / "outputs/opc" / "CTM和levelset图像" / "Ls_mask" / "ls_imagematrix_0011.txt"


# ---------- 解析 ----------
_PT_RE = re.compile(r"\[\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\]")


def parse_main(p: Path) -> List[np.ndarray]:
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        pts = [(float(m.group(1)), float(m.group(2)))
               for m in _PT_RE.finditer(line)]
        if pts:
            out.append(np.asarray(pts))
    return out


def parse_sraf(p: Path) -> List[np.ndarray]:
    out, cur = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if cur:
                out.append(np.asarray(cur, float))
                cur = []
            continue
        a, b = s.split()[:2]
        cur.append([float(a), float(b)])
    if cur:
        out.append(np.asarray(cur, float))
    return out


# ---------- 自交叉检测 ----------
def _seg_intersect(p1, p2, p3, p4) -> bool:
    """两条线段是否严格相交 (不算端点共享)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return False


def has_self_intersection(pts: np.ndarray) -> bool:
    """闭合多边形是否自交叉 (O(n^2), 调试足够)."""
    n = len(pts)
    if n < 4:
        return False
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for (i, e1), (j, e2) in combinations(enumerate(edges), 2):
        # 跳过相邻边和共享端点的边 (含闭合处的相邻)
        if abs(i - j) <= 1 or abs(i - j) == n - 1:
            continue
        if _seg_intersect(e1[0], e1[1], e2[0], e2[1]):
            return True
    return False


# ---------- 绘图 ----------
def plot_cps(ax, blocks, label_prefix, base_cmap):
    bad = []
    for i, c in enumerate(blocks):
        color = base_cmap(i / max(1, len(blocks) - 1))
        is_bad = has_self_intersection(c)
        if is_bad:
            bad.append((i, c))

        # 闭合连线
        closed = np.vstack([c, c[:1]])
        lw = 2.5 if is_bad else 0.8
        edgecolor = "red" if is_bad else color
        ax.plot(closed[:, 1], closed[:, 0], "-",
                color=edgecolor, lw=lw,
                alpha=0.95 if is_bad else 0.6)
        ax.scatter(c[:, 1], c[:, 0],
                   s=18 if is_bad else 8,
                   c=[color], edgecolors="white", linewidths=0.4, zorder=5)

        if is_bad:
            cx, cy = c[:, 1].mean(), c[:, 0].mean()
            ax.text(cx, cy, f"BAD\n{label_prefix}{i}",
                    color="red", fontsize=7, ha="center", va="center",
                    weight="bold",
                    bbox=dict(boxstyle="round", fc="white",
                              ec="red", alpha=0.85))
    return bad


def main():
    mains = parse_main(MAIN_PATH)
    srafs = parse_sraf(SRAF_PATH)
    print(f"main blocks : {len(mains)}, total CP {sum(len(c) for c in mains)}")
    print(f"sraf blocks : {len(srafs)}, total CP {sum(len(c) for c in srafs)}")

    # 背景
    bg = np.loadtxt(BG_LSM) if BG_LSM.exists() else np.zeros((257, 257))

    fig, ax = plt.subplots(figsize=(11, 11), dpi=120)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)

    bad_main = plot_cps(ax, mains, "M", plt.cm.spring)
    bad_sraf = plot_cps(ax, srafs, "S", plt.cm.cool)

    ax.set_xlim(-0.5, bg.shape[1] - 0.5)
    ax.set_ylim(bg.shape[0] - 0.5, -0.5)
    ax.set_axis_off()
    ax.set_title(
        f"CP order diagnostic | "
        f"main bad: {len(bad_main)}/{len(mains)}, "
        f"sraf bad: {len(bad_sraf)}/{len(srafs)}"
    )
    plt.tight_layout()
    out = RESULT_DIR / "diagnose_cp_order.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out}")

    # 把有问题的块单独打印
    print("\n========== self-intersecting blocks ==========")
    for idx, c in bad_main:
        print(f"  [main {idx}]  n_cp={len(c)}")
    for idx, c in bad_sraf:
        n_min = min(len(c), 8)
        first = ", ".join([f"({p[0]:.2f},{p[1]:.2f})" for p in c[:n_min]])
        print(f"  [sraf {idx}]  n_cp={len(c)}  first {n_min}: {first}")

    if not bad_main and not bad_sraf:
        print("  (no self-intersecting CPs detected)")


if __name__ == "__main__":
    main()
