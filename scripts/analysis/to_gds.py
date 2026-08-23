"""把"优化后的曲线掩膜 (CP -> 闭合 B-spline)"导出为 GDS 文件.

特点:
    - 主图形 + SRAF 一起导出 (B 方案)
    - 像素 -> 物理坐标 (默认 pixel_size = 6 nm) + y 翻转 (GDS 习惯坐标)
    - 每条 contour 用周期 B-spline 插值离散到密集多边形顶点 (主图形 400 点 / SRAF 200 点)
    - 主图形 layer=1, SRAF layer=2
    - 不依赖 simulator/装配链路, 只读取 setup 阶段已落盘的 cps txt

输入:
    main_cps_path: 主图形 CP, 格式可为
        - "[y x] [y x] ..." 一行一条 contour (set_meef 输出格式, 例 WEPE最优的控制点坐标.txt)
        - "y x" 一行一个点 + "# main contour N" 分组 (本管线 main_cps.txt 格式)
    sraf_cps_path: SRAF CP, 走 "# sraf block N" 分组 (本管线 sraf_cps.txt 格式)

用法 (默认指向 MEEF_pipeline 那份结果):
    python3 -m scripts.analysis.export_cps_to_gds
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import gdstk
from scipy.interpolate import splprep, splev


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 默认输入输出 (按当前 yaml 的产物指向 MEEF_pipeline)
# ---------------------------------------------------------------------------
DEFAULT_RESULT_DIR = ROOT / "outputs/opc" / "MEEF_pipeline" / "BS" / \
    "matrix_0011_k=5_sym=none_srafArc=5.0_BS_pipeline_test"

# 主图形 CP: 优先选 WEPE 最优 (优化结果); 退化时用 setup 阶段的 main_cps.txt
DEFAULT_MAIN_CPS = DEFAULT_RESULT_DIR / "WEPE最优的控制点坐标.txt"
DEFAULT_MAIN_CPS_FALLBACK = DEFAULT_RESULT_DIR / "main_cps.txt"
DEFAULT_SRAF_CPS = DEFAULT_RESULT_DIR / "sraf_cps.txt"
DEFAULT_OUT_GDS = DEFAULT_RESULT_DIR / "optimized_mask.gds"

# 物理参数 (与 yaml 一致)
PIXEL_SIZE_NM = 6.0
MASK_HEIGHT_PX = 256                # y 翻转用; 与 target_mask shape 一致

# B-spline 离散精度
NUM_PTS_MAIN = 400
NUM_PTS_SRAF = 200

# Layer
LAYER_MAIN = 1
LAYER_SRAF = 2
DATATYPE = 0

# GDS 标准单位 (1 user unit = 1 µm; 精度 1 nm)
GDS_UNIT = 1e-6
GDS_PRECISION = 1e-9


# ---------------------------------------------------------------------------
# 解析 CP 文件 (兼容两种格式)
# ---------------------------------------------------------------------------
_PT_BRACKET_RE = re.compile(r"\[\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\]")


def parse_cps_txt(path: Path) -> List[np.ndarray]:
    """解析 CP 文件, 返回 list[ndarray(N, 2)] (按 [y, x] 排列).

    支持两种格式:
        (a) 每行一条 contour, 包含若干 [y x]:
                [159.75 97.16] [161.80 93.53] ...
            (set_meef.py 在 best_*_cps 文件用的格式)
        (b) 行注释分组 + 每行一个点:
                # sraf block 0
                0.0000 63.0000
                3.0000 66.2354
                ...
                # sraf block 1
                ...
            (本管线 setup.py 在 main_cps.txt / sraf_cps.txt 用的格式)
    """
    contours: List[np.ndarray] = []
    has_bracket_in_first_data_line = False
    lines = path.read_text(encoding="utf-8").splitlines()

    # 先扫一眼判断格式
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "[" in s:
            has_bracket_in_first_data_line = True
        break

    if has_bracket_in_first_data_line:
        # 格式 (a): 每行一条 contour
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pts = [(float(m.group(1)), float(m.group(2)))
                   for m in _PT_BRACKET_RE.finditer(line)]
            if pts:
                contours.append(np.asarray(pts, dtype=np.float64))
    else:
        # 格式 (b): 注释分组 + 每行一个点
        cur: List[List[float]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if cur:
                    contours.append(np.asarray(cur, dtype=np.float64))
                    cur = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                cur.append([float(parts[0]), float(parts[1])])
        if cur:
            contours.append(np.asarray(cur, dtype=np.float64))

    return contours


# ---------------------------------------------------------------------------
# B-spline 闭合拟合
# ---------------------------------------------------------------------------
def fit_closed_bspline(pts_yx: np.ndarray, smoothing: float = 0.3,
                       num_points: int = 400) -> np.ndarray:
    """周期 B-spline 拟合, 返回 (num_points, 2) 的 [y, x] 密集采样.

    点数 < 4 时直接返回原点 (B-spline 拟合最少需要 4 个控制点).
    """
    if len(pts_yx) < 4:
        return pts_yx
    y, x = pts_yx[:, 0], pts_yx[:, 1]
    tck, _ = splprep([x, y], s=smoothing, per=True)
    u = np.linspace(0.0, 1.0, num_points, endpoint=False)
    x_fit, y_fit = splev(u, tck)
    return np.stack([y_fit, x_fit], axis=1)


# ---------------------------------------------------------------------------
# 像素 -> GDS 物理坐标 (µm)
# ---------------------------------------------------------------------------
def pix_to_gds_um(curve_yx: np.ndarray,
                  pixel_size_nm: float,
                  mask_height_px: int) -> np.ndarray:
    """像素 [y, x] -> GDS [x, y] (µm).

    - x_gds = x_pix * pixel_size_nm / 1000
    - y_gds = (mask_height_px - y_pix) * pixel_size_nm / 1000   ← y 翻转 (图像 y 向下 → GDS y 向上)
    - 转换到 µm (因为 GDS user unit 默认 1 µm)
    """
    pixel_size_um = pixel_size_nm * 1e-3
    x_gds = curve_yx[:, 1] * pixel_size_um
    y_gds = (mask_height_px - curve_yx[:, 0]) * pixel_size_um
    return np.stack([x_gds, y_gds], axis=1)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def export_cps_to_gds(
    main_cps_path: Path,
    sraf_cps_path: Path,
    out_gds_path: Path,
    pixel_size_nm: float = PIXEL_SIZE_NM,
    mask_height_px: int = MASK_HEIGHT_PX,
    num_pts_main: int = NUM_PTS_MAIN,
    num_pts_sraf: int = NUM_PTS_SRAF,
    layer_main: int = LAYER_MAIN,
    layer_sraf: int = LAYER_SRAF,
    cell_name: str = "OPTIMIZED_MASK",
):
    """把 main + sraf 的曲线掩膜写入 GDS 文件."""

    # 1. 解析
    main_cps = parse_cps_txt(main_cps_path)
    sraf_cps = parse_cps_txt(sraf_cps_path) if sraf_cps_path.exists() else []
    print(f"[parse] main: {len(main_cps)} contours, "
          f"total {sum(len(c) for c in main_cps)} CP")
    print(f"[parse] sraf: {len(sraf_cps)} blocks, "
          f"total {sum(len(c) for c in sraf_cps)} CP")

    # 2. 创建 GDS Library + Cell
    lib = gdstk.Library(unit=GDS_UNIT, precision=GDS_PRECISION)
    cell = lib.new_cell(cell_name)

    # 3. 主图形: 闭合 B-spline -> Polygon (layer 1)
    n_main_polys = 0
    for c in main_cps:
        curve = fit_closed_bspline(c, num_points=num_pts_main)
        if len(curve) < 3:
            continue
        verts = pix_to_gds_um(curve, pixel_size_nm, mask_height_px)
        poly = gdstk.Polygon(verts, layer=layer_main, datatype=DATATYPE)
        cell.add(poly)
        n_main_polys += 1

    # 4. SRAF: 每块一条闭合 B-spline -> Polygon (layer 2)
    n_sraf_polys = 0
    for c in sraf_cps:
        if len(c) < 4:
            continue
        curve = fit_closed_bspline(c, num_points=num_pts_sraf)
        if len(curve) < 3:
            continue
        verts = pix_to_gds_um(curve, pixel_size_nm, mask_height_px)
        poly = gdstk.Polygon(verts, layer=layer_sraf, datatype=DATATYPE)
        cell.add(poly)
        n_sraf_polys += 1

    # 5. 写文件
    out_gds_path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(out_gds_path))

    # 6. 报告
    extent_um = mask_height_px * pixel_size_nm * 1e-3
    print(f"[gds]  main polygons = {n_main_polys}  (layer {layer_main})")
    print(f"[gds]  sraf polygons = {n_sraf_polys}  (layer {layer_sraf})")
    print(f"[gds]  cell extent ~= {extent_um:.3f} x {extent_um:.3f} µm")
    print(f"[gds]  written to    : {out_gds_path}")


def main():
    main_path = DEFAULT_MAIN_CPS if DEFAULT_MAIN_CPS.exists() else DEFAULT_MAIN_CPS_FALLBACK
    if not main_path.exists():
        raise FileNotFoundError(
            f"找不到主图形 CP 文件: {DEFAULT_MAIN_CPS} 或 {DEFAULT_MAIN_CPS_FALLBACK}"
        )
    print(f"[input] main CPs: {main_path}")
    print(f"[input] sraf CPs: {DEFAULT_SRAF_CPS}")
    export_cps_to_gds(
        main_cps_path=main_path,
        sraf_cps_path=DEFAULT_SRAF_CPS,
        out_gds_path=DEFAULT_OUT_GDS,
    )


if __name__ == "__main__":
    main()
