"""
Illumination（照明源）模型。

支持的 source 类型（SourceType）：
- CONVENTIONAL : 圆形（实心圆盘），半径 = sigma_out
- ANNULAR      : 环形，sigma_in <= rho <= sigma_out
- DIPOLE       : 偶极（沿 ±x 两瓣），在 ANNULAR 基础上叠加角度掩膜
- QUASAR       : 四极（45°/135°/-135°/-45° 四瓣）

约定：
- source 平面坐标使用归一化频率 sigma = f / f_max（f_max = NA / lambda）。
- 通过 grid.Mask 提供的中心化频率轴 Fx_1d / Fy_1d 截取出 compact source 坐标轴；
  之后在 cell 内做 upsample 子采样并面积平均，避免边界锯齿/混叠。
- 输出有两份：
    self.source_map         : compact 网格上的强度分布（cell average）
    self.source_weight_map  : source_map * cell_area，并归一化到总和为 1，
                              便于后续 SOCS / Abbe 求和直接当权重使用。
"""
from __future__ import annotations

import math
from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import Grid
from .pupil import Pupil


class SourceType(Enum):
    """支持的光源类型。"""

    CONVENTIONAL = "conventional"
    ANNULAR = "annular"
    DIPOLE = "dipole"
    QUASAR = "quasar"


# ---------- 通用工具 ----------

def _smooth_step(width: float, x: torch.Tensor) -> torch.Tensor:
    """
    平滑阶跃：当 x >= 0 时趋近 1，x < 0 时趋近 0。
    width == 0 退化为硬阈值；width > 0 时用 erf 平滑过渡。
    """
    if width == 0.0:
        return (x >= 0).to(x.dtype)

    width = max(float(width), 1e-12)
    return 0.5 * (1.0 + torch.erf(x / width))


def _angle_diff(theta: torch.Tensor, center: float) -> torch.Tensor:
    """返回 theta 到 center 的最短角差，取值范围 [-pi, pi]。"""
    diff = theta - center
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def _build_lobe_mask(
    theta: torch.Tensor,
    centers: List[float],
    half_width: float,
    smoothing: float,
) -> torch.Tensor:
    """
    多瓣角度掩膜。

    每一瓣以 `centers[i]` 为中心、半宽 `half_width`（弧度），
    用 _smooth_step 做软边过渡。多瓣之间取最大值，保证两瓣交叠区也是 1。
    """
    if not centers:
        return torch.zeros_like(theta)

    mask = torch.zeros_like(theta)
    for c in centers:
        d = torch.abs(_angle_diff(theta, c))
        # half_width - d >= 0 时位于瓣内
        lobe = _smooth_step(smoothing, half_width - d)
        mask = torch.maximum(mask, lobe)
    return mask


class Illumination(nn.Module):
    """根据归一化频率轴构造 compact source map。"""

    def __init__(
        self,
        pupil: Pupil,
        grid: Grid,
        source_type: SourceType,
        sigma_in: float = 0.6,
        sigma_out: float = 0.9,
        smoothing: float = 0.0,
        upsample: int = 10,
        lobe_half_width_deg: float = 30.0,
        angular_smoothing_deg: float = 0.0,
    ):
        """
        参数
        ----
        pupil : 已经构造好的 Pupil 实例（提供 NA / lambda）。
        grid : Grid 实例（提供 Fx_1d / Fy_1d 频率轴）。
        source_type : SourceType 枚举值。
        sigma_in / sigma_out : 归一化半径，CONVENTIONAL 时 sigma_in 被忽略。
        smoothing : 径向 erf 平滑宽度（同 sigma 单位），0 表示硬边界。
        upsample : 每个 cell 内 sub-pixel 采样数，>=1。
        lobe_half_width_deg : DIPOLE/QUASAR 单瓣半角宽度（度）。
        angular_smoothing_deg : 角度方向 erf 平滑宽度（度）。
        """
        super().__init__()
        if isinstance(source_type, str):
            source_type = SourceType(source_type)

        if not (0.0 <= sigma_in < sigma_out):
            raise ValueError(
                f"非法 sigma 设置：sigma_in={sigma_in}, sigma_out={sigma_out}"
            )
        if upsample < 1:
            raise ValueError("upsample 必须 >= 1")

        self.pupil = pupil
        self.grid = grid
        self.source_type = source_type
        self.sigma_in = float(sigma_in)
        self.sigma_out = float(sigma_out)
        self.smoothing = float(smoothing)
        self.upsample = int(upsample)
        self.lobe_half_width_rad = math.radians(float(lobe_half_width_deg))
        self.angular_smoothing_rad = math.radians(float(angular_smoothing_deg))

        self.real_dtype = pupil.real_dtype
        self.device = pupil.device

        # 占位
        self.source_fx: torch.Tensor
        self.source_fy: torch.Tensor
        self.source_map: torch.Tensor
        self.source_weight_map: torch.Tensor

        self._compute_source_map()

    # ---------- 紧支撑坐标轴 ----------

    def _compute_extent(self, step_x: float, step_y: float) -> Tuple[float, float]:
        """根据 sigma_out 与 smoothing 估计 source 在 sigma 平面的紧支撑半径。"""
        margin_x = max(2.0 * step_x, 3.0 * self.smoothing, 0.05)
        margin_y = max(2.0 * step_y, 3.0 * self.smoothing, 0.05)
        return self.sigma_out + margin_x, self.sigma_out + margin_y

    @staticmethod
    def _axis_edges_and_widths(
        axis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """由 cell center 轴恢复 cell 边界与宽度。支持单调一维轴。"""
        if axis.ndim != 1:
            raise ValueError("axis 必须为 1D 张量")
        if axis.numel() < 2:
            raise ValueError("axis 长度必须至少为 2")

        diffs = axis[1:] - axis[:-1]
        if torch.any(diffs <= 0):
            raise ValueError("axis 必须严格递增")

        inner_edges = 0.5 * (axis[:-1] + axis[1:])
        left_edge = axis[0] - 0.5 * diffs[0]
        right_edge = axis[-1] + 0.5 * diffs[-1]
        edges = torch.cat(
            [left_edge.unsqueeze(0), inner_edges, right_edge.unsqueeze(0)]
        )
        widths = edges[1:] - edges[:-1]
        return edges, widths

    @staticmethod
    def _subdivide_axis_from_edges(
        edges: torch.Tensor, upsample: int
    ) -> torch.Tensor:
        """以每个 coarse cell 为单位等分 upsample 份，返回子单元中心。"""
        device = edges.device
        dtype = edges.dtype

        left = edges[:-1]
        right = edges[1:]
        widths = right - left

        frac = (torch.arange(upsample, device=device, dtype=dtype) + 0.5) / float(
            upsample
        )
        centers = left[:, None] + widths[:, None] * frac[None, :]
        return centers.reshape(-1)

    def _extract_compact_axes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 grid 的频率轴中截取 compact source 坐标轴。"""
        f_max = self.pupil.NA / self.pupil.wave_length
        fx_norm_1d = self.grid.Fx_1d / f_max
        fy_norm_1d = self.grid.Fy_1d / f_max
        if fx_norm_1d.numel() < 2 or fy_norm_1d.numel() < 2:
            raise ValueError("归一化频率轴长度不足，无法构造 source map")

        step_x = torch.abs(fx_norm_1d[1] - fx_norm_1d[0]).item()
        step_y = torch.abs(fy_norm_1d[1] - fy_norm_1d[0]).item()
        extent_x, extent_y = self._compute_extent(step_x=step_x, step_y=step_y)

        max_abs_x = torch.max(torch.abs(fx_norm_1d)).item()
        max_abs_y = torch.max(torch.abs(fy_norm_1d)).item()
        tol_x = max(5.0 * step_x, 1e-12)
        tol_y = max(5.0 * step_y, 1e-12)
        if extent_x > max_abs_x + tol_x or extent_y > max_abs_y + tol_y:
            raise ValueError(
                "optics 提供的归一化频率范围不足以覆盖当前 source 支撑。"
                f" 需要至少 |fx|<={extent_x:.4f}, |fy|<={extent_y:.4f}，"
                f" 但当前仅有 |fx|<={max_abs_x:.4f}, |fy|<={max_abs_y:.4f}。"
                " 请检查 grid 频域范围、pixel size 或 sigma 设置。"
            )

        mask_x = torch.abs(fx_norm_1d) <= extent_x
        mask_y = torch.abs(fy_norm_1d) <= extent_y

        source_fx_norm = fx_norm_1d[mask_x]
        source_fy_norm = fy_norm_1d[mask_y]

        if source_fx_norm.numel() < 3 or source_fy_norm.numel() < 3:
            raise ValueError(
                "compact source 坐标轴过小，请检查 grid 分辨率、pixel_size 或 sigma 设置"
            )

        return source_fx_norm, source_fy_norm

    # ---------- pattern 生成 ----------

    def _generate_source_pattern(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """在归一化 sigma 平面坐标上生成 source 强度分布。"""
        r = torch.sqrt(x * x + y * y)
        theta = torch.atan2(y, x)

        if self.source_type == SourceType.CONVENTIONAL:
            return _smooth_step(self.smoothing, self.sigma_out - r)

        base_annulus = (
            _smooth_step(self.smoothing, self.sigma_out - r)
            * _smooth_step(self.smoothing, r - self.sigma_in)
        )

        if self.source_type == SourceType.ANNULAR:
            return base_annulus

        if self.source_type == SourceType.DIPOLE:
            angle_mask = _build_lobe_mask(
                theta=theta,
                centers=[0.0, math.pi],
                half_width=self.lobe_half_width_rad,
                smoothing=self.angular_smoothing_rad,
            )
            return base_annulus * angle_mask

        if self.source_type == SourceType.QUASAR:
            angle_mask = _build_lobe_mask(
                theta=theta,
                centers=[
                    math.pi / 4.0,
                    3.0 * math.pi / 4.0,
                    -3.0 * math.pi / 4.0,
                    -math.pi / 4.0,
                ],
                half_width=self.lobe_half_width_rad,
                smoothing=self.angular_smoothing_rad,
            )
            return base_annulus * angle_mask

        raise RuntimeError(f"未支持的 source_type: {self.source_type}")

    # ---------- 主流程 ----------

    def _compute_source_map(self) -> torch.Tensor:
        source_fx_norm, source_fy_norm = self._extract_compact_axes()
        edges_x, widths_x = self._axis_edges_and_widths(source_fx_norm)
        edges_y, widths_y = self._axis_edges_and_widths(source_fy_norm)

        if self.upsample == 1:
            hi_fx = source_fx_norm
            hi_fy = source_fy_norm
        else:
            hi_fx = self._subdivide_axis_from_edges(edges_x, self.upsample)
            hi_fy = self._subdivide_axis_from_edges(edges_y, self.upsample)

        # indexing="ij" 后 dim0 = fy, dim1 = fx
        hi_yy, hi_xx = torch.meshgrid(hi_fy, hi_fx, indexing="ij")
        source_hi = self._generate_source_pattern(hi_xx, hi_yy)

        if self.upsample == 1:
            source_map = source_hi
        else:
            source_hi_4d = source_hi.unsqueeze(0).unsqueeze(0)
            source_map = (
                F.avg_pool2d(
                    source_hi_4d,
                    kernel_size=self.upsample,
                    stride=self.upsample,
                )
                .squeeze(0)
                .squeeze(0)
            )

        source_map = source_map.to(self.real_dtype).clamp_min(0.0)
        if source_map.numel() == 0 or torch.max(source_map).item() <= 0.0:
            raise ValueError(
                "生成的 source_map 全为 0。请检查 source_type、sigma、smoothing 与 grid 坐标范围。"
            )

        cell_area = widths_y[:, None] * widths_x[None, :]
        source_weight_map = source_map * cell_area
        total_weight = source_weight_map.sum()
        if not torch.isfinite(total_weight) or total_weight.item() <= 0.0:
            raise ValueError("source_weight_map 总权重无效。请检查 source 参数与坐标构造。")

        # 归一化为概率分布（总和=1）
        source_weight_map = source_weight_map / total_weight


        # 用普通属性赋值，避免重复调用时 register_buffer 报错
        self.source_fx_norm = source_fx_norm
        self.source_fy_norm = source_fy_norm
        self.source_edges_x = edges_x
        self.source_edges_y = edges_y
        self.source_widths_x = widths_x
        self.source_widths_y = widths_y
        self.source_map = source_map
        self.source_weight_map = source_weight_map
        

        return source_map

    # ---------- 对外便捷接口 ----------

    def get_source(self) -> torch.Tensor:
        """返回 compact source map（cell average，未做总和归一）。"""
        return self.source_map

    def get_weights(self) -> torch.Tensor:
        """返回归一化的 source 权重图（总和 = 1，cell area 已合并）。"""
        return self.source_weight_map

    def get_axes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (source_fx, source_fy)，单位为归一化 sigma。"""
        return self.source_fx_norm, self.source_fy_norm

if __name__ == "__main__":
    grid = Grid(N=257, dx=4.0)

