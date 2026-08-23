"""
光学系统参数配置。

本文件是 "唯一参数入口"：所有光刻物理参数集中定义于此，
numpy / torch 双后端共享同一份配置。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class SourceConfig:
    """光源配置"""
    type: str = "annular"          # annular / conventional / dipole / quasar
    sigma_in: float = 0.6          # 内半径 (NA 归一化)
    sigma_out: float = 0.9         # 外半径
    oversampling: int =1         # 光源超采样倍数


@dataclass
class OpticsConfig:
    """投影物镜配置"""
    na: float = 1.35               # 数值孔径
    wavelength_nm: float = 193.0   # 波长 [nm]
    aberrations: dict = field(default_factory=dict)  # Zernike 系数


@dataclass
class ResistConfig:
    """光刻胶参数"""
    threshold: float = 0.15        # 显影阈值
    alpha: float = 85.0            # Sigmoid 陡峭度


@dataclass
class MaskConfig:
    """Mask 配置"""
    pixel_size_nm: float = 4.0     # 像素尺寸 [nm]
    image_name: str = ""           # BMP 输入文件名 (可选)


@dataclass
class OpticalConfig:
    """
    完整光学系统配置，所有物理参数均定义于此。

    Attributes
    ----------
    mask     : mask / 网格参数
    optics   : 投影物镜参数 (NA, 波长, 像差)
    source   : 光源参数
    resist   : 光刻胶 / 显影参数
    grid_n   : 仿真网格大小 (N×N), None 表示由 mask 自动决定
    socs_k   : SOCS 核保留数, None 表示自动 (4nm→50, 6nm→60)
    """
    mask: MaskConfig = field(default_factory=MaskConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    resist: ResistConfig = field(default_factory=ResistConfig)
    grid_n: Optional[int] = None
    socs_k: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "OpticalConfig":
        """从 YAML 文件加载配置。"""
        with open(path, "r") as f:
            d = yaml.safe_load(f)

        def _populate(dataclass_type, source_dict, prefix=""):
            kwargs = {}
            for field_name in dataclass_type.__dataclass_fields__:
                key = f"{prefix}{field_name}" if prefix else field_name
                if key in source_dict:
                    kwargs[field_name] = source_dict[key]
            return dataclass_type(**kwargs) if kwargs else dataclass_type()

        return cls(
            mask=_populate(MaskConfig, d, "mask_"),
            optics=_populate(OpticsConfig, d, "optics_"),
            source=_populate(SourceConfig, d, "source_"),
            resist=_populate(ResistConfig, d, "resist_"),
            grid_n=d.get("grid_n", None),
            socs_k=d.get("socs_k", None),
        )

    @classmethod
    def default(cls) -> "OpticalConfig":
        """返回 ArF 浸没式光刻的默认参数 (NA=1.35, 环形光源)。"""
        return cls()


# ========== 预设配置 ==========
# 可以通过 to_dict() 保存为 YAML，或直接传给 OpticalConfig 构造

DEFAULT_193NM_ARF = OpticalConfig.default()
"""默认 ArF 浸没式光刻: 193nm, NA=1.35, 环形光源 sigma=0.6-0.9"""

LOW_NA_DRY = OpticalConfig(
    optics=OpticsConfig(na=0.75, wavelength_nm=193.0),
    source=SourceConfig(type="conventional", sigma_in=0.3, sigma_out=0.7),
)
"""低 NA 干法光刻"""

EUV_135 = OpticalConfig(
    optics=OpticsConfig(na=0.33, wavelength_nm=13.5),
    source=SourceConfig(type="annular", sigma_in=0.4, sigma_out=0.8),
    resist=ResistConfig(threshold=0.3, alpha=100),
    mask=MaskConfig(pixel_size_nm=2.0),
)
"""EUV 光刻示例 (13.5nm)"""
