import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class SystemParams:
    wavelength_nm: float
    pixel_size_nm: float

@dataclass
class MaskParams:
    image_name: str

@dataclass
class OpticsParams:
    na: float
    refractive_index: float
    aberrations: Dict[str, float] = field(default_factory=dict)

@dataclass
class SourceParams:
    type: str
    sigma_in: float
    sigma_out: float

@dataclass
class ResistParams:
    model: str
    threshold: float
    alpha: int

@dataclass
class MEEFParams:
    curve_type: str
    file_name: str
    simply_type: str
    val_simply: float
    pattern_name: str
    ifOPC: bool
    SRAF: bool
    # 控制点移动策略: "bisector"(角平分线一维位移) 或 "xy"(X/Y 解耦合成二维位移)
    move_strategy: str = "bisector"

@dataclass
class SRAF_widthParams:
    curve_type: str
    file_name: str
    pattern_name: str
    sraf_orders: bool
    sraf_simply: bool
    fix_sraf: bool
    epe_cost: bool
    pvband_cost: bool
    # MEEF 优化后主图形控制点 TXT 路径 (pvband_cost=True 时需填写)
    meef_opt_cps_path: str = ""


@dataclass
class MEEFPipelineParams:
    """meef_pipeline 段的 dataclass; 与 op_model.meef_pipeline.MEEFPipelineConfig 字段对齐."""
    pattern_name: str
    ls_mask_path: str
    file_name: str
    move_strategy: str 
    main_cp_interval: int = 5
    main_symmetry: str = "none"
    sraf_cp_arclen: float = 3.0
    sraf_min_cps: int = 8
    sraf_smoothing: float = 0.3
    sraf_curve_pts: int = 200
    sraf_min_area: int = 20
    msaa_level: int = 16
    curve_type: str = "BS"
    delta: float = 0.15
    iterations: int = 40
    dilate_radius: int = 2
    interval_line: int = 5
    interval_corner: int = 2
    mid_weight: float = 4.0
    other_weight: float = 1.0


@dataclass
class SimulationParameters:
    system: SystemParams
    mask: MaskParams
    optics: OpticsParams
    source: SourceParams
    resist: ResistParams
    meef: MEEFParams
    sraf: SRAF_widthParams
    # 新管线 (case_meef_pipeline.py 使用); 老入口可不填该段, 加载时给 None
    meef_pipeline: Optional[MEEFPipelineParams] = None

    @classmethod
    def from_yaml(cls, file_path: str) -> 'SimulationParameters':
        """Loads and parses all parameters from a YAML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        meef_pipeline_cfg = (
            MEEFPipelineParams(**config['meef_pipeline'])
            if 'meef_pipeline' in config else None
        )

        return cls(
            system=SystemParams(**config['system']),
            mask=MaskParams(**config['mask']),
            optics=OpticsParams(**config['optics']),
            source=SourceParams(**config['source']),
            resist=ResistParams(**config['resist']),
            meef=MEEFParams(**config['meef']),
            sraf=SRAF_widthParams(**config['sraf']),
            meef_pipeline=meef_pipeline_cfg,
        )