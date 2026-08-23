"""
仿真配置（dataclass + YAML 加载）。

YAML 字段约定见 `configs/litho_system.yaml`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


VECTOR_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------- 各分块 ----------------

@dataclass
class SystemConfig:
    wavelength_nm: float
    pixel_size_nm: float
    mask_size: int
    # YAML 里写成 list（如 [1, 0]），保持 list 即可，下游需要 tensor 时再转
    jones_matrix: List[float] = field(default_factory=lambda: [1.0, 0.0])


@dataclass
class MaskConfig:
    mask_name: str
    mask_dir: str
    normalize: bool = True

    @property
    def mask_path(self) -> Path:
        return Path(self.mask_dir) / self.mask_name


@dataclass
class PupilConfig:
    na: float
    n_image: float
    defocus_nm: float = 0.0
    aberration: Dict[Union[int, str], float] = field(default_factory=dict)


@dataclass
class SourceConfig:
    type: str
    sigma_in: float
    sigma_out: float
    smoothing: float = 0.0
    upsample: int = 10


@dataclass
class SigmoidResistParams:
    threshold: float = 0.25
    alpha: float = 85.0


@dataclass
class ResistConfig:
    model: str = "sigmoid"
    sigmoid_params: SigmoidResistParams = field(default_factory=SigmoidResistParams)


# ---------------- 顶层 ----------------

@dataclass
class SimulationConfig:
    system: SystemConfig
    mask: MaskConfig
    pupil: PupilConfig
    source: SourceConfig
    resist: ResistConfig

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "SimulationConfig":
        """从 YAML 文件构造 SimulationConfig。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls(
            system=cls._parse_system(raw.get("system", {})),
            mask=cls._parse_mask(raw.get("mask", {})),
            pupil=cls._parse_pupil(raw.get("pupil", {})),
            source=cls._parse_source(raw.get("source", {})),
            resist=cls._parse_resist(raw.get("resist", {})),
        )

    # ---------- 各分块解析（容忍字段命名差异 / 多余字段） ----------

    @staticmethod
    def _parse_system(d: Dict[str, Any]) -> SystemConfig:
        # 兼容 pixel_size / pixel_size_nm 两种命名
        if "pixel_size_nm" not in d and "pixel_size" in d:
            d = {**d, "pixel_size_nm": d["pixel_size"]}
            d.pop("pixel_size", None)
        return SystemConfig(
            wavelength_nm=float(d["wavelength_nm"]),
            pixel_size_nm=float(d["pixel_size_nm"]),
            mask_size=int(d["mask_size"]),
            jones_matrix=list(d.get("jones_matrix", [1.0, 0.0])),
        )

    @staticmethod
    def _parse_mask(d: Dict[str, Any]) -> MaskConfig:
        mask_dir = Path(str(d.get("mask_dir", ""))).expanduser()
        if not mask_dir.is_absolute():
            mask_dir = VECTOR_PROJECT_ROOT / mask_dir
        return MaskConfig(
            mask_name=str(d.get("mask_name", "")),
            mask_dir=str(mask_dir.resolve()),
            normalize=bool(d.get("normalize", True)),
        )

    @staticmethod
    def _parse_pupil(d: Dict[str, Any]) -> PupilConfig:
        # 兼容拼写错误：defcous_nm -> defocus_nm
        defocus = d.get("defocus_nm", d.get("defcous_nm", 0.0))
        return PupilConfig(
            na=float(d["na"]),
            n_image=float(d["n_image"]),
            defocus_nm=float(defocus),
            aberration=dict(d.get("aberration", {}) or {}),
        )

    @staticmethod
    def _parse_source(d: Dict[str, Any]) -> SourceConfig:
        return SourceConfig(
            type=str(d.get("type", "annular")),
            sigma_in=float(d.get("sigma_in", 0.0)),
            sigma_out=float(d.get("sigma_out", 1.0)),
            smoothing=float(d.get("smoothing", 0.0)),
            upsample=int(d.get("upsample", 10)),
        )

    @staticmethod
    def _parse_resist(d: Dict[str, Any]) -> ResistConfig:
        model = str(d.get("model", "sigmoid")).strip().lower()
        sigmoid_raw: Dict[str, Any] = dict(d.get("sigmoid", {}) or {})
        return ResistConfig(
            model=model,
            sigmoid_params=SigmoidResistParams(
                threshold=float(sigmoid_raw.get("threshold", 0.25)),
                alpha=float(sigmoid_raw.get("alpha", 85.0)),
            ),
        )
