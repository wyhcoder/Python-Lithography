"""项目目录的唯一来源。

业务代码不要再依赖当前工作目录（``Path.cwd()``），而是从本文件解析
输入数据、配置和输出目录。这样从项目根目录、IDE 或 ``examples/`` 直接
启动脚本时，访问的都是同一份文件。
"""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PATTERN_DATA_DIR = DATA_DIR / "patterns"
DEBUG_DATA_DIR = DATA_DIR / "debug_dumps"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OPC_OUTPUT_DIR = OUTPUTS_DIR / "opc"
DIAGNOSTICS_OUTPUT_DIR = OUTPUTS_DIR / "diagnostics"

SCALAR_CONFIG_PATH = PROJECT_ROOT / "litho_model" / "config.yaml"
VECTOR_PROJECT_DIR = PROJECT_ROOT / "projects" / "vector_imaging"


def project_path(path: str | Path) -> Path:
    """把配置中的相对路径解析为相对于项目根目录的绝对路径。"""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "PATTERN_DATA_DIR",
    "DEBUG_DATA_DIR",
    "OUTPUTS_DIR",
    "OPC_OUTPUT_DIR",
    "DIAGNOSTICS_OUTPUT_DIR",
    "SCALAR_CONFIG_PATH",
    "VECTOR_PROJECT_DIR",
    "project_path",
]
