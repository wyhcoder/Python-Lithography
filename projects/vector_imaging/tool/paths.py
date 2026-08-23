"""项目路径工具。

集中管理"项目根目录"和"输出目录"的解析逻辑，避免各个脚本硬编码绝对路径。
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """返回项目根目录（即包含 tool/ 与 litho_model/ 的目录）。"""
    # tool/paths.py -> tool/ -> project root
    return Path(__file__).resolve().parent.parent


def output_dir(sub: str | None = None, ensure: bool = True) -> Path:
    """
    返回输出目录 `<project_root>/output/`，可选择再进入子目录 sub。
    ensure=True 时自动创建目录。
    """
    out = project_root() / "output"
    if sub:
        out = out / sub
    if ensure:
        out.mkdir(parents=True, exist_ok=True)
    return out


def output_path(filename: str, sub: str | None = None) -> str:
    """返回输出文件的绝对路径字符串（自动创建上级目录）。"""
    return str(output_dir(sub=sub) / filename)


__all__ = ["project_root", "output_dir", "output_path"]
