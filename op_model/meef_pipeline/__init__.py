"""MEEF Pipeline (独立管线).

把 LSM 优化后的版图作为输入, 自洽地完成:
    1) 拆分 主图形 / SRAF
    2) 主图形 CP 提取 (沿用 extract_mask_control_points)
    3) SRAF CP 提取 (每条 SRAF 闭合外轮廓 + 弧长等距采样)
    4) SRAF B-spline 闭合拟合 + MSAA 渲染 (一次性, 得到常量 sraf_mask)
    5) 装配成与 set_meef.MEEF 鸭子兼容的接口, 直接喂给 MEEF_Optimizer.run()

不依赖 set_sraf / sraf_width_opt 那一套宽度优化项目.
"""

from .setup import MEEFPipelineSetup
from .console_logger import tee_console_to_file

__all__ = ["MEEFPipelineSetup", "tee_console_to_file"]
