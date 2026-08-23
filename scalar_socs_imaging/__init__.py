"""
scalar_socs_imaging — 独立的标量 SOCS 加速光刻成像系统

双后端实现：
  - numpy  版: 纯 CPU 数值仿真，继承现有代码库的物理模型
  - pytorch 版: GPU 友好、支持 autograd、支持批量仿真

用法:
    from scalar_socs_imaging import (
        OpticalConfig, ScalarImagingNumpy, ScalarImagingTorch
    )
"""

from .config import OpticalConfig
from .numpy_backend import ScalarImagingNumpy
from .torch_backend import ScalarImagingTorch, ScalarSOCSImagingTorch
from .visualization import (
    plot_imaging_result,
    plot_batch_results,
    plot_socs_kernels,
    plot_energy_spectrum,
)
